//! Positive-mode preservation via k-means clustering of labeled positives.
//!
//! Real-world seed positives are often multimodal — e.g. an advertiser's
//! historical converters may consist of finance users, travel users, gamers,
//! and parents, where the dominant mode (say, finance) might be 10× the size
//! of the rare modes. A vanilla GBDT trained on these positives plus the
//! unlabeled pool will preferentially fit the dominant mode and effectively
//! ignore the smaller ones, because their gradient contribution per round is
//! tiny.
//!
//! This module provides a lightweight k-means routine that produces a per-row
//! cluster ID vector suitable for use as a `RowMetadata` segment dimension.
//! Combined with a `SegmentQuota { min_per_segment: m, .. }` constraint on the
//! `AdaptiveSampler`, this enforces that every mode contributes at least `m`
//! examples to every tree, protecting rare modes from being drowned out.
//!
//! Unlabeled rows receive the sentinel cluster ID `NO_CLUSTER = u32::MAX`.
//! Downstream samplers should treat that ID as "not a positive mode" — e.g.
//! by exempting it from `min_per_segment` constraints.

use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

/// Sentinel cluster ID assigned to unlabeled (and missing) rows.
pub const NO_CLUSTER: u32 = u32::MAX;

/// k-means++ initialisation: pick centroids that are far apart in feature space.
fn kmeans_pp_init(points: &[&[f32]], k: usize, rng: &mut SmallRng) -> Vec<Vec<f32>> {
    let n = points.len();
    let d = points[0].len();
    let mut centroids = Vec::with_capacity(k);

    // First centroid: uniform random pick.
    let first = rng.gen_range(0..n);
    centroids.push(points[first].to_vec());

    let mut sq_dist = vec![f32::INFINITY; n];
    for _ in 1..k {
        // Refresh nearest-centroid squared distance for every point.
        let last = centroids.last().unwrap();
        for i in 0..n {
            let dist = euclidean_sq(points[i], last);
            if dist < sq_dist[i] {
                sq_dist[i] = dist;
            }
        }
        // Sample next centroid weighted by squared distance.
        let total: f32 = sq_dist.iter().sum();
        if total <= 0.0 {
            // All points coincide with an existing centroid; fall back to uniform.
            centroids.push(points[rng.gen_range(0..n)].to_vec());
            continue;
        }
        let mut target = rng.gen::<f32>() * total;
        let mut chosen = n - 1;
        for (i, &w) in sq_dist.iter().enumerate() {
            if target <= w { chosen = i; break; }
            target -= w;
        }
        centroids.push(points[chosen].to_vec());
        let _ = d;
    }
    centroids
}

#[inline]
fn euclidean_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| { let d = x - y; d * d })
        .sum()
}

/// One k-means pass: assignment + centroid update.
fn kmeans_iterate(points: &[&[f32]], centroids: &mut Vec<Vec<f32>>) -> bool {
    let k = centroids.len();
    let d = centroids[0].len();
    let n = points.len();

    let mut new_centroids = vec![vec![0.0f32; d]; k];
    let mut counts = vec![0u32; k];
    let mut assignment = vec![0usize; n];

    for (i, p) in points.iter().enumerate() {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for (c, ctr) in centroids.iter().enumerate() {
            let dist = euclidean_sq(p, ctr);
            if dist < best_d { best_d = dist; best = c; }
        }
        assignment[i] = best;
        counts[best] += 1;
        for j in 0..d {
            new_centroids[best][j] += p[j];
        }
    }

    let mut changed = false;
    for c in 0..k {
        if counts[c] == 0 { continue; } // leave empty cluster centroid as-is
        let inv = 1.0 / counts[c] as f32;
        for j in 0..d {
            new_centroids[c][j] *= inv;
            if (new_centroids[c][j] - centroids[c][j]).abs() > 1e-5 {
                changed = true;
            }
        }
        centroids[c] = new_centroids[c].clone();
    }
    changed
}

/// Cluster only the labeled positives among `rows` into `k` modes.
///
/// Returns a per-row cluster ID vector of length `rows.len()`:
///   - Positive rows: their assigned cluster ID, in `0..k`.
///   - Other rows: `NO_CLUSTER` (= `u32::MAX`).
///
/// Uses k-means++ initialisation and Lloyd iterations until convergence
/// (capped at `max_iter`). For very large positive sets a mini-batch refinement
/// would be more efficient, but seed-positive sets are typically small.
pub fn cluster_positives(
    rows: &[&[f32]],
    labels: &[f32],
    k: usize,
    seed: u64,
    max_iter: usize,
) -> Vec<u32> {
    assert!(rows.len() == labels.len(), "rows / labels length mismatch");
    assert!(k >= 1, "k must be >= 1");

    let pos_indices: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter(|(_, &y)| y > 0.5)
        .map(|(i, _)| i)
        .collect();

    let mut out = vec![NO_CLUSTER; rows.len()];

    if pos_indices.is_empty() {
        return out;
    }

    // Degenerate case: fewer positives than requested clusters → one per positive.
    let effective_k = k.min(pos_indices.len()).max(1);

    let pos_rows: Vec<&[f32]> = pos_indices.iter().map(|&i| rows[i]).collect();
    let mut rng = SmallRng::seed_from_u64(seed ^ 0xc7d8_9f29);

    let mut centroids = kmeans_pp_init(&pos_rows, effective_k, &mut rng);
    for _ in 0..max_iter {
        if !kmeans_iterate(&pos_rows, &mut centroids) {
            break;
        }
    }

    // Final assignment pass.
    for (j, &orig_idx) in pos_indices.iter().enumerate() {
        let mut best = 0u32;
        let mut best_d = f32::INFINITY;
        for (c, ctr) in centroids.iter().enumerate() {
            let dist = euclidean_sq(pos_rows[j], ctr);
            if dist < best_d { best_d = dist; best = c as u32; }
        }
        out[orig_idx] = best;
    }
    out
}

/// Report the per-cluster positive counts. Useful for sizing `min_per_segment`.
pub fn cluster_sizes(cluster_ids: &[u32], k: usize) -> Vec<u32> {
    let mut counts = vec![0u32; k];
    for &c in cluster_ids {
        if c != NO_CLUSTER && (c as usize) < k {
            counts[c as usize] += 1;
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clusters_two_obvious_modes() {
        // Two well-separated Gaussian blobs in 2-D.
        let mode_a: Vec<Vec<f32>> = (0..40)
            .map(|i| vec![0.0 + 0.01 * i as f32, 0.0])
            .collect();
        let mode_b: Vec<Vec<f32>> = (0..40)
            .map(|i| vec![10.0 + 0.01 * i as f32, 10.0])
            .collect();

        let mut rows: Vec<Vec<f32>> = mode_a.into_iter().chain(mode_b).collect();
        // Add 20 unlabeled junk rows.
        for _ in 0..20 { rows.push(vec![5.0, 5.0]); }
        let labels: Vec<f32> = std::iter::repeat(1.0).take(80).chain(std::iter::repeat(0.0).take(20)).collect();
        let row_refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();

        let ids = cluster_positives(&row_refs, &labels, 2, 7, 30);
        // Unlabeled rows must be NO_CLUSTER.
        for i in 80..100 { assert_eq!(ids[i], NO_CLUSTER); }
        // The 80 positives split into the two clusters; expect ~40 each.
        let sizes = cluster_sizes(&ids[..80], 2);
        assert!(sizes[0] >= 35 && sizes[0] <= 45, "uneven split: {sizes:?}");
        assert!(sizes[1] >= 35 && sizes[1] <= 45, "uneven split: {sizes:?}");
    }

    #[test]
    fn empty_positives_no_panic() {
        let rows: Vec<Vec<f32>> = (0..5).map(|_| vec![0.0, 0.0]).collect();
        let row_refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
        let labels = vec![0.0; 5];
        let ids = cluster_positives(&row_refs, &labels, 3, 0, 5);
        assert!(ids.iter().all(|&c| c == NO_CLUSTER));
    }
}
