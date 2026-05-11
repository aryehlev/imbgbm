/// Maximum number of discrete bins per feature.
pub const MAX_BINS: usize = 256;

/// Raw feature matrix + labels, column-major (features[col][row]).
#[derive(Clone)]
pub struct Dataset {
    /// features[col][row]
    pub features: Vec<Vec<f32>>,
    pub labels: Vec<f32>,
    pub n_rows: usize,
    pub n_cols: usize,
}

impl Dataset {
    /// Construct from column-major storage.
    pub fn new(features: Vec<Vec<f32>>, labels: Vec<f32>) -> Self {
        let n_cols = features.len();
        let n_rows = labels.len();
        for col in &features {
            assert_eq!(col.len(), n_rows, "column length mismatch");
        }
        Dataset { features, labels, n_rows, n_cols }
    }

    /// Convenience: construct from row-major slice.
    pub fn from_rows(rows: &[&[f32]], labels: Vec<f32>) -> Self {
        let n_rows = rows.len();
        assert_eq!(n_rows, labels.len());
        let n_cols = if n_rows > 0 { rows[0].len() } else { 0 };
        let mut features = vec![vec![0.0f32; n_rows]; n_cols];
        for (r, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), n_cols, "row length mismatch");
            for (c, &v) in row.iter().enumerate() {
                features[c][r] = v;
            }
        }
        Dataset { features, labels, n_rows, n_cols }
    }

    /// Return a view of the dataset restricted to the given row indices.
    pub fn subset(&self, indices: &[u32]) -> Dataset {
        let n_rows = indices.len();
        let features = self
            .features
            .iter()
            .map(|col| indices.iter().map(|&i| col[i as usize]).collect())
            .collect();
        let labels = indices.iter().map(|&i| self.labels[i as usize]).collect();
        Dataset { features, labels, n_rows, n_cols: self.n_cols }
    }
}

/// Feature matrix with values pre-discretised into bins, column-major.
///
/// Bin indices are `u8` (0 .. n_bins-1).  Each feature may have a different
/// effective number of bins when there are fewer unique values than `n_bins`.
#[derive(Clone)]
pub struct BinnedDataset {
    /// bins[col][row] = bin index
    pub bins: Vec<Vec<u8>>,
    /// bin_thresholds[col][b] = upper boundary of bin b (last entry = f32::INFINITY)
    pub bin_thresholds: Vec<Vec<f32>>,
    pub labels: Vec<f32>,
    pub n_rows: usize,
    pub n_cols: usize,
    /// Requested maximum bin count (actual may be lower per feature).
    pub n_bins: usize,
}

impl BinnedDataset {
    /// Quantile-bin a `Dataset` into at most `n_bins` equal-frequency bins per feature.
    pub fn from_dataset(dataset: &Dataset, n_bins: usize) -> Self {
        assert!(n_bins >= 2 && n_bins <= MAX_BINS);
        let n_rows = dataset.n_rows;
        let n_cols = dataset.n_cols;

        let mut bins_out = Vec::with_capacity(n_cols);
        let mut thresholds_out = Vec::with_capacity(n_cols);

        for col in 0..n_cols {
            let vals = &dataset.features[col];

            // Collect finite values, sort, deduplicate.
            let mut sorted: Vec<f32> =
                vals.iter().copied().filter(|v| v.is_finite()).collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted.dedup();

            let n_unique = sorted.len();
            let actual_bins = n_bins.min(n_unique).max(1);

            // Quantile cut-points: threshold[b] = sorted[ceil((b+1)*n_unique/actual_bins) - 1]
            let mut thresholds: Vec<f32> = (0..actual_bins)
                .map(|b| {
                    let idx =
                        (((b + 1) * n_unique + actual_bins - 1) / actual_bins).min(n_unique) - 1;
                    sorted[idx]
                })
                .collect();
            // Last threshold = +inf so every value falls into a valid bin.
            if let Some(last) = thresholds.last_mut() {
                *last = f32::INFINITY;
            }

            // Map each raw value to a bin index via binary search.
            let col_bins: Vec<u8> = vals
                .iter()
                .map(|&v| {
                    let b = thresholds.partition_point(|&t| t < v);
                    b.min(actual_bins - 1) as u8
                })
                .collect();

            bins_out.push(col_bins);
            thresholds_out.push(thresholds);
        }

        BinnedDataset {
            bins: bins_out,
            bin_thresholds: thresholds_out,
            labels: dataset.labels.clone(),
            n_rows,
            n_cols,
            n_bins,
        }
    }

    /// Number of effective bins for a given column.
    pub fn n_bins_for_col(&self, col: usize) -> usize {
        self.bin_thresholds[col].len()
    }

    /// Fraction of positive labels (y == 1.0).
    pub fn class_prior(&self) -> f32 {
        let pos = self.labels.iter().filter(|&&y| y > 0.5).count();
        pos as f32 / self.n_rows.max(1) as f32
    }
}
