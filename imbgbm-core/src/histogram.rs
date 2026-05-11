/// Per-bin gradient/hessian accumulator, optionally split by class.
#[derive(Clone, Default)]
pub struct BinStats {
    pub sum_g: f32,
    pub sum_h: f32,
    pub count: u32,
    /// Sum of gradients for positive-class (y == 1) examples only.
    pub sum_g_pos: f32,
    pub sum_h_pos: f32,
    pub count_pos: u32,
}

impl BinStats {
    /// Accumulate one example.  `w` is the IPC weight (1/π_i); use 1.0 for
    /// uniform/unweighted accumulation.
    #[inline]
    pub fn add(&mut self, g: f32, h: f32, w: f32, is_pos: bool) {
        self.sum_g += g * w;
        self.sum_h += h * w;
        self.count += 1;
        if is_pos {
            self.sum_g_pos += g * w;
            self.sum_h_pos += h * w;
            self.count_pos += 1;
        }
    }

    #[inline]
    pub fn subtract(&self, other: &BinStats) -> BinStats {
        BinStats {
            sum_g: self.sum_g - other.sum_g,
            sum_h: self.sum_h - other.sum_h,
            count: self.count - other.count,
            sum_g_pos: self.sum_g_pos - other.sum_g_pos,
            sum_h_pos: self.sum_h_pos - other.sum_h_pos,
            count_pos: self.count_pos - other.count_pos,
        }
    }
}

/// Gradient/hessian histogram for a single feature.
#[derive(Clone)]
pub struct Histogram {
    pub bins: Vec<BinStats>,
    pub feature: usize,
}

impl Histogram {
    pub fn new(feature: usize, n_bins: usize) -> Self {
        Histogram {
            bins: vec![BinStats::default(); n_bins],
            feature,
        }
    }

    /// Accumulate one example.  `w` is the IPC weight; use 1.0 if no
    /// correction is needed.
    #[inline]
    pub fn accumulate(&mut self, bin: u8, g: f32, h: f32, w: f32, is_pos: bool) {
        self.bins[bin as usize].add(g, h, w, is_pos);
    }

    /// Total stats across all bins (root node aggregate).
    pub fn total(&self) -> BinStats {
        self.bins.iter().fold(BinStats::default(), |mut acc, b| {
            acc.sum_g += b.sum_g;
            acc.sum_h += b.sum_h;
            acc.count += b.count;
            acc.sum_g_pos += b.sum_g_pos;
            acc.sum_h_pos += b.sum_h_pos;
            acc.count_pos += b.count_pos;
            acc
        })
    }

    /// Compute the complement histogram (parent − self) for histogram subtraction.
    pub fn complement(&self, parent: &Histogram) -> Histogram {
        assert_eq!(self.bins.len(), parent.bins.len());
        let bins = self
            .bins
            .iter()
            .zip(parent.bins.iter())
            .map(|(s, p)| p.subtract(s))
            .collect();
        Histogram { bins, feature: self.feature }
    }
}

/// Build per-feature histograms for the given row indices.
///
/// `ipc_weights[j]` is the inverse-probability correction weight for
/// `indices[j]` (i.e. 1/π_{indices[j]}).  Pass a slice of all-ones (or an
/// empty slice) to accumulate without correction.
pub fn build_histograms(
    bins_col_major: &[Vec<u8>],
    n_bins_per_col: &[usize],
    labels: &[f32],
    gradients: &[f32],
    hessians: &[f32],
    indices: &[u32],
    ipc_weights: &[f32],
) -> Vec<Histogram> {
    let all: Vec<usize> = (0..bins_col_major.len()).collect();
    build_histograms_for_features(
        bins_col_major, n_bins_per_col, labels,
        gradients, hessians, indices, ipc_weights, &all,
    )
}

/// Like `build_histograms` but only for the given feature indices (column subsampling).
pub fn build_histograms_for_features(
    bins_col_major: &[Vec<u8>],
    n_bins_per_col: &[usize],
    labels: &[f32],
    gradients: &[f32],
    hessians: &[f32],
    indices: &[u32],
    ipc_weights: &[f32],
    feature_indices: &[usize],
) -> Vec<Histogram> {
    let use_ipc = ipc_weights.len() == indices.len();
    let mut histograms: Vec<Histogram> = feature_indices
        .iter()
        .map(|&c| Histogram::new(c, n_bins_per_col[c]))
        .collect();
    for (j, &row) in indices.iter().enumerate() {
        let r = row as usize;
        let g = gradients[r];
        let h = hessians[r];
        let w = if use_ipc { ipc_weights[j] } else { 1.0 };
        let is_pos = labels[r] > 0.5;
        for (&col, hist) in feature_indices.iter().zip(histograms.iter_mut()) {
            hist.accumulate(bins_col_major[col][r], g, h, w, is_pos);
        }
    }
    histograms
}
