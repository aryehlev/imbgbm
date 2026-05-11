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
    #[inline]
    pub fn add(&mut self, g: f32, h: f32, is_pos: bool) {
        self.sum_g += g;
        self.sum_h += h;
        self.count += 1;
        if is_pos {
            self.sum_g_pos += g;
            self.sum_h_pos += h;
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

    /// Accumulate one example into this histogram.
    #[inline]
    pub fn accumulate(&mut self, bin: u8, g: f32, h: f32, is_pos: bool) {
        self.bins[bin as usize].add(g, h, is_pos);
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

    /// Compute the histogram for the complement (parent - this), used for histogram subtraction.
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

/// Build per-feature histograms for a set of row indices.
pub fn build_histograms(
    bins_col_major: &[Vec<u8>],
    n_bins_per_col: &[usize],
    labels: &[f32],
    gradients: &[f32],
    hessians: &[f32],
    indices: &[u32],
) -> Vec<Histogram> {
    let n_cols = bins_col_major.len();
    let mut histograms: Vec<Histogram> = (0..n_cols)
        .map(|c| Histogram::new(c, n_bins_per_col[c]))
        .collect();

    for &row in indices {
        let r = row as usize;
        let g = gradients[r];
        let h = hessians[r];
        let is_pos = labels[r] > 0.5;
        for (col, hist) in histograms.iter_mut().enumerate() {
            hist.accumulate(bins_col_major[col][r], g, h, is_pos);
        }
    }

    histograms
}
