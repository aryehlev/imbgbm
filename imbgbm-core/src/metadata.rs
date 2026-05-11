/// Per-row contextual metadata used by samplers and calibrators.
///
/// All arrays must have length equal to `n_rows` in the associated `Dataset`,
/// or be empty (meaning "not provided for this dimension").
#[derive(Clone, Default)]
pub struct RowMetadata {
    /// Categorical segment IDs, one slice per dimension.
    ///
    /// `segments[dim][row]` is the segment ID for row `row` along dimension
    /// `dim`.  Typical dimensions: advertiser, campaign, publisher, geo bucket,
    /// hour-of-day, device type, supply-path.
    pub segments: Vec<Vec<u32>>,

    /// Human-readable names for each segment dimension (for logging).
    pub segment_names: Vec<String>,

    /// Unix timestamps (seconds) for each row.
    ///
    /// Required when using `FoldStrategy::Temporal` in training.
    pub timestamps: Option<Vec<i64>>,
}

impl RowMetadata {
    /// Construct metadata with no segments and no timestamps.
    pub fn empty() -> Self {
        RowMetadata::default()
    }

    /// Add a segment dimension.
    ///
    /// `ids` must have the same length as all other segment arrays.
    pub fn with_segment(mut self, name: impl Into<String>, ids: Vec<u32>) -> Self {
        self.segment_names.push(name.into());
        self.segments.push(ids);
        self
    }

    /// Attach per-row Unix timestamps.
    pub fn with_timestamps(mut self, ts: Vec<i64>) -> Self {
        self.timestamps = Some(ts);
        self
    }

    /// Number of segment dimensions.
    pub fn n_segment_dims(&self) -> usize {
        self.segments.len()
    }

    /// Segment ID for a given row and dimension.  Returns `None` if the
    /// dimension or row index is out of bounds.
    pub fn segment_id(&self, dim: usize, row: usize) -> Option<u32> {
        self.segments.get(dim)?.get(row).copied()
    }

    /// Timestamp for a given row.  Returns `None` if timestamps are absent.
    pub fn timestamp(&self, row: usize) -> Option<i64> {
        self.timestamps.as_ref()?.get(row).copied()
    }
}
