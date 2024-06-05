//
// Constants
//

use serde::{Deserialize, Serialize};

//
// Span stuff (this should probably move from here)
//

#[derive(
    Default, Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize, Serialize,
)]
pub struct Span {
    pub filename_id: usize,
    pub start: usize,
    pub len: usize,
}

impl From<Span> for miette::SourceSpan {
    fn from(span: Span) -> Self {
        Self::new(span.start.into(), span.len.into())
    }
}

impl Span {
    #[must_use]
    pub fn new(filename_id: usize, start: usize, len: usize) -> Self {
        Self {
            filename_id,
            start,
            len,
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.start == 0
    }

    #[must_use]
    pub fn end(&self) -> usize {
        self.start + self.len
    }

    #[must_use]
    pub fn merge_with(self, other: Self) -> Self {
        assert_eq!(self.filename_id, other.filename_id);
        let real_len = other.end() - self.start;
        Self::new(self.filename_id, self.start, real_len)
    }
}
