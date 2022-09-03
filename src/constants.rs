//
// Constants
//

pub const NUM_REGISTERS: usize = kimchi::circuits::wires::COLUMNS;

//
// Aliases
//

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Span(pub usize, pub usize);

impl Span {
    pub fn start(&self) -> usize {
        self.0
    }

    pub fn len(&self) -> usize {
        self.1
    }

    pub fn is_empty(&self) -> bool {
        self.1 == 0
    }

    pub fn end(&self) -> usize {
        self.0 + self.1
    }

    pub fn merge_with(self, other: Span) -> Span {
        let real_len = other.end() - self.start();
        Span(self.0, real_len)
    }
}

impl From<Span> for miette::SourceSpan {
    fn from(span: Span) -> Self {
        Self::new(span.0.into(), span.1.into())
    }
}
