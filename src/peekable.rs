//! Since [Peekable] in Rust advances the iterator,
//! I can't use it for peeking tokens.
//! I haven't found a better way than implementing a wrapper
//! that allows me to peek...

use std::collections::VecDeque;

use crate::lexer::Token;

pub struct Tokens<I>
where
    I: Iterator<Item = Token>,
{
    queue: VecDeque<Token>,
    inner: I,
}

impl<I> Tokens<I>
where
    I: Iterator<Item = Token>,
{
    pub fn new(inner: I) -> Self {
        Self {
            queue: VecDeque::new(),
            inner,
        }
    }
}

/// Wondering why std doesn't have such a trait.
pub trait Peekable: Iterator {
    fn peek(&mut self) -> Option<Self::Item>;
}

impl<I> Peekable for Tokens<I>
where
    I: Iterator<Item = Token>,
{
    fn peek(&mut self) -> Option<Token> {
        // something in the queue
        if let Some(token) = self.queue.front() {
            Some(token.clone())
        } else {
            // otherwise get from iterator and store in queue
            if let Some(token) = self.inner.next() {
                self.queue.push_back(token.clone());
                Some(token)
            } else {
                None
            }
        }
    }
}

impl<I> Iterator for Tokens<I>
where
    I: Iterator<Item = Token>,
{
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(token) = self.queue.pop_front() {
            Some(token)
        } else {
            self.inner.next()
        }
    }
}
