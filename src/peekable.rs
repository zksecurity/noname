//! Since [Peekable] in Rust advances the iterator,
//! I can't use it for peeking tokens.
//! I haven't found a better way than implementing a wrapper
//! that allows me to peek...

use std::collections::VecDeque;

use crate::{
    error::{Error, ErrorTy},
    lexer::{Token, TokenType},
    parser::ParserCtx,
};

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

    /// Like next() except that it also stores the last seen token in the given context
    /// (useful for debugging)
    fn bump(&mut self, ctx: &mut ParserCtx) -> Option<Token>;

    /// Returns the next token or errors with `err` pointing to the latest token
    #[must_use]
    fn bump_err(&mut self, ctx: &mut ParserCtx, err: ErrorTy) -> Result<Token, Error>;

    #[must_use]
    fn bump_expected(&mut self, ctx: &mut ParserCtx, typ: TokenType) -> Result<Token, Error>;
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

    fn bump(&mut self, ctx: &mut ParserCtx) -> Option<Token> {
        if let Some(token) = self.next() {
            ctx.last_token = Some(token.clone());
            Some(token)
        } else {
            None
        }
    }

    fn bump_err(&mut self, ctx: &mut ParserCtx, err: ErrorTy) -> Result<Token, Error> {
        self.bump(ctx).ok_or(Error {
            error: err,
            span: ctx.last_span(),
        })
    }

    fn bump_expected(&mut self, ctx: &mut ParserCtx, typ: TokenType) -> Result<Token, Error> {
        let token = self.bump_err(ctx, ErrorTy::MissingToken)?;
        if token.typ == typ {
            Ok(token)
        } else {
            Err(Error {
                error: ErrorTy::ExpectedToken(typ),
                span: ctx.last_span(),
            })
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
