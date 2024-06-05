//! Since [`std::iter::Peekable`] in Rust advances the iterator,
//! I can't use it for peeking tokens.
//! I haven't found a better way than implementing a wrapper
//! that allows me to peek...

use std::vec::IntoIter;

use crate::{
    error::{ErrorKind, Result},
    lexer::{Token, TokenKind},
    parser::{types::Ident, ParserCtx},
};

#[derive(Debug)]
pub struct Tokens {
    pub peeked: Option<Token>,
    inner: IntoIter<Token>,
}

impl Tokens {
    #[must_use]
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            peeked: None,
            inner: tokens.into_iter(),
        }
    }

    /// Peeks into the next token without advancing the iterator.
    pub fn peek(&mut self) -> Option<Token> {
        // something in the peeked
        if let Some(token) = &self.peeked {
            Some(token.clone())
        } else {
            // otherwise get from iterator and store in peeked
            let token = self.inner.next();
            self.peeked.clone_from(&token);
            token
        }
    }

    /// Like `next()` except that it also stores the last seen token in the given context
    /// (useful for debugging)
    pub fn bump(&mut self, ctx: &mut ParserCtx) -> Option<Token> {
        if let Some(token) = self.peeked.take() {
            ctx.last_token = Some(token.clone());
            Some(token)
        } else {
            let token = self.inner.next();
            if token.is_some() {
                ctx.last_token.clone_from(&token);
            }
            token
        }
    }
    /// Like [`Self::bump`] but errors with `err` pointing to the latest token
    pub fn bump_err(&mut self, ctx: &mut ParserCtx, err: ErrorKind) -> Result<Token> {
        self.bump(ctx)
            .ok_or_else(|| ctx.error(err, ctx.last_span()))
    }

    /// Like [`Self::bump`] but errors if the token is not `typ`
    pub fn bump_expected(&mut self, ctx: &mut ParserCtx, typ: TokenKind) -> Result<Token> {
        let token = self.bump_err(ctx, ErrorKind::MissingToken)?;
        if token.kind == typ {
            Ok(token)
        } else {
            let span = ctx.last_span();
            Err(ctx.error(ErrorKind::ExpectedToken(typ), span))
        }
    }

    pub fn bump_ident(&mut self, ctx: &mut ParserCtx, kind: ErrorKind) -> Result<Ident> {
        match self.bump(ctx) {
            Some(Token {
                kind: TokenKind::Identifier(value),
                span,
            }) => Ok(Ident { value, span }),
            Some(token) => Err(ctx.error(kind, token.span)),
            None => Err(ctx.error(kind, ctx.last_span())),
        }
    }
}
