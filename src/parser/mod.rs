use miette::NamedSource;

use crate::{
    constants::Span,
    error::{Error, ErrorKind, Result},
    lexer::{Keyword, Token, TokenKind},
    tokens::Tokens,
};

use self::types::{Const, Function, Root, RootKind, Struct, UsePath};

pub mod types;

//~
//~ # Grammar
//~
//~ ## Notation
//~
//~ We use a notation similar to the Backus-Naur Form (BNF)
//~ to describe the grammar:
//~
//~ <pre>
//~ land := city "|"
//~  ^        ^   ^
//~  |        |  terminal: a token
//~  |        |
//~  |      another non-terminal
//~  |
//~  non-terminal: definition of a piece of code
//~
//~ city := [ sign ] "," { house }
//~         ^            ^
//~         optional     |
//~                     0r or more houses
//~
//~ sign := /a-zA-Z_/
//~         ^
//~         regex-style definition
//~ </pre>
//~

//
// Context
//

/// A context for the parser.
#[derive(Debug, Default)]
pub struct ParserCtx {
    /// A counter used to uniquely identify different nodes in the AST.
    pub node_id: usize,

    /// Used mainly for error reporting,
    /// when we don't have a token to read
    // TODO: replace with `(usize::MAX, usize::MAX)`?
    pub last_token: Option<Token>,

    /// The file we're parsing
    filename: String,

    /// The code we're parsing
    code: String,
}

impl ParserCtx {
    pub fn new(filename: String, code: String) -> Self {
        Self {
            node_id: 0,
            last_token: None,
            filename,
            code,
        }
    }

    pub fn error(&self, kind: ErrorKind, span: Span) -> Error {
        Error::new_with_code(kind, &self.filename, self.code.clone(), span)
    }

    /// Returns a new unique node id.
    pub fn next_node_id(&mut self) -> usize {
        self.node_id += 1;
        self.node_id
    }

    // TODO: I think I don't need this, I should always be able to use the last token I read if I don't see anything, otherwise maybe just write -1 to say "EOF"
    pub fn last_span(&self) -> Span {
        let span = self
            .last_token
            .as_ref()
            .map(|token| token.span)
            .unwrap_or(Span(0, 0));
        Span(span.end(), 0)
    }
}

//
// AST
//

#[derive(Debug, Default)]
pub struct AST(pub Vec<Root>);

impl AST {
    pub fn parse(filename: &str, code: &str, tokens: Tokens) -> Result<AST> {
        Self::parse_inner(tokens).map_err(|mut e| {
            e.src = NamedSource::new(filename, code.to_string());
            e
        })
    }

    fn parse_inner(mut tokens: Tokens) -> Result<AST> {
        let mut ast = vec![];
        let ctx = &mut ParserCtx::default();

        // use statements must appear first
        let mut function_observed = false;

        while let Some(token) = tokens.bump(ctx) {
            match &token.kind {
                // `use crypto::poseidon;`
                TokenKind::Keyword(Keyword::Use) => {
                    if function_observed {
                        return Err(ctx.error(ErrorKind::UseAfterFn, token.span));
                    }

                    let path = UsePath::parse(ctx, &mut tokens)?;
                    ast.push(Root {
                        kind: RootKind::Use(path),
                        span: token.span,
                    });

                    // end of line
                    let next_token = tokens.bump(ctx);
                    if !matches!(
                        next_token,
                        Some(Token {
                            kind: TokenKind::SemiColon,
                            ..
                        })
                    ) {
                        return Err(ctx.error(ErrorKind::InvalidEndOfLine, token.span));
                    }
                }

                // `const FOO = 42;`
                TokenKind::Keyword(Keyword::Const) => {
                    let cst = Const::parse(ctx, &mut tokens)?;

                    ast.push(Root {
                        kind: RootKind::Const(cst),
                        span: token.span,
                    });
                }

                // `fn main() { }`
                TokenKind::Keyword(Keyword::Fn) => {
                    function_observed = true;

                    let func = Function::parse(ctx, &mut tokens)?;
                    ast.push(Root {
                        kind: RootKind::Function(func),
                        span: token.span,
                    });
                }

                // `struct Foo { a: Field, b: Field }`
                TokenKind::Keyword(Keyword::Struct) => {
                    let s = Struct::parse(ctx, &mut tokens)?;
                    ast.push(Root {
                        kind: RootKind::Struct(s),
                        span: token.span,
                    });
                }

                // `// some comment`
                TokenKind::Comment(comment) => {
                    ast.push(Root {
                        kind: RootKind::Comment(comment.clone()),
                        span: token.span,
                    });
                }

                // unrecognized
                _ => {
                    return Err(ctx.error(ErrorKind::InvalidToken, token.span));
                }
            }
        }

        Ok(Self(ast))
    }
}

//
// Tests
//
#[cfg(test)]
mod tests {
    use crate::parser::types::Stmt;

    use super::*;

    #[test]
    fn fn_signature() {
        let code = r#"main(pub public_input: [Fel; 3], private_input: [Fel; 3]) -> [Fel; 3] { return public_input; }"#;
        let tokens = &mut Token::parse("example.no", code).unwrap();
        let ctx = &mut ParserCtx::default();
        let parsed = Function::parse(ctx, tokens).unwrap();
        println!("{:?}", parsed);
    }

    #[test]
    fn statement_assign() {
        let code = r#"let digest = poseidon(private_input);"#;
        let tokens = &mut Token::parse("example.no", code).unwrap();
        let ctx = &mut ParserCtx::default();
        let parsed = Stmt::parse(ctx, tokens).unwrap();
        println!("{:?}", parsed);
    }

    #[test]
    fn statement_assert() {
        let code = r#"assert(digest == public_input);"#;
        let tokens = &mut Token::parse("example.no", code).unwrap();
        let ctx = &mut ParserCtx::default();
        let parsed = Stmt::parse(ctx, tokens).unwrap();
        println!("{:?}", parsed);
    }
}
