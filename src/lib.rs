//! noname project

use miette::{Diagnostic, Result, SourceSpan};

use error::{Error, ErrorTy};
use lexer::Token;

pub mod error;
pub mod lexer;
pub mod parser;

#[derive(Default, Debug)]
pub struct Context {
    offset: usize,
    inline_offset: usize,
}

#[derive(Debug)]
pub struct Path(Vec<String>);

impl Path {
    /// Parses a path from a list of tokens.
    pub fn parse_path(
        ctx: &mut Context,
        tokens: &mut impl Iterator<Item = TokenType>,
    ) -> Result<Self, Error> {
        let mut path = vec![];

        let mut tokens = tokens.peekable();
        loop {
            let module_or_leaf = tokens.next();

            match module_or_leaf {
                // a chunk of the path
                Some(TokenType::AlphaNumeric_(chunk)) => {
                    if !is_valid_module(&chunk) {
                        return Err(Error {
                            error: ErrorTy::InvalidModule,
                            span: (ctx.offset + ctx.inline_offset, 1),
                        });
                    }

                    path.push(chunk.to_string());
                    ctx.inline_offset += chunk.len();

                    // next, we expect a `::` to continue the path,
                    // or a separator, for example, `;` or `)`
                    match tokens.peek() {
                        None => {
                            return Err(Error {
                                error: ErrorTy::InvalidEndOfLine,
                                span: (ctx.offset + ctx.inline_offset, 1),
                            })
                        }
                        Some(TokenType::DoubleColon) => {
                            tokens.next();
                            ctx.inline_offset += 2;
                            continue;
                        }
                        Some(TokenType::SemiColon) => break,
                        x => {
                            dbg!(&path);
                            dbg!(x);
                            return Err(Error {
                                error: ErrorTy::InvalidToken,
                                span: (ctx.offset + ctx.inline_offset, 1),
                            });
                        }
                    }
                }
                _ => {
                    return Err(Error {
                        error: ErrorTy::InvalidPath,
                        span: (ctx.offset + ctx.inline_offset, 1),
                    });
                }
            }
        }

        Ok(Self(path))
    }
}

#[derive(Debug)]
pub enum Ty {
    Field,
    Array(Box<Self>, usize),
    Bool,
}

#[derive(Debug)]
pub enum ComparisonOp {
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Equal,
}

#[derive(Debug)]
pub enum Expression {
    Literal(String),
    Function(Function),
    Variable(String),
    Comparison(ComparisonOp, Box<Expression>, Box<Expression>),
    True,
    False,
}

#[derive(Debug)]
pub struct Function {
    pub name: String,
    pub arguments: Vec<(String, Ty)>,
    pub return_type: Option<Ty>,
    pub body: Vec<Statement>,
}

impl Function {
    pub fn parse_fn(
        ctx: &Context,
        tokens: &mut impl Iterator<Item = TokenType>,
    ) -> Result<Self, Error> {
        // parse function name
        let name = tokens.next().ok_or(Error {
            error: ErrorTy::InvalidFunctionSignature,
            span: (ctx.offset + ctx.inline_offset, 1),
        })?;

        // it's possible that there are no spaces: main(first_arg:type,etc.)
        // or a bunch of spaces: main ( first_arg : type , etc. )
        // should we enforce a single style?

        let func = Self {
            name: todo!(),
            arguments: vec![],
            return_type: todo!(),
            body: todo!(),
        };
        let name = tokens.next().ok_or(Error {
            error: ErrorTy::InvalidModule,
            span: (ctx.offset + ctx.inline_offset, 1),
        })?;

        Ok(func)
    }
}

#[derive(Debug)]
pub enum Statement {
    Assign { lhs: String, rhs: Expression },
    Assert(Expression),
    Return(Expression),
    Comment(String),
}

#[derive(Debug)]
pub enum Root {
    Use(Path),
    Function(Function),
    Comment(String),
}

pub fn parse(code: &'static str) -> Result<Vec<Root>, Error> {
    let mut root = vec![];
    let mut ctx = Context::default();

    for line in code.lines() {
        parse_line(&mut ctx, &mut root, line)?;
        ctx.offset += line.len();
        ctx.inline_offset = 0;
    }

    Ok(root)
}

pub fn parse_line(
    ctx: &mut Context,
    root: &mut Vec<Root>,
    line: &'static str,
) -> Result<(), Error> {
    // parse tokens as vec
    let tokens = TokenType::parse(ctx, line)?;

    let mut tokens = tokens.into_iter().peekable();

    // get first token
    let token = if let Some(token) = tokens.next() {
        token
    } else {
        return Ok(()); // empty line
    };

    // match special keywords
    match token {
        TokenType::Keyword(Keyword::Use) => {
            ctx.inline_offset += 4;
            let path = Path::parse_path(ctx, &mut tokens)?;
            root.push(Root::Use(path));

            // end of line
            if matches!(tokens.next(), Some(TokenType::SemiColon)) {
                ctx.inline_offset += 1;
            } else {
                return Err(Error {
                    error: ErrorTy::InvalidEndOfLine,
                    span: (ctx.offset + ctx.inline_offset, 1),
                });
            }
        }
        TokenType::Keyword(Keyword::Fn) => {
            let func = Function::parse_fn(ctx, &mut tokens)?;
            root.push(Root::Function(func));
        }
        TokenType::Comment => {
            let comment = if line.len() < 3 { "" } else { &line[3..] };
            root.push(Root::Comment(comment.to_string()));
        }
        x => {
            return Err(Error {
                error: ErrorTy::InvalidToken,
                span: (ctx.offset + ctx.inline_offset, 1),
            });
        }
    }

    // we should have parsed everything in the line
    if tokens.len() > 0 {
        return Err(Error {
            error: ErrorTy::InvalidEndOfLine,
            span: (ctx.offset + ctx.inline_offset, 1),
        });
    }

    Ok(())
}

pub fn is_valid_module(module: &str) -> bool {
    let mut chars = module.chars();
    !module.is_empty()
        && chars.next().unwrap().is_ascii_alphabetic()
        && chars.all(|c| c.is_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    const CODE: &str = r#"use crypto::poseidon;

fn main(public_input: [fel; 3], private_input: [fel; 3]) -> [fel; 8] {
    let digest = poseidon(private_input);
    assert(digest == public_input);
"#;

    #[test]
    fn test_parse() {
        match parse(CODE) {
            Ok(root) => {
                println!("{:#?}", root);
            }
            Err(e) => {
                println!("{:?}", e);
            }
        }
    }
}
