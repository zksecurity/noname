use crate::error::{Error, ErrorTy};
use crate::lexer::{Keyword, Token, TokenType};
use crate::peekable::Peekable;

//
// Context
//

#[derive(Debug, Default)]
pub struct ParserCtx {
    /// Used mainly for error reporting,
    /// when we don't have a token to read
    last_token: Option<Token>,
}

impl ParserCtx {
    pub fn last_span(&self) -> (usize, usize) {
        let span = self
            .last_token
            .as_ref()
            .map(|token| token.span)
            .unwrap_or((0, 0));
        (span.0 + span.1, 0)
    }
}

//
// Path
//

#[derive(Debug)]
pub struct Path(Vec<String>);

impl Path {
    /// Parses a path from a list of tokens.
    pub fn parse_path<I>(ctx: &mut ParserCtx, tokens: &mut I) -> Result<Self, Error>
    where
        I: Iterator<Item = Token> + Peekable,
    {
        let mut path = vec![];
        loop {
            // no token to read
            let token = tokens.next().ok_or(Error {
                error: ErrorTy::InvalidPath,
                span: ctx.last_span(),
            })?;
            ctx.last_token = Some(token.clone());

            dbg!(&token);

            match &token.typ {
                // a chunk of the path
                TokenType::AlphaNumeric_(chunk) => {
                    if !is_valid_module(&chunk) {
                        return Err(Error {
                            error: ErrorTy::InvalidModule,
                            span: token.span,
                        });
                    }

                    path.push(chunk.to_string());

                    // next, we expect a `::` to continue the path,
                    // or a separator, for example, `;` or `)`
                    match tokens.peek() {
                        None => {
                            return Err(Error {
                                error: ErrorTy::InvalidEndOfLine,
                                span: token.span,
                            })
                        }
                        // path separator
                        Some(Token {
                            typ: TokenType::DoubleColon,
                            ..
                        }) => {
                            let token = tokens.next();
                            ctx.last_token = token;
                            continue;
                        }
                        // end of path
                        x => {
                            dbg!(x);
                            return Ok(Path(path));
                        }
                    }
                }
                // path must start with an alphanumeric thing
                _ => {
                    return Err(Error {
                        error: ErrorTy::InvalidPath,
                        span: token.span,
                    });
                }
            }
        }
    }
}

//
// Type
//

#[derive(Debug)]
pub enum Ty {
    Field,
    Array(Box<Self>, usize),
    Bool,
}

//
// Expression
//

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

//
// Function
//

#[derive(Debug)]
pub struct Function {
    pub name: String,
    pub arguments: Vec<(String, Ty)>,
    pub return_type: Option<Ty>,
    pub body: Vec<Statement>,
}

impl Function {
    pub fn parse_fn<I>(ctx: &mut ParserCtx, tokens: &mut I) -> Result<Self, Error>
    where
        I: Iterator<Item = Token> + Peekable,
    {
        // parse function name
        let name = tokens.next().ok_or(Error {
            error: ErrorTy::InvalidFunctionSignature,
            span: ctx.last_span(),
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
            span: ctx.last_span(),
        })?;

        Ok(func)
    }
}

//
// Statement
//

#[derive(Debug)]
pub enum Statement {
    Assign { lhs: String, rhs: Expression },
    Assert(Expression),
    Return(Expression),
    Comment(String),
}

// TODO: where do I enforce that there's not several `use` with the same module name? or several functions with the same names? I guess that's something I need to enforce in any scope anyway...
#[derive(Debug)]

/// Things you can have in a scope (including the root scope).
pub enum Scope {
    Use(Path),
    Function(Function),
    Comment(String),
    //    Struct(Struct)
}
#[derive(Debug, Default)]
pub struct AST(Vec<Scope>);

impl AST {
    pub fn parse<I>(tokens: I) -> Result<AST, Error>
    where
        I: Iterator<Item = Token> + Peekable,
    {
        let mut ast = vec![];
        let mut ctx = ParserCtx::default();

        // get first token
        let mut tokens = tokens.into_iter();
        let token = if let Some(token) = tokens.next() {
            token
        } else {
            return Ok(AST::default()); // empty line
        };

        // match special keywords
        match &token.typ {
            TokenType::Keyword(Keyword::Use) => {
                let path = Path::parse_path(&mut ctx, &mut tokens)?;
                dbg!(&path);
                ast.push(Scope::Use(path));

                // end of line
                let next_token = tokens.next();
                if !matches!(
                    next_token,
                    Some(Token {
                        typ: TokenType::SemiColon,
                        ..
                    })
                ) {
                    dbg!(next_token);
                    return Err(Error {
                        error: ErrorTy::InvalidEndOfLine,
                        span: token.span,
                    });
                }
            }
            TokenType::Keyword(Keyword::Fn) => {
                let func = Function::parse_fn(&mut ctx, &mut tokens)?;
                ast.push(Scope::Function(func));
            }
            TokenType::Comment(comment) => {
                ast.push(Scope::Comment(comment.clone()));
            }
            _ => {
                return Err(Error {
                    error: ErrorTy::InvalidToken,
                    span: token.span,
                });
            }
        }

        // we should have parsed everything in the line
        if let Some(token) = tokens.next() {
            return Err(Error {
                error: ErrorTy::InvalidEndOfLine,
                span: token.span,
            });
        }

        Ok(Self(ast))
    }
}

pub fn is_valid_module(module: &str) -> bool {
    let mut chars = module.chars();
    !module.is_empty()
        && chars.next().unwrap().is_ascii_alphabetic()
        && chars.all(|c| c.is_alphanumeric() || c == '_')
}
