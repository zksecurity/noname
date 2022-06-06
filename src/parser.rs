use crate::error::{Error, ErrorTy};
use crate::lexer::{Keyword, Token, TokenType};

#[derive(Debug, Default)]
pub struct ParserCtx {
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

#[derive(Debug)]
pub struct Path(Vec<String>);

impl Path {
    /// Parses a path from a list of tokens.
    pub fn parse_path<'a>(
        ctx: &mut ParserCtx,
        tokens: &mut impl Iterator<Item = &'a Token>,
    ) -> Result<Self, Error> {
        let mut path = vec![];

        let mut tokens = tokens.peekable();
        loop {
            // no token to read
            let token = tokens.next().ok_or(Error {
                error: ErrorTy::InvalidPath,
                span: ctx.last_span(),
            })?;
            ctx.last_token = Some(token.clone());

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
                            ctx.last_token = token.cloned();
                            continue;
                        }
                        // end of path
                        _ => {
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
    pub fn parse_fn<'a>(
        ctx: &mut ParserCtx,
        tokens: &mut impl Iterator<Item = &'a Token>,
    ) -> Result<Self, Error> {
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

#[derive(Debug, Default)]
pub struct AST(Vec<Root>);

impl AST {
    pub fn parse(tokens: &[Token]) -> Result<AST, Error> {
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
                ast.push(Root::Use(path));

                // end of line
                if !matches!(
                    tokens.next(),
                    Some(Token {
                        typ: TokenType::SemiColon,
                        ..
                    })
                ) {
                    return Err(Error {
                        error: ErrorTy::InvalidEndOfLine,
                        span: token.span,
                    });
                }
            }
            TokenType::Keyword(Keyword::Fn) => {
                let func = Function::parse_fn(&mut ctx, &mut tokens)?;
                ast.push(Root::Function(func));
            }
            TokenType::Comment(comment) => {
                ast.push(Root::Comment(comment.clone()));
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
