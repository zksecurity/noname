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
    pub last_token: Option<Token>,
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
            let token = tokens.bump(ctx).ok_or(Error {
                error: ErrorTy::InvalidPath,
                span: ctx.last_span(),
            })?;
            ctx.last_token = Some(token.clone());

            match &token.typ {
                // a chunk of the path
                TokenType::Identifier(chunk) => {
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
                            let token = tokens.bump(ctx);
                            ctx.last_token = token;
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

//
// Type
//

#[derive(Debug)]
pub enum Ty {
    Struct(String),
    Array(Box<Self>, u32),
}

impl Ty {
    pub fn parse<I>(ctx: &mut ParserCtx, tokens: &mut I) -> Result<Self, Error>
    where
        I: Iterator<Item = Token> + Peekable,
    {
        let token = tokens.bump_err(ctx, ErrorTy::MissingType)?;

        match token.typ {
            // struct name
            TokenType::Identifier(name) => {
                if !is_valid_fn_type(&name) {
                    return Err(Error {
                        error: ErrorTy::InvalidTypeName,
                        span: token.span,
                    });
                }
                Ok(Ty::Struct(name))
            }

            // array
            // [type; size]
            // ^
            TokenType::LeftBracket => {
                // [type; size]
                //   ^
                let ty = Box::new(Ty::parse(ctx, tokens)?);

                // [type; size]
                //         ^
                let siz = tokens.bump_err(ctx, ErrorTy::InvalidToken)?;
                let siz = match siz.typ {
                    TokenType::BigInt(s) => s.parse().map_err(|_| Error {
                        error: ErrorTy::InvalidArraySize,
                        span: siz.span,
                    })?,
                    _ => {
                        return Err(Error {
                            error: ErrorTy::InvalidArraySize,
                            span: siz.span,
                        })
                    }
                };

                // [type; size]
                //            ^
                let right_bracket = tokens.bump_expected(ctx, TokenType::RightBracket)?;

                Ok(Ty::Array(ty, siz))
            }

            // unrecognized
            _ => {
                return Err(Error {
                    error: ErrorTy::InvalidType,
                    span: token.span,
                })
            }
        }
    }
}

//~
//~ ## Expression
//~
//~ Backus–Naur Form (BNF) grammar:
//~
//~ expr ::=
//~     | expr { bin_op expr }
//~     | "-" expr
//~     | "(" expr ")"
//~     | numeric
//~     | ident
//~     | fn_call
//~     | array_access
//~ bin_op ::= "+" | "-" | "/" | "*"
//~ numeric ::= /[0-9]+/
//~ ident ::= /[A-Za-z_][A-Za-z_0-9]*/
//~ fn_call ::= ident "(" expr { "," expr } ")"
//~ array_access ::= ident "[" expr "]"
//~
//~ powexpr ::= "-" powexpr | "+" powexpr | atom [ "^" powexpr ]
//~ atom ::= ident [ "(" expr ")" ] | numeric | "(" expr ")"
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
    //    Literal(String),
    FnCall {
        function_name: String,
        args: Vec<Expression>,
    },
    Variable(String),
    Comparison(ComparisonOp, Box<Expression>, Box<Expression>),
    Op(Op2, Box<Expression>, Box<Expression>),
    Negated(Box<Expression>),
    BigInt(String),
    Identifier(String),
    ArrayAccess(String, Box<Expression>),
}

#[derive(Debug)]
pub enum Op2 {
    Addition,
    Subtraction,
    Multiplication,
    Division,
}

impl Op2 {
    pub fn parse_maybe<I>(ctx: &mut ParserCtx, tokens: &mut I) -> Option<Self>
    where
        I: Iterator<Item = Token> + Peekable,
    {
        let token = tokens.peek()?;

        match token.typ {
            TokenType::Plus => Some(Op2::Addition),
            TokenType::Minus => Some(Op2::Subtraction),
            TokenType::Star => Some(Op2::Multiplication),
            TokenType::Slash => Some(Op2::Division),
            _ => None,
        }
    }
}

impl Expression {
    /// Parses until it finds something it doesn't know, then returns without consuming the token it doesn't know (the caller will have to make sense of it)
    pub fn parse<I>(ctx: &mut ParserCtx, tokens: &mut I) -> Result<Self, Error>
    where
        I: Iterator<Item = Token> + Peekable,
    {
        let token = tokens.bump_err(ctx, ErrorTy::MissingExpression)?;
        let lhs = match token.typ {
            // numeric
            TokenType::BigInt(b) => Expression::BigInt(b),

            // identifier
            TokenType::Identifier(ident) => {
                // could be array access, fn call
                let peeked = match tokens.peek() {
                    None => panic!("woot"),
                    Some(x) => x,
                };

                match peeked.typ {
                    // array access
                    TokenType::LeftBracket => {
                        let expr = Expression::parse(ctx, tokens)?;
                        let right_bracket = tokens.bump_expected(ctx, TokenType::RightBracket)?;
                        Expression::ArrayAccess(ident, Box::new(expr))
                    }
                    // fn call
                    TokenType::LeftParen => {
                        let mut args = vec![];
                        loop {
                            let arg = Expression::parse(ctx, tokens)?;
                            args.push(arg);

                            match tokens.peek() {
                                Some(x) => match x.typ {
                                    TokenType::Comma => {
                                        tokens.bump(ctx);
                                    }
                                    TokenType::RightParen => {
                                        tokens.bump(ctx);
                                        break;
                                    }
                                    _ => (),
                                },
                                None => {
                                    return Err(Error {
                                        error: ErrorTy::InvalidFnCall,
                                        span: ctx.last_span(),
                                    })
                                }
                            }
                        }
                        Expression::FnCall {
                            function_name: ident,
                            args,
                        }
                    }
                    _ => {
                        // just a variable
                        Expression::Identifier(ident)
                    }
                }
            }

            // negated expr
            TokenType::Minus => {
                let expr = Expression::parse(ctx, tokens)?;
                Expression::Negated(Box::new(expr))
            }

            // parenthesis
            TokenType::LeftParen => {
                let expr = Expression::parse(ctx, tokens)?;
                let right_paren = tokens.bump_expected(ctx, TokenType::RightParen)?;
                expr
            }

            // unrecognized pattern
            _ => {
                return Err(Error {
                    error: ErrorTy::InvalidExpression,
                    span: token.span,
                })
            }
        };

        // bin op or return lhs
        if let Some(op) = Op2::parse_maybe(ctx, tokens) {
            let rhs = Expression::parse(ctx, tokens)?;
            Ok(Expression::Op(op, Box::new(lhs), Box::new(rhs)))
        } else {
            Ok(lhs)
        }
    }
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
    pub fn parse_name<I>(ctx: &mut ParserCtx, tokens: &mut I) -> Result<String, Error>
    where
        I: Iterator<Item = Token> + Peekable,
    {
        let token = tokens.bump_err(ctx, ErrorTy::InvalidFunctionSignature)?;

        let name = match token {
            Token {
                typ: TokenType::Identifier(name),
                ..
            } => {
                if is_valid_fn_name(&name) {
                    name
                } else {
                    return Err(Error {
                        error: ErrorTy::InvalidFunctionSignature,
                        span: token.span,
                    });
                }
            }
            _ => {
                return Err(Error {
                    error: ErrorTy::InvalidFunctionSignature,
                    span: token.span,
                });
            }
        };

        Ok(name)
    }

    pub fn parse_args<I>(ctx: &mut ParserCtx, tokens: &mut I) -> Result<Vec<(String, Ty)>, Error>
    where
        I: Iterator<Item = Token> + Peekable,
    {
        // (pub arg1: type1, arg2: type2)
        // ^
        let left_paren = tokens.bump_err(ctx, ErrorTy::InvalidFunctionSignature)?;
        if !matches!(left_paren.typ, TokenType::LeftParen) {
            return Err(Error {
                error: ErrorTy::InvalidFunctionSignature,
                span: left_paren.span,
            });
        }

        // (pub arg1: type1, arg2: type2)
        //   ^
        let mut args = vec![];
        loop {
            // `pub arg1: type1`
            //   ^   ^
            let token = tokens.bump_err(ctx, ErrorTy::InvalidFunctionSignature)?;

            let (public, arg_name) = match token.typ {
                // public input
                TokenType::Keyword(Keyword::Pub) => {
                    let arg_name = parse_ident(ctx, tokens)?;

                    if !is_valid_fn_name(&arg_name) {
                        return Err(Error {
                            error: ErrorTy::InvalidFunctionSignature,
                            span: token.span,
                        });
                    }
                    (true, arg_name)
                }
                // private input
                TokenType::Identifier(name) => {
                    if !is_valid_fn_name(&name) {
                        return Err(Error {
                            error: ErrorTy::InvalidFunctionSignature,
                            span: token.span,
                        });
                    }
                    (false, name)
                }
                _ => {
                    return Err(Error {
                        error: ErrorTy::InvalidFunctionSignature,
                        span: token.span,
                    })
                }
            };

            // :
            let colon = tokens.bump_err(ctx, ErrorTy::InvalidFunctionSignature)?;
            if matches!(colon.typ, TokenType::Colon) {
                return Err(Error {
                    error: ErrorTy::InvalidFunctionSignature,
                    span: colon.span,
                });
            }

            // type
            let arg_typ = Ty::parse(ctx, tokens)?;

            // , or )
            let separator = tokens.bump_err(ctx, ErrorTy::InvalidFunctionSignature)?;
            match separator.typ {
                // (pub arg1: type1, arg2: type2)
                //                 ^
                TokenType::Comma => {
                    args.push((arg_name, arg_typ));
                }
                // (pub arg1: type1, arg2: type2)
                //                              ^
                TokenType::RightParen => {
                    args.push((arg_name, arg_typ));
                    break;
                }
                _ => {
                    return Err(Error {
                        error: ErrorTy::InvalidFunctionSignature,
                        span: separator.span,
                    });
                }
            }
        }

        Ok(args)
    }

    pub fn parse_fn_return_type<I>(ctx: &mut ParserCtx, tokens: &mut I) -> Result<Option<Ty>, Error>
    where
        I: Iterator<Item = Token> + Peekable,
    {
        match tokens.peek() {
            Some(Token {
                typ: TokenType::LeftCurlyBracket,
                ..
            }) => {
                return Ok(None);
            }
            _ => (),
        };

        let _right_arrow = tokens.bump_expected(ctx, TokenType::RightArrow)?;

        let return_type = Ty::parse(ctx, tokens)?;
        Ok(Some(return_type))
    }

    pub fn parse_fn_body<I>(ctx: &mut ParserCtx, tokens: &mut I) -> Result<Vec<Statement>, Error>
    where
        I: Iterator<Item = Token> + Peekable,
    {
        // should I retrieve all the tokens until I see `}`, then pass that to Statement::parse() ?
        todo!()
    }

    pub fn parse_fn<I>(ctx: &mut ParserCtx, tokens: &mut I) -> Result<Self, Error>
    where
        I: Iterator<Item = Token> + Peekable,
    {
        let name = Self::parse_name(ctx, tokens)?;
        let arguments = Self::parse_args(ctx, tokens)?;
        let return_type = Self::parse_fn_return_type(ctx, tokens)?;
        let body = Self::parse_fn_body(ctx, tokens)?;

        let func = Self {
            name,
            arguments,
            return_type,
            body,
        };
        let name = tokens.bump(ctx).ok_or(Error {
            error: ErrorTy::InvalidModule,
            span: ctx.last_span(),
        })?;

        Ok(func)
    }
}

// TODO: enforce snake_case?
pub fn is_valid_fn_name(name: &str) -> bool {
    if let Some(first_char) = name.chars().next() {
        // first character is not a number
        (first_char.is_alphabetic() || first_char == '_')
            // first character is lowercase
            && first_char.is_lowercase()
            // all other characters are alphanumeric or underscore
            && name.chars().all(|c| c.is_alphanumeric() || c == '_')
    } else {
        false
    }
}

// TODO: enforce CamelCase?
pub fn is_valid_fn_type(name: &str) -> bool {
    if let Some(first_char) = name.chars().next() {
        // first character is not a number or alpha
        first_char.is_alphabetic()
            // first character is uppercase
            && first_char.is_uppercase()
            // all other characters are alphanumeric or underscore
            && name.chars().all(|c| c.is_alphanumeric() || c == '_')
    } else {
        false
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

impl Statement {
    /// Returns a list of statement parsed until seeing the end of a block (`}`).
    pub fn parse<I>(ctx: &mut ParserCtx, tokens: &mut I) -> Result<Vec<Self>, Error>
    where
        I: Iterator<Item = Token> + Peekable,
    {
        let mut statements = vec![];

        loop {
            let token = tokens.bump_err(ctx, ErrorTy::InvalidStatement)?;
            match token.typ {
                // end of block
                TokenType::RightCurlyBracket => break,
                // assignment
                TokenType::Keyword(Keyword::Let) => {
                    let lhs = parse_ident(ctx, tokens)?;
                    let _equal = tokens.bump_expected(ctx, TokenType::Equal)?;
                    let rhs = Expression::parse(ctx, tokens)?;
                    let _semi_colon = tokens.bump_expected(ctx, TokenType::SemiColon)?;
                    statements.push(Statement::Assign { lhs, rhs });
                }
                // assert
                TokenType::Keyword(Keyword::Assert) => {
                    let expr = Expression::parse(ctx, tokens)?;
                    let _semi_colon = tokens.bump_expected(ctx, TokenType::SemiColon)?;
                    statements.push(Statement::Assert(expr));
                }
                // return
                TokenType::Keyword(Keyword::Return) => {
                    let expr = Expression::parse(ctx, tokens)?;
                    let _semi_colon = tokens.bump_expected(ctx, TokenType::SemiColon)?;
                    statements.push(Statement::Return(expr));
                }
                // comment
                TokenType::Comment(c) => {
                    statements.push(Statement::Comment(c));
                }
                //
                _ => {
                    return Err(Error {
                        error: ErrorTy::InvalidStatement,
                        span: token.span,
                    });
                }
            }
        }

        Ok(statements)
    }
}

//
// Scope
//

// TODO: where do I enforce that there's not several `use` with the same module name? or several functions with the same names? I guess that's something I need to enforce in any scope anyway...
#[derive(Debug)]

/// Things you can have in a scope (including the root scope).
pub enum Scope {
    Use(Path),
    Function(Function),
    Comment(String),
    //    Struct(Struct)
}

//
// AST
//

#[derive(Debug, Default)]
pub struct AST(Vec<Scope>);

impl AST {
    pub fn parse<I>(tokens: I) -> Result<AST, Error>
    where
        I: Iterator<Item = Token> + Peekable,
    {
        let mut ast = vec![];
        let ctx = &mut ParserCtx::default();

        // get first token
        let mut tokens = tokens.into_iter();
        let token = if let Some(token) = tokens.bump(ctx) {
            token
        } else {
            return Ok(AST::default()); // empty line
        };

        // match special keywords
        match &token.typ {
            TokenType::Keyword(Keyword::Use) => {
                let path = Path::parse_path(ctx, &mut tokens)?;
                ast.push(Scope::Use(path));

                // end of line
                let next_token = tokens.bump(ctx);
                if !matches!(
                    next_token,
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
                let func = Function::parse_fn(ctx, &mut tokens)?;
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
        if let Some(token) = tokens.bump(ctx) {
            return Err(Error {
                error: ErrorTy::InvalidEndOfLine,
                span: token.span,
            });
        }

        Ok(Self(ast))
    }
}

//
// Helpers
//

pub fn is_valid_module(module: &str) -> bool {
    let mut chars = module.chars();
    !module.is_empty()
        && chars.next().unwrap().is_ascii_alphabetic()
        && chars.all(|c| c.is_alphanumeric() || c == '_')
}

// maybe should be implemented on token?

pub fn parse_ident<I>(ctx: &mut ParserCtx, tokens: &mut I) -> Result<String, Error>
where
    I: Iterator<Item = Token> + Peekable,
{
    let token = tokens.bump_err(ctx, ErrorTy::MissingToken)?;
    match token.typ {
        TokenType::Identifier(ident) => Ok(ident),
        _ => Err(Error {
            error: ErrorTy::InvalidFunctionSignature,
            span: token.span,
        }),
    }
}
