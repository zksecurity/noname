use std::fmt::Display;

use crate::{
    ast::Environment,
    constants::Span,
    error::{Error, ErrorTy},
    lexer::{Keyword, Token, TokenKind},
    tokens::Tokens,
};

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

#[derive(Debug, Default)]
pub struct ParserCtx {
    /// Used mainly for error reporting,
    /// when we don't have a token to read
    pub last_token: Option<Token>,
}

impl ParserCtx {
    // TODO: I think I don't need this, I should always be able to use the last token I read if I don't see anything, otherwise maybe just write -1 to say "EOF"
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
pub struct Path(pub Vec<String>);

impl Path {
    /// Parses a path from a list of tokens.
    pub fn parse_path(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self, Error> {
        let mut path = vec![];
        loop {
            // no token to read
            let token = tokens.bump(ctx).ok_or(Error {
                error: ErrorTy::InvalidPath,
                span: ctx.last_span(),
            })?;
            ctx.last_token = Some(token.clone());

            match &token.kind {
                // a chunk of the path
                TokenKind::Identifier(chunk) => {
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
                            kind: TokenKind::DoubleColon,
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

//~
//~ ## Type
//~
//~ Backus–Naur Form (BNF) grammar:
//~
//~ type ::=
//~     | /[A-Z] (A-Za-z0-9)*/
//~     | "[" type ";" numeric "]"
//~
//~ numeric ::= /[0-9]+/
//~

#[derive(Debug, Clone)]
pub struct Ty {
    pub kind: TyKind,
    pub span: (usize, usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TyKind {
    Custom(String),
    Field,
    BigInt,
    Array(Box<TyKind>, u32),
}

impl Display for TyKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TyKind::Custom(name) => write!(f, "{}", name),
            TyKind::Field => write!(f, "Field"),
            TyKind::BigInt => write!(f, "BigInt"),
            TyKind::Array(ty, size) => write!(f, "[{}; {}]", ty, size),
        }
    }
}

impl Ty {
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self, Error> {
        let token = tokens.bump_err(ctx, ErrorTy::MissingType)?;

        match token.kind {
            // struct name
            TokenKind::Type(name) => {
                if name == "Field" {
                    Ok(Self {
                        kind: TyKind::Field,
                        span: token.span,
                    })
                } else {
                    Ok(Self {
                        kind: TyKind::Custom(name.to_string()),
                        span: token.span,
                    })
                }
            }

            // array
            // [type; size]
            // ^
            TokenKind::LeftBracket => {
                // [type; size]
                //   ^
                let ty = Ty::parse(ctx, tokens)?;

                // [type; size]
                //      ^
                tokens.bump_expected(ctx, TokenKind::SemiColon)?;

                // [type; size]
                //         ^
                let siz = tokens.bump_err(ctx, ErrorTy::InvalidToken)?;
                let siz = match siz.kind {
                    TokenKind::BigInt(s) => s.parse().map_err(|_e| Error {
                        error: ErrorTy::InvalidArraySize,
                        span: siz.span,
                    })?,
                    _ => {
                        return Err(Error {
                            error: ErrorTy::ExpectedToken(TokenKind::BigInt("".to_string())),
                            span: siz.span,
                        });
                    }
                };

                // [type; size]
                //            ^
                tokens.bump_expected(ctx, TokenKind::RightBracket)?;

                Ok(Ty {
                    kind: TyKind::Array(Box::new(ty.kind), siz),
                    span: token.span,
                })
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
//~ bin_op ::= "+" | "-" | "/" | "*" | "=="
//~ numeric ::= /[0-9]+/
//~ ident ::= /[A-Za-z_][A-Za-z_0-9]*/
//~ fn_call ::= ident "(" expr { "," expr } ")"
//~ array_access ::= ident "[" expr "]"
//~
#[derive(Debug, Clone)]
pub enum ComparisonOp {
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Equal,
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub typ: Option<TyKind>,
    pub span: (usize, usize),
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    //    Literal(String),
    FnCall {
        function_name: String,
        args: Vec<Expr>,
    },
    Variable(String),
    // TODO: move to Op?
    Comparison(ComparisonOp, Box<Expr>, Box<Expr>),
    Op(Op2, Box<Expr>, Box<Expr>),
    Negated(Box<Expr>),
    BigInt(String),
    Identifier(String),
    ArrayAccess(String, Box<Expr>),
}

#[derive(Debug, Clone)]
pub enum Op2 {
    Addition,
    Subtraction,
    Multiplication,
    Division,
    Equality,
}

impl Op2 {
    pub fn parse_maybe(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Option<Self> {
        let token = tokens.peek()?;

        let token = match token.kind {
            TokenKind::Plus => Some(Op2::Addition),
            TokenKind::Minus => Some(Op2::Subtraction),
            TokenKind::Star => Some(Op2::Multiplication),
            TokenKind::Slash => Some(Op2::Division),
            TokenKind::DoubleEqual => Some(Op2::Equality),
            _ => None,
        };

        if token.is_some() {
            tokens.bump(ctx);
        }

        token
    }
}

impl Expr {
    /// Parses until it finds something it doesn't know, then returns without consuming the token it doesn't know (the caller will have to make sense of it)
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self, Error> {
        let token = tokens.bump_err(ctx, ErrorTy::MissingExpression)?;
        let span = token.span;

        let lhs = match token.kind {
            // numeric
            TokenKind::BigInt(b) => Expr {
                kind: ExprKind::BigInt(b),
                typ: Some(TyKind::BigInt),
                span,
            },

            // identifier
            TokenKind::Identifier(ident) => {
                // could be array access, fn call
                let peeked = match tokens.peek() {
                    None => {
                        return Err(Error {
                            error: ErrorTy::InvalidEndOfLine,
                            span: ctx.last_span(),
                        })
                    }
                    Some(x) => x,
                };

                match peeked.kind {
                    // array access
                    TokenKind::LeftBracket => {
                        tokens.bump(ctx); // [

                        let expr = Expr::parse(ctx, tokens)?;
                        tokens.bump_expected(ctx, TokenKind::RightBracket)?;

                        Expr {
                            kind: ExprKind::ArrayAccess(ident, Box::new(expr)),
                            typ: None,
                            span,
                        }
                    }
                    // fn call
                    TokenKind::LeftParen => {
                        tokens.bump(ctx); // (

                        let mut args = vec![];
                        loop {
                            let arg = Expr::parse(ctx, tokens)?;

                            args.push(arg);

                            let pp = tokens.peek();

                            match pp {
                                Some(x) => match x.kind {
                                    TokenKind::Comma => {
                                        tokens.bump(ctx);
                                    }
                                    TokenKind::RightParen => {
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

                        Expr {
                            kind: ExprKind::FnCall {
                                function_name: ident,
                                args,
                            },
                            typ: None,
                            span,
                        }
                    }
                    _ => {
                        // just a variable
                        Expr {
                            kind: ExprKind::Identifier(ident),
                            typ: None,
                            span,
                        }
                    }
                }
            }

            // negated expr
            TokenKind::Minus => {
                let expr = Expr::parse(ctx, tokens)?;

                Expr {
                    kind: ExprKind::Negated(Box::new(expr)),
                    typ: None,
                    span,
                }
            }

            // parenthesis
            TokenKind::LeftParen => {
                let expr = Expr::parse(ctx, tokens)?;
                tokens.bump_expected(ctx, TokenKind::RightParen)?;
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
            // TODO: there's a bug here, rhs parses the lhs again
            let rhs = Expr::parse(ctx, tokens)?;
            let span = {
                let end = rhs.span.0 + rhs.span.1;
                (span.0, end - span.0)
            };
            Ok(Expr {
                kind: ExprKind::Op(op, Box::new(lhs), Box::new(rhs)),
                typ: None,
                span,
            })
        } else {
            Ok(lhs)
        }
    }

    pub fn compute_type(&self, scope: &mut Environment) -> Result<Option<TyKind>, Error> {
        match &self.kind {
            ExprKind::FnCall {
                function_name,
                args,
            } => todo!(),
            ExprKind::Variable(_) => todo!(),
            ExprKind::Comparison(_, _, _) => todo!(),
            ExprKind::Op(_, lhs, rhs) => {
                let lhs_typ = lhs.compute_type(scope)?.unwrap();
                let rhs_typ = rhs.compute_type(scope)?.unwrap();

                if lhs_typ != rhs_typ {
                    // only allow bigint mixed with field
                    match (&lhs_typ, &rhs_typ) {
                        (TyKind::BigInt, TyKind::Field) | (TyKind::Field, TyKind::BigInt) => (),
                        _ => {
                            return Err(Error {
                                error: ErrorTy::MismatchType(lhs_typ.clone(), rhs_typ.clone()),
                                span: self.span,
                            })
                        }
                    }
                }

                Ok(Some(lhs_typ))
            }
            ExprKind::Negated(_) => todo!(),
            ExprKind::BigInt(_) => Ok(Some(TyKind::BigInt)),
            ExprKind::Identifier(ident) => {
                let typ = scope.get_type(ident).ok_or(Error {
                    error: ErrorTy::UndefinedVariable,
                    span: self.span,
                })?;

                Ok(Some(typ.clone()))
            }
            ExprKind::ArrayAccess(_, _) => todo!(),
        }
    }
}

//~
//~ ## Functions
//~
//~ Backus–Naur Form (BNF) grammar:
//~
//~ fn_sig ::= ident "(" param { "," param } ")" [ return_val ]
//~ return_val ::= "->" type
//~ param ::= { "pub" } ident ":" type
//~

#[derive(Debug)]
pub struct FunctionSig {
    pub name: Ident,

    /// (pub, ident, type)
    pub arguments: Vec<(Attribute, Ident, Ty)>,

    pub return_type: Option<Ty>,
}

impl FunctionSig {
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self, Error> {
        let name = Function::parse_name(ctx, tokens)?;

        let arguments = Function::parse_args(ctx, tokens)?;

        let return_type = Function::parse_fn_return_type(ctx, tokens)?;

        Ok(Self {
            name,
            arguments,
            return_type,
        })
    }
}

#[derive(Debug, Default, Clone)]
pub struct Ident {
    pub value: String,
    pub span: Span,
}

#[derive(Debug, Default, Clone, Copy)]
pub enum Attribute {
    #[default]
    Priv,
    Pub,
}

impl Attribute {
    pub fn is_public(&self) -> bool {
        matches!(self, Self::Pub)
    }
}

#[derive(Debug)]
pub struct Function {
    pub name: Ident,

    /// (pub, ident, type)
    pub arguments: Vec<(Attribute, Ident, Ty)>,

    pub return_type: Option<Ty>,

    pub body: Vec<Stmt>,

    pub span: Span,
}

impl Function {
    pub fn is_main(&self) -> bool {
        self.name.value == "main"
    }

    pub fn parse_name(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Ident, Error> {
        let token = tokens.bump_err(
            ctx,
            ErrorTy::InvalidFunctionSignature("expected function name"),
        )?;

        let name = match token {
            Token {
                kind: TokenKind::Identifier(name),
                ..
            } => name,
            _ => {
                return Err(Error {
                    error: ErrorTy::InvalidFunctionSignature("expected function name to be lowercase alphanumeric (including underscore `_`) and starting with a letter"),
                    span: token.span,
                });
            }
        };

        Ok(Ident {
            value: name,
            span: token.span,
        })
    }

    pub fn parse_args(
        ctx: &mut ParserCtx,
        tokens: &mut Tokens,
    ) -> Result<Vec<(Attribute, Ident, Ty)>, Error> {
        // (pub arg1: type1, arg2: type2)
        // ^
        tokens.bump_expected(ctx, TokenKind::LeftParen)?;

        // (pub arg1: type1, arg2: type2)
        //   ^
        let mut args = vec![];
        loop {
            // `pub arg1: type1`
            //   ^   ^
            let token = tokens.bump_err(
                ctx,
                ErrorTy::InvalidFunctionSignature("expected function arguments"),
            )?;

            let (public, arg_name) = match token.kind {
                // public input
                TokenKind::Keyword(Keyword::Pub) => {
                    let arg_name = parse_ident(ctx, tokens)?;
                    (Attribute::Pub, arg_name)
                }
                // private input
                TokenKind::Identifier(name) => (
                    Attribute::Priv,
                    Ident {
                        value: name,
                        span: token.span,
                    },
                ),
                _ => {
                    return Err(Error {
                        error: ErrorTy::InvalidFunctionSignature("expected identifier"),
                        span: token.span,
                    });
                }
            };

            // :
            tokens.bump_expected(ctx, TokenKind::Colon)?;

            // type
            let arg_typ = Ty::parse(ctx, tokens)?;

            // , or )
            let separator = tokens.bump_err(
                ctx,
                ErrorTy::InvalidFunctionSignature("expected end of function or other argument"),
            )?;

            match separator.kind {
                // (pub arg1: type1, arg2: type2)
                //                 ^
                TokenKind::Comma => {
                    args.push((public, arg_name, arg_typ));
                }
                // (pub arg1: type1, arg2: type2)
                //                              ^
                TokenKind::RightParen => {
                    args.push((public, arg_name, arg_typ));
                    break;
                }
                _ => {
                    return Err(Error {
                        error: ErrorTy::InvalidFunctionSignature(
                            "expected end of function or other argument",
                        ),
                        span: separator.span,
                    });
                }
            }
        }

        Ok(args)
    }

    pub fn parse_fn_return_type(
        ctx: &mut ParserCtx,
        tokens: &mut Tokens,
    ) -> Result<Option<Ty>, Error> {
        match tokens.peek() {
            Some(Token {
                kind: TokenKind::RightArrow,
                ..
            }) => {
                tokens.bump(ctx);

                let return_type = Ty::parse(ctx, tokens)?;
                Ok(Some(return_type))
            }
            _ => Ok(None),
        }
    }

    pub fn parse_fn_body(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Vec<Stmt>, Error> {
        let mut body = vec![];

        tokens.bump_expected(ctx, TokenKind::LeftCurlyBracket)?;

        loop {
            // end of the function
            let next_token = tokens.peek();
            if matches!(
                next_token,
                Some(Token {
                    kind: TokenKind::RightCurlyBracket,
                    ..
                })
            ) {
                tokens.bump(ctx);
                break;
            }

            // parse next statement
            let statement = Stmt::parse(ctx, tokens)?;
            body.push(statement);
        }

        Ok(body)
    }

    /// Parse a function, without the `fn` keyword.
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self, Error> {
        // ghetto way of getting the span of the function: get the span of the first token (name), then try to get the span of the last token
        let mut span = tokens
            .peek()
            .ok_or(Error {
                error: ErrorTy::InvalidFunctionSignature("expected function name"),
                span: ctx.last_span(),
            })?
            .span;

        let name = Self::parse_name(ctx, tokens)?;
        let arguments = Self::parse_args(ctx, tokens)?;
        let return_type = Self::parse_fn_return_type(ctx, tokens)?;
        let body = Self::parse_fn_body(ctx, tokens)?;

        // here's the last token, that is if the function is not empty (maybe we should disallow empty functions?)
        if let Some(t) = body.last() {
            span.1 = t.span.1;
        } else {
            return Err(Error {
                error: ErrorTy::InvalidFunctionSignature("expected function body"),
                span: ctx.last_span(),
            });
        }

        let func = Self {
            name,
            arguments,
            return_type,
            body,
            span,
        };

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
// Stmt
//

#[derive(Debug)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: (usize, usize),
}

#[derive(Debug)]
pub enum StmtKind {
    Assign { lhs: Ident, rhs: Box<Expr> },
    FnCall { name: Ident, args: Vec<Box<Expr>> },
    Return(Box<Expr>),
    Comment(String),
}

impl Stmt {
    /// Returns a list of statement parsed until seeing the end of a block (`}`).
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self, Error> {
        let token = tokens.bump_err(ctx, ErrorTy::InvalidStatement)?;
        let span = token.span;

        match token.kind {
            // assignment
            TokenKind::Keyword(Keyword::Let) => {
                let lhs = parse_ident(ctx, tokens)?;
                tokens.bump_expected(ctx, TokenKind::Equal)?;
                let rhs = Box::new(Expr::parse(ctx, tokens)?);
                tokens.bump_expected(ctx, TokenKind::SemiColon)?;
                Ok(Stmt {
                    kind: StmtKind::Assign { lhs, rhs },
                    span,
                })
            }
            // function call
            TokenKind::Identifier(name) => {
                let name = Ident {
                    value: name,
                    span: token.span,
                };

                tokens.bump_expected(ctx, TokenKind::LeftParen)?;

                let mut args = vec![];
                loop {
                    let arg = Expr::parse(ctx, tokens)?;

                    args.push(Box::new(arg));

                    let pp = tokens.peek();

                    match pp {
                        Some(x) => match x.kind {
                            TokenKind::Comma => {
                                tokens.bump(ctx);
                            }
                            TokenKind::RightParen => {
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

                tokens.bump_expected(ctx, TokenKind::SemiColon)?;

                Ok(Stmt {
                    kind: StmtKind::FnCall { name, args },
                    span,
                })
            }
            // assert
            /*
            TokenType::Keyword(Keyword::Assert) => {
                tokens.bump_expected(ctx, TokenType::LeftParen)?;
                let expr = Expr::parse(ctx, tokens)?;
                tokens.bump_expected(ctx, TokenType::RightParen)?;
                tokens.bump_expected(ctx, TokenType::SemiColon)?;
                Ok(Stmt {
                    typ: StmtKind::Assert(expr),
                    span,
                })
            }
            */
            // return
            TokenKind::Keyword(Keyword::Return) => {
                let expr = Expr::parse(ctx, tokens)?;
                tokens.bump_expected(ctx, TokenKind::SemiColon)?;
                Ok(Stmt {
                    kind: StmtKind::Return(Box::new(expr)),
                    span,
                })
            }
            // comment
            TokenKind::Comment(c) => Ok(Stmt {
                kind: StmtKind::Comment(c),
                span,
            }),
            //
            _ => {
                return Err(Error {
                    error: ErrorTy::InvalidStatement,
                    span: token.span,
                });
            }
        }
    }
}

//
// Scope
//

// TODO: where do I enforce that there's not several `use` with the same module name? or several functions with the same names? I guess that's something I need to enforce in any scope anyway...
#[derive(Debug)]

/// Things you can have in a scope (including the root scope).
pub struct Root {
    pub kind: RootKind,
    pub span: (usize, usize),
}

#[derive(Debug)]
pub enum RootKind {
    Use(Path),
    Function(Function),
    Comment(String),
    //    Struct(Struct)
}

//
// AST
//

#[derive(Debug, Default)]
pub struct AST(pub Vec<Root>);

impl AST {
    pub fn parse(mut tokens: Tokens) -> Result<AST, Error> {
        let mut ast = vec![];
        let ctx = &mut ParserCtx::default();

        // use statements must appear first
        let mut function_observed = false;

        loop {
            let token = match tokens.bump(ctx) {
                Some(token) => token,
                None => break,
            };

            match &token.kind {
                // `use crypto::poseidon;`
                TokenKind::Keyword(Keyword::Use) => {
                    if function_observed {
                        return Err(Error {
                            error: ErrorTy::UseAfterFn,
                            span: token.span,
                        });
                    }

                    let path = Path::parse_path(ctx, &mut tokens)?;
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
                        return Err(Error {
                            error: ErrorTy::InvalidEndOfLine,
                            span: token.span,
                        });
                    }
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

                // `// some comment`
                TokenKind::Comment(comment) => {
                    ast.push(Root {
                        kind: RootKind::Comment(comment.clone()),
                        span: token.span,
                    });
                }

                // unrecognized
                _ => {
                    return Err(Error {
                        error: ErrorTy::InvalidToken,
                        span: token.span,
                    });
                }
            }
        }

        Ok(Self(ast))
    }
}

//
// Helpers
//

pub fn parse_ident(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Ident, Error> {
    let token = tokens.bump_err(ctx, ErrorTy::MissingToken)?;
    match token.kind {
        TokenKind::Identifier(ident) => Ok(Ident {
            value: ident,
            span: token.span,
        }),
        _ => Err(Error {
            error: ErrorTy::ExpectedToken(TokenKind::Identifier("".to_string())),
            span: token.span,
        }),
    }
}

//
// Tests
//

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fn_signature() {
        let code = r#"main(pub public_input: [Fel; 3], private_input: [Fel; 3]) -> [Fel; 8] { }"#;
        let tokens = &mut Token::parse(code).unwrap();
        let ctx = &mut ParserCtx::default();
        let parsed = Function::parse(ctx, tokens).unwrap();
        println!("{:?}", parsed);
    }

    #[test]
    fn statement_assign() {
        let code = r#"let digest = poseidon(private_input);"#;
        let tokens = &mut Token::parse(code).unwrap();
        let ctx = &mut ParserCtx::default();
        let parsed = Stmt::parse(ctx, tokens).unwrap();
        println!("{:?}", parsed);
    }

    #[test]
    fn statement_assert() {
        let code = r#"assert(digest == public_input);"#;
        let tokens = &mut Token::parse(code).unwrap();
        let ctx = &mut ParserCtx::default();
        let parsed = Stmt::parse(ctx, tokens).unwrap();
        println!("{:?}", parsed);
    }
}
