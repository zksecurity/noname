use crate::{
    error::{Error, ErrorTy},
    lexer::{Keyword, Token, TokenType},
    tokens::Tokens,
};

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

            match &token.typ {
                // a chunk of the path
                TokenType::Identifier(chunk) => {
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
    pub typ: TyKind,
    pub span: (usize, usize),
}

#[derive(Debug, Clone)]
pub enum TyKind {
    Custom(String),
    Field,
    Array(Box<Ty>, u32),
}

impl Ty {
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self, Error> {
        let token = tokens.bump_err(ctx, ErrorTy::MissingType)?;

        match token.typ {
            // struct name
            TokenType::Type(name) => Ok(Self {
                typ: TyKind::Custom(name.to_string()),
                span: token.span,
            }),

            // array
            // [type; size]
            // ^
            TokenType::LeftBracket => {
                // [type; size]
                //   ^
                let ty = Box::new(Ty::parse(ctx, tokens)?);

                // [type; size]
                //      ^
                tokens.bump_expected(ctx, TokenType::SemiColon)?;

                // [type; size]
                //         ^
                let siz = tokens.bump_err(ctx, ErrorTy::InvalidToken)?;
                let siz = match siz.typ {
                    TokenType::BigInt(s) => s.parse().map_err(|_e| Error {
                        error: ErrorTy::InvalidArraySize,
                        span: siz.span,
                    })?,
                    _ => {
                        return Err(Error {
                            error: ErrorTy::ExpectedToken(TokenType::BigInt("".to_string())),
                            span: siz.span,
                        });
                    }
                };

                // [type; size]
                //            ^
                tokens.bump_expected(ctx, TokenType::RightBracket)?;

                Ok(Ty {
                    typ: TyKind::Array(ty, siz),
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
#[derive(Debug)]
pub enum ComparisonOp {
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Equal,
}

pub struct Expr {
    typ: ExprKind,
    span: (usize, usize),
}

#[derive(Debug)]
pub enum ExprKind {
    //    Literal(String),
    FnCall {
        function_name: String,
        args: Vec<ExprKind>,
    },
    Variable(String),
    Comparison(ComparisonOp, Box<ExprKind>, Box<ExprKind>),
    Op(Op2, Box<ExprKind>, Box<ExprKind>),
    Negated(Box<ExprKind>),
    BigInt(String),
    Identifier(String),
    ArrayAccess(String, Box<ExprKind>),
}

#[derive(Debug)]
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

        let token = match token.typ {
            TokenType::Plus => Some(Op2::Addition),
            TokenType::Minus => Some(Op2::Subtraction),
            TokenType::Star => Some(Op2::Multiplication),
            TokenType::Slash => Some(Op2::Division),
            TokenType::DoubleEqual => Some(Op2::Equality),
            _ => None,
        };

        if token.is_some() {
            tokens.bump(ctx);
        }

        token
    }
}

impl ExprKind {
    /// Parses until it finds something it doesn't know, then returns without consuming the token it doesn't know (the caller will have to make sense of it)
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self, Error> {
        let token = tokens.bump_err(ctx, ErrorTy::MissingExpression)?;
        let lhs = match token.typ {
            // numeric
            TokenType::BigInt(b) => ExprKind::BigInt(b),

            // identifier
            TokenType::Identifier(ident) => {
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

                match peeked.typ {
                    // array access
                    TokenType::LeftBracket => {
                        tokens.bump(ctx); // [

                        let expr = ExprKind::parse(ctx, tokens)?;
                        tokens.bump_expected(ctx, TokenType::RightBracket)?;
                        ExprKind::ArrayAccess(ident, Box::new(expr))
                    }
                    // fn call
                    TokenType::LeftParen => {
                        tokens.bump(ctx); // (

                        let mut args = vec![];
                        loop {
                            let arg = ExprKind::parse(ctx, tokens)?;

                            args.push(arg);

                            let pp = tokens.peek();

                            match pp {
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
                        ExprKind::FnCall {
                            function_name: ident,
                            args,
                        }
                    }
                    _ => {
                        // just a variable
                        ExprKind::Identifier(ident)
                    }
                }
            }

            // negated expr
            TokenType::Minus => {
                let expr = ExprKind::parse(ctx, tokens)?;
                ExprKind::Negated(Box::new(expr))
            }

            // parenthesis
            TokenType::LeftParen => {
                let expr = ExprKind::parse(ctx, tokens)?;
                tokens.bump_expected(ctx, TokenType::RightParen)?;
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
            let rhs = ExprKind::parse(ctx, tokens)?;
            Ok(ExprKind::Op(op, Box::new(lhs), Box::new(rhs)))
        } else {
            Ok(lhs)
        }
    }
}

//~
//~ ## Functions
//~
//~ Backus–Naur Form (BNF) grammar:
//~
//~ fn ::= ident "(" param { "," param } ")" [ return_val ] "{" { stmt ";" } "}"
//~ return_val ::= "->" type
//~ parm ::= { "pub" } ident ":" type
//~ stmt ::= ...
//~

#[derive(Debug)]
pub struct FunctionSig {
    pub name: String,

    /// (pub, ident, type)
    pub arguments: Vec<(bool, String, Ty)>,

    pub return_type: Option<Ty>,
}

#[derive(Debug)]
pub struct Function {
    pub name: String,

    /// (pub, ident, type)
    pub arguments: Vec<(bool, String, Ty)>,

    pub return_type: Option<Ty>,

    pub body: Vec<Statement>,
}

impl Function {
    pub fn parse_name(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<String, Error> {
        let token = tokens.bump_err(
            ctx,
            ErrorTy::InvalidFunctionSignature("expected function name"),
        )?;

        let name = match token {
            Token {
                typ: TokenType::Identifier(name),
                ..
            } => name,
            _ => {
                return Err(Error {
                    error: ErrorTy::InvalidFunctionSignature("expected function name to be lowercase alphanumeric (including underscore `_`) and starting with a letter"),
                    span: token.span,
                });
            }
        };

        Ok(name)
    }

    pub fn parse_args(
        ctx: &mut ParserCtx,
        tokens: &mut Tokens,
    ) -> Result<Vec<(bool, String, Ty)>, Error> {
        // (pub arg1: type1, arg2: type2)
        // ^
        tokens.bump_expected(ctx, TokenType::LeftParen)?;

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

            let (public, arg_name) = match token.typ {
                // public input
                TokenType::Keyword(Keyword::Pub) => {
                    let arg_name = parse_ident(ctx, tokens)?;
                    (true, arg_name)
                }
                // private input
                TokenType::Identifier(name) => (false, name),
                _ => {
                    return Err(Error {
                        error: ErrorTy::InvalidFunctionSignature("expected identifier"),
                        span: token.span,
                    });
                }
            };

            // :
            tokens.bump_expected(ctx, TokenType::Colon)?;

            // type
            let arg_typ = Ty::parse(ctx, tokens)?;

            // , or )
            let separator = tokens.bump_err(
                ctx,
                ErrorTy::InvalidFunctionSignature("expected end of function or other argument"),
            )?;

            match separator.typ {
                // (pub arg1: type1, arg2: type2)
                //                 ^
                TokenType::Comma => {
                    args.push((public, arg_name, arg_typ));
                }
                // (pub arg1: type1, arg2: type2)
                //                              ^
                TokenType::RightParen => {
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
                typ: TokenType::LeftCurlyBracket,
                ..
            }) => {
                return Ok(None);
            }
            _ => (),
        };

        tokens.bump_expected(ctx, TokenType::RightArrow)?;

        let return_type = Ty::parse(ctx, tokens)?;
        Ok(Some(return_type))
    }

    pub fn parse_fn_body(
        ctx: &mut ParserCtx,
        tokens: &mut Tokens,
    ) -> Result<Vec<Statement>, Error> {
        let mut body = vec![];

        tokens.bump_expected(ctx, TokenType::LeftCurlyBracket)?;

        loop {
            // end of the function
            let next_token = tokens.peek();
            if matches!(
                next_token,
                Some(Token {
                    typ: TokenType::RightCurlyBracket,
                    ..
                })
            ) {
                tokens.bump(ctx);
                break;
            }

            // parse next statement
            let statement = Statement::parse(ctx, tokens)?;
            body.push(statement);
        }

        Ok(body)
    }

    /// Parse a function, without the `fn` keyword.
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self, Error> {
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
    Assign { lhs: String, rhs: ExprKind },
    Assert(ExprKind),
    Return(ExprKind),
    Comment(String),
}

impl Statement {
    /// Returns a list of statement parsed until seeing the end of a block (`}`).
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self, Error> {
        let token = tokens.bump_err(ctx, ErrorTy::InvalidStatement)?;

        match token.typ {
            // assignment
            TokenType::Keyword(Keyword::Let) => {
                let lhs = parse_ident(ctx, tokens)?;
                tokens.bump_expected(ctx, TokenType::Equal)?;
                let rhs = ExprKind::parse(ctx, tokens)?;
                tokens.bump_expected(ctx, TokenType::SemiColon)?;
                Ok(Statement::Assign { lhs, rhs })
            }
            // assert
            TokenType::Keyword(Keyword::Assert) => {
                tokens.bump_expected(ctx, TokenType::LeftParen)?;
                let expr = ExprKind::parse(ctx, tokens)?;
                tokens.bump_expected(ctx, TokenType::RightParen)?;
                tokens.bump_expected(ctx, TokenType::SemiColon)?;
                Ok(Statement::Assert(expr))
            }
            // return
            TokenType::Keyword(Keyword::Return) => {
                let expr = ExprKind::parse(ctx, tokens)?;
                tokens.bump_expected(ctx, TokenType::SemiColon)?;
                Ok(Statement::Return(expr))
            }
            // comment
            TokenType::Comment(c) => Ok(Statement::Comment(c)),
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
pub enum Root {
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

            match &token.typ {
                // `use crypto::poseidon;`
                TokenType::Keyword(Keyword::Use) => {
                    if function_observed {
                        return Err(Error {
                            error: ErrorTy::UseAfterFn,
                            span: token.span,
                        });
                    }

                    let path = Path::parse_path(ctx, &mut tokens)?;
                    ast.push(Root::Use(path));

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

                // `fn main() { }`
                TokenType::Keyword(Keyword::Fn) => {
                    function_observed = true;

                    let func = Function::parse(ctx, &mut tokens)?;
                    ast.push(Root::Function(func));
                }

                // `// some comment`
                TokenType::Comment(comment) => {
                    ast.push(Root::Comment(comment.clone()));
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

pub fn parse_ident(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<String, Error> {
    let token = tokens.bump_err(ctx, ErrorTy::MissingToken)?;
    match token.typ {
        TokenType::Identifier(ident) => Ok(ident),
        _ => Err(Error {
            error: ErrorTy::ExpectedToken(TokenType::Identifier("".to_string())),
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
        let parsed = Statement::parse(ctx, tokens).unwrap();
        println!("{:?}", parsed);
    }

    #[test]
    fn statement_assert() {
        let code = r#"assert(digest == public_input);"#;
        let tokens = &mut Token::parse(code).unwrap();
        let ctx = &mut ParserCtx::default();
        let parsed = Statement::parse(ctx, tokens).unwrap();
        println!("{:?}", parsed);
    }
}
