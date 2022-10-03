use std::fmt::Display;

use crate::{
    constants::{Field, Span},
    error::{Error, ErrorKind, Result},
    lexer::{Keyword, Token, TokenKind},
    syntax::is_type,
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

/// A context for the parser.
#[derive(Debug, Default)]
pub struct ParserCtx {
    /// A counter used to uniquely identify different nodes in the AST.
    pub node_id: usize,

    /// Used mainly for error reporting,
    /// when we don't have a token to read
    // TODO: replace with `(usize::MAX, usize::MAX)`?
    pub last_token: Option<Token>,
}

impl ParserCtx {
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

pub fn parse_type_declaration(
    ctx: &mut ParserCtx,
    tokens: &mut Tokens,
    ident: Ident,
) -> Result<Expr> {
    if !is_type(&ident.value) {
        panic!("this looks like a type declaration but not on a type (types start with an uppercase) (TODO: better error)");
    }

    // Thing { x: 1, y: 2 }
    //       ^
    tokens.bump(ctx);

    let mut fields = vec![];

    // Thing { x: 1, y: 2 }
    //         ^^^^^^^^^^^^
    loop {
        // Thing { x: 1, y: 2 }
        //                    ^
        if let Some(Token {
            kind: TokenKind::RightCurlyBracket,
            ..
        }) = tokens.peek()
        {
            tokens.bump(ctx);
            break;
        };

        // Thing { x: 1, y: 2 }
        //         ^
        let field_name = Ident::parse(ctx, tokens)?;

        // Thing { x: 1, y: 2 }
        //          ^
        tokens.bump_expected(ctx, TokenKind::Colon)?;

        // Thing { x: 1, y: 2 }
        //            ^
        let field_value = Expr::parse(ctx, tokens)?;
        fields.push((field_name, field_value));

        // Thing { x: 1, y: 2 }
        //             ^      ^
        match tokens.bump_err(ctx, ErrorKind::InvalidEndOfLine)? {
            Token {
                kind: TokenKind::Comma,
                ..
            } => (),
            Token {
                kind: TokenKind::RightCurlyBracket,
                ..
            } => break,
            _ => {
                return Err(Error {
                    kind: ErrorKind::InvalidEndOfLine,
                    span: ctx.last_span(),
                })
            }
        };
    }

    let span = ident.span.merge_with(ctx.last_span());

    Ok(Expr::new(
        ctx,
        ExprKind::CustomTypeDeclaration {
            struct_name: ident,
            fields,
        },
        span,
    ))
}

pub fn parse_fn_call_args(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Vec<Expr>> {
    tokens.bump(ctx); // (

    let mut args = vec![];
    loop {
        let pp = tokens.peek();

        match pp {
            Some(x) => match x.kind {
                // ,
                TokenKind::Comma => {
                    tokens.bump(ctx);
                }

                // )
                TokenKind::RightParen => {
                    tokens.bump(ctx);
                    break;
                }

                // an argument (as expression)
                _ => {
                    let arg = Expr::parse(ctx, tokens)?;

                    args.push(arg);
                }
            },

            None => {
                return Err(Error {
                    kind: ErrorKind::InvalidFnCall("unexpected end of function call"),
                    span: ctx.last_span(),
                })
            }
        }
    }

    Ok(args)
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
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TyKind {
    /// The main primitive type. 'Nuf said.
    Field,

    /// Custom / user-defined types
    Custom(String),

    /// This could be the same as Field, but we use this to also track the fact that it's a constant.
    // TODO: get rid of this type tho no?
    BigInt,

    /// An array of a fixed size.
    Array(Box<TyKind>, u32),

    /// A boolean (`true` or `false`).
    Bool,
    // Tuple(Vec<TyKind>),
    // Bool,
    // U8,
    // U16,
    // U32,
    // U64,
}

impl Display for TyKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TyKind::Custom(name) => write!(f, "{}", name),
            TyKind::Field => write!(f, "Field"),
            TyKind::BigInt => write!(f, "BigInt"),
            TyKind::Array(ty, size) => write!(f, "[{}; {}]", ty, size),
            TyKind::Bool => write!(f, "Bool"),
        }
    }
}

impl Ty {
    pub fn reserved_types(ty_name: &str) -> TyKind {
        match ty_name {
            "Field" => TyKind::Field,
            "Bool" => TyKind::Bool,
            _ => TyKind::Custom(ty_name.to_string()),
        }
    }

    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self> {
        let token = tokens.bump_err(ctx, ErrorKind::MissingType)?;
        match token.kind {
            // struct name
            TokenKind::Identifier(name) => {
                if !is_type(&name) {
                    panic!("bad name for a type (TODO: better error)");
                }

                let ty_kind = Self::reserved_types(&name);
                Ok(Self {
                    kind: ty_kind,
                    span: token.span,
                })
            }

            // array
            // [type; size]
            // ^
            TokenKind::LeftBracket => {
                let span = Span(token.span.0, 0);

                // [type; size]
                //   ^
                let ty = Ty::parse(ctx, tokens)?;

                // [type; size]
                //      ^
                tokens.bump_expected(ctx, TokenKind::SemiColon)?;

                // [type; size]
                //         ^
                let siz = tokens.bump_err(ctx, ErrorKind::InvalidToken)?;
                let siz: u32 = match siz.kind {
                    TokenKind::BigInt(s) => s.parse().map_err(|_e| Error {
                        kind: ErrorKind::InvalidArraySize,
                        span: siz.span,
                    })?,
                    _ => {
                        return Err(Error {
                            kind: ErrorKind::ExpectedToken(TokenKind::BigInt("".to_string())),
                            span: siz.span,
                        });
                    }
                };

                // [type; size]
                //            ^
                let right_paren = tokens.bump_expected(ctx, TokenKind::RightBracket)?;

                let span = span.merge_with(right_paren.span);

                Ok(Ty {
                    kind: TyKind::Array(Box::new(ty.kind), siz),
                    span,
                })
            }

            // unrecognized
            _ => Err(Error {
                kind: ErrorKind::InvalidType,
                span: token.span,
            }),
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
pub struct Expr {
    pub node_id: usize,
    pub kind: ExprKind,
    pub span: Span,
}

impl Expr {
    pub fn new(ctx: &mut ParserCtx, kind: ExprKind, span: Span) -> Self {
        let node_id = ctx.next_node_id();
        Self {
            node_id,
            kind,
            span,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    /// `lhs(args)`
    FnCall {
        module: Option<Ident>,
        fn_name: Ident,
        args: Vec<Expr>,
    },

    /// `lhs.method_name(args)`
    MethodCall {
        lhs: Box<Expr>,
        method_name: Ident,
        args: Vec<Expr>,
    },

    /// `let lhs = rhs`
    Assignment { lhs: Box<Expr>, rhs: Box<Expr> },

    /// `lhs.rhs`
    FieldAccess { lhs: Box<Expr>, rhs: Ident },

    /// `lhs <op> rhs`
    Op {
        op: Op2,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },

    /// `!expr`
    Negated(Box<Expr>),

    /// any numbers
    BigInt(String),

    /// a variable. For example, `mod::A`, `x`, `y`, etc.
    // TODO: change to `identifier` or `path`?
    Variable { module: Option<Ident>, name: Ident },

    /// An array access, for example:
    /// `lhs[idx]`
    ArrayAccess {
        module: Option<Ident>,
        name: Ident,
        idx: Box<Expr>,
    },

    /// `[ ... ]`
    ArrayDeclaration(Vec<Expr>),

    /// `name { fields }`
    CustomTypeDeclaration {
        struct_name: Ident,
        fields: Vec<(Ident, Expr)>,
    },

    /// `true` or `false`
    Bool(bool),
}

#[derive(Debug, Clone)]
pub enum Op2 {
    Addition,
    Subtraction,
    Multiplication,
    Division,
    Equality,
    BoolAnd,
    BoolOr,
    BoolNot,
}

impl Expr {
    /// Parses until it finds something it doesn't know, then returns without consuming the token it doesn't know (the caller will have to make sense of it)
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self> {
        let token = tokens.bump_err(ctx, ErrorKind::MissingExpression)?;
        let span = token.span;

        let lhs = match token.kind {
            // numeric
            TokenKind::BigInt(b) => Expr::new(ctx, ExprKind::BigInt(b), span),

            // identifier
            TokenKind::Identifier(value) => {
                let maybe_module = Ident { value, span };

                // is it a qualified identifier?
                // name::other_name
                //     ^^
                match tokens.peek() {
                    Some(Token {
                        kind: TokenKind::DoubleColon,
                        ..
                    }) => {
                        tokens.bump(ctx); // ::

                        // mod::expr
                        //      ^^^^
                        let name = match tokens.bump(ctx) {
                            Some(Token {
                                kind: TokenKind::Identifier(value),
                                span,
                            }) => Ident { value, span },
                            _ => panic!("cannot qualify a non-identifier"),
                        };

                        Expr::new(
                            ctx,
                            ExprKind::Variable {
                                module: Some(maybe_module),
                                name,
                            },
                            span,
                        )
                    }

                    // just an identifier
                    _ => Expr::new(
                        ctx,
                        ExprKind::Variable {
                            module: None,
                            name: maybe_module,
                        },
                        span,
                    ),
                }
            }

            // negated expr
            TokenKind::Minus => {
                let expr = Expr::parse(ctx, tokens)?;

                Expr::new(ctx, ExprKind::Negated(Box::new(expr)), span)
            }

            // parenthesis
            TokenKind::LeftParen => {
                let expr = Expr::parse(ctx, tokens)?;
                tokens.bump_expected(ctx, TokenKind::RightParen)?;
                expr
            }

            // true or false
            TokenKind::Keyword(Keyword::True) | TokenKind::Keyword(Keyword::False) => {
                let is_true = matches!(token.kind, TokenKind::Keyword(Keyword::True));

                Expr::new(ctx, ExprKind::Bool(is_true), span)
            }

            // negation (logical NOT)
            TokenKind::Exclamation => {
                let expr = Expr::parse(ctx, tokens)?;

                Expr::new(ctx, ExprKind::Negated(Box::new(expr)), span)
            }

            // array declaration
            TokenKind::LeftBracket => {
                let mut items = vec![];
                let last_span;

                // [1, 2];
                //  ^^^^
                loop {
                    let token = tokens.peek();

                    // [1, 2];
                    //      ^
                    if let Some(Token {
                        kind: TokenKind::RightBracket,
                        span,
                    }) = token
                    {
                        last_span = span;
                        tokens.bump(ctx);
                        break;
                    };

                    // [1, 2];
                    //  ^
                    let item = Expr::parse(ctx, tokens)?;
                    items.push(item);

                    // [1, 2];
                    //   ^  ^
                    let token = tokens.bump_err(ctx, ErrorKind::InvalidEndOfLine)?;
                    match &token.kind {
                        TokenKind::RightBracket => {
                            last_span = token.span;
                            break;
                        }
                        TokenKind::Comma => (),
                        _ => {
                            return Err(Error {
                                kind: ErrorKind::InvalidEndOfLine,
                                span: token.span,
                            })
                        }
                    };
                }

                if items.is_empty() {
                    panic!("empty array declaration (TODO: better error)");
                }

                Expr::new(
                    ctx,
                    ExprKind::ArrayDeclaration(items),
                    span.merge_with(last_span),
                )
            }

            // unrecognized pattern
            _ => {
                return Err(Error {
                    kind: ErrorKind::InvalidExpression,
                    span: token.span,
                });
            }
        };

        // continue parsing. Potentially there's more
        lhs.parse_rhs(ctx, tokens)
    }

    /// an expression is sometimes unfinished when we parse it with [Self::parse],
    /// we use this function to see if the expression we just parsed (`self`) is actually part of a bigger expression
    fn parse_rhs(self, ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Expr> {
        // we peek into what's next to see if there's an expression that uses
        // the expression `self` we just parsed.
        // warning: ALL of the rules here should make use of `self`.
        let lhs = match tokens.peek() {
            // assignment
            Some(Token {
                kind: TokenKind::Equal,
                span,
            }) => {
                tokens.bump(ctx); // =

                // sanitize
                if !matches!(
                    &self.kind,
                    ExprKind::Variable { .. } | ExprKind::ArrayAccess { .. },
                ) {
                    return Err(Error {
                        kind: ErrorKind::InvalidAssignmentExpression,
                        span: self.span.merge_with(span),
                    });
                }

                let rhs = Expr::parse(ctx, tokens)?;
                let span = self.span.merge_with(rhs.span);

                Expr::new(
                    ctx,
                    ExprKind::Assignment {
                        lhs: Box::new(self),
                        rhs: Box::new(rhs),
                    },
                    span,
                )
            }

            // binary operation
            Some(Token {
                kind:
                    TokenKind::Plus
                    | TokenKind::Minus
                    | TokenKind::Star
                    | TokenKind::Slash
                    | TokenKind::DoubleEqual
                    | TokenKind::Ampersand
                    | TokenKind::Pipe
                    | TokenKind::Exclamation,
                span,
            }) => {
                // lhs + rhs
                //     ^
                let op = match tokens.bump(ctx).unwrap().kind {
                    TokenKind::Plus => Op2::Addition,
                    TokenKind::Minus => Op2::Subtraction,
                    TokenKind::Star => Op2::Multiplication,
                    TokenKind::Slash => Op2::Division,
                    TokenKind::DoubleEqual => Op2::Equality,
                    TokenKind::Ampersand => Op2::BoolAnd,
                    TokenKind::Pipe => Op2::BoolOr,
                    TokenKind::Exclamation => Op2::BoolNot,
                    _ => unreachable!(),
                };

                // lhs + rhs
                //       ^^^
                let rhs = Expr::parse(ctx, tokens)?;

                let span = span.merge_with(rhs.span);
                Expr::new(
                    ctx,
                    ExprKind::Op {
                        op,
                        lhs: Box::new(self),
                        rhs: Box::new(rhs),
                    },
                    span,
                )
            }

            // type declaration
            Some(Token {
                kind: TokenKind::LeftCurlyBracket,
                ..
            }) => {
                let ident = match self.kind {
                    ExprKind::Variable { module, name } => {
                        if module.is_some() {
                            panic!("a type declaration cannot be qualified");
                        }

                        name
                    }
                    _ => panic!("bad type declaration"),
                };
                parse_type_declaration(ctx, tokens, ident)?
            }

            // array access
            Some(Token {
                kind: TokenKind::LeftBracket,
                ..
            }) => {
                tokens.bump(ctx); // [

                // sanitize array
                let (module, name) = match self.kind {
                    ExprKind::Variable { module, name } => (module, name),
                    _ => panic!("array access on a non-variable"),
                };

                // array[idx]
                //       ^^^
                let idx = Expr::parse(ctx, tokens)?;

                // array[idx]
                //          ^
                tokens.bump_expected(ctx, TokenKind::RightBracket)?;

                let span = self.span.merge_with(idx.span);

                Expr::new(
                    ctx,
                    ExprKind::ArrayAccess {
                        module,
                        name,
                        idx: Box::new(idx),
                    },
                    span,
                )
            }

            // fn call
            Some(Token {
                kind: TokenKind::LeftParen,
                span,
            }) => {
                // sanitize
                let (module, fn_name) = match self.kind {
                    ExprKind::Variable { module, name } => (module, name),
                    _ => panic!("invalid fn name"),
                };

                // parse the arguments
                let args = parse_fn_call_args(ctx, tokens)?;

                let span = if let Some(arg) = args.last() {
                    span.merge_with(arg.span)
                } else {
                    span
                };

                Expr::new(
                    ctx,
                    ExprKind::FnCall {
                        module,
                        fn_name,
                        args,
                    },
                    span,
                )
            }

            // field access or method call
            // thing.thing2
            //      ^
            Some(Token {
                kind: TokenKind::Dot,
                ..
            }) => {
                tokens.bump(ctx); // .

                // sanitize
                if !matches!(
                    &self.kind,
                    ExprKind::FieldAccess { .. }
                        | ExprKind::Variable { .. }
                        | ExprKind::ArrayAccess { .. }
                ) {
                    panic!("field or method calls can only follow a field of another struct, a struct, or an array access (TODO: better error)");
                }

                // lhs.field
                //     ^^^^^
                let rhs = Ident::parse(ctx, tokens)?;
                let span = self.span.merge_with(rhs.span);

                // lhs.field or lhs.method_name()
                //     ^^^^^        ^^^^^^^^^^^^^
                match tokens.peek() {
                    // method call:
                    // lhs.method_name(...)
                    //     ^^^^^^^^^^^^^^^^
                    Some(Token {
                        kind: TokenKind::LeftParen,
                        ..
                    }) => {
                        // lhs.method_name(args)
                        //                 ^^^^
                        let args = parse_fn_call_args(ctx, tokens)?;

                        Expr::new(
                            ctx,
                            ExprKind::MethodCall {
                                lhs: Box::new(self),
                                method_name: rhs,
                                args,
                            },
                            span,
                        )
                    }

                    // field access
                    // lhs.field
                    //     ^^^^^
                    _ => Expr::new(
                        ctx,
                        ExprKind::FieldAccess {
                            lhs: Box::new(self),
                            rhs,
                        },
                        span,
                    ),
                }
            }

            // it looks like the lhs is a valid expression in itself
            _ => return Ok(self),
        };

        lhs.parse_rhs(ctx, tokens)
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

#[derive(Debug, Default, Clone)]
pub struct FnSig {
    pub name: FnNameDef,

    /// (pub, ident, type)
    pub arguments: Vec<FnArg>,

    pub return_type: Option<Ty>,
}

impl FnSig {
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self> {
        let name = FnNameDef::parse(ctx, tokens)?;

        let arguments = Function::parse_args(ctx, tokens, name.self_name.as_ref())?;

        let return_type = Function::parse_fn_return_type(ctx, tokens)?;

        Ok(Self {
            name,
            arguments,
            return_type,
        })
    }
}

/// Any kind of text that can represent a type, a variable, a function name, etc.
#[derive(Debug, Default, Clone)]
pub struct Ident {
    pub value: String,
    pub span: Span,
}

impl Ident {
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self> {
        let token = tokens.bump_err(ctx, ErrorKind::MissingToken)?;
        match token.kind {
            TokenKind::Identifier(ident) => Ok(Self {
                value: ident,
                span: token.span,
            }),

            _ => Err(Error {
                kind: ErrorKind::ExpectedToken(TokenKind::Identifier("".to_string())),
                span: token.span,
            }),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AttributeKind {
    Pub,
}

impl AttributeKind {
    pub fn is_public(&self) -> bool {
        matches!(self, Self::Pub)
    }
}

#[derive(Debug, Clone)]
pub struct Attribute {
    pub kind: AttributeKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub sig: FnSig,

    pub body: Vec<Stmt>,

    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct FnArg {
    pub name: Ident,
    pub typ: Ty,
    pub attribute: Option<Attribute>,
    pub span: Span,
}

impl FnArg {
    pub fn is_public(&self) -> bool {
        matches!(
            self.attribute,
            Some(Attribute {
                kind: AttributeKind::Pub,
                ..
            })
        )
    }
}

/// Represents the name of a function.
#[derive(Debug, Clone, Default)]
pub struct FnNameDef {
    /// The name of the type that this function is implemented on.
    pub self_name: Option<Ident>,

    /// The name of the function.
    pub name: Ident,

    /// The span of the function.
    pub span: Span,
}

impl FnNameDef {
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self> {
        // fn House.verify(   or   fn verify(
        //    ^^^^^                   ^^^^^
        let maybe_self_name = tokens.bump_ident(
            ctx,
            ErrorKind::InvalidFunctionSignature("expected function name"),
        )?;
        let span = maybe_self_name.span;

        // fn House.verify(
        //    ^^^^^
        if is_type(&maybe_self_name.value) {
            // fn House.verify(
            //         ^
            tokens.bump_expected(ctx, TokenKind::Dot)?;

            // fn House.verify(
            //          ^^^^^^
            let name = tokens.bump_ident(
                ctx,
                ErrorKind::InvalidFunctionSignature("expected function name"),
            )?;

            let span = span.merge_with(name.span);

            Ok(Self {
                self_name: Some(maybe_self_name),
                name,
                span,
            })
        } else {
            // fn verify(
            //    ^^^^^^
            Ok(Self {
                self_name: None,
                name: maybe_self_name,
                span,
            })
        }
    }
}

impl Function {
    pub fn is_main(&self) -> bool {
        self.sig.name.name.value == "main"
    }

    pub fn parse_args(
        ctx: &mut ParserCtx,
        tokens: &mut Tokens,
        self_name: Option<&Ident>,
    ) -> Result<Vec<FnArg>> {
        // (pub arg1: type1, arg2: type2)
        // ^
        let Token { span, .. } = tokens.bump_expected(ctx, TokenKind::LeftParen)?;

        // (pub arg1: type1, arg2: type2)
        //   ^
        let mut args = vec![];

        loop {
            // `pub arg1: type1`
            //   ^   ^
            let token = tokens.bump_err(
                ctx,
                ErrorKind::InvalidFunctionSignature("expected function arguments"),
            )?;

            let (public, arg_name) = match token.kind {
                TokenKind::RightParen => break,
                // public input
                TokenKind::Keyword(Keyword::Pub) => {
                    let arg_name = Ident::parse(ctx, tokens)?;
                    (
                        Some(Attribute {
                            kind: AttributeKind::Pub,
                            span: token.span,
                        }),
                        arg_name,
                    )
                }
                // private input
                TokenKind::Identifier(name) => (
                    None,
                    Ident {
                        value: name,
                        span: token.span,
                    },
                ),
                _ => {
                    return Err(Error {
                        kind: ErrorKind::InvalidFunctionSignature("expected identifier"),
                        span: token.span,
                    });
                }
            };

            // self takes no value
            let arg_typ = if arg_name.value == "self" {
                let self_name = self_name.ok_or(Error {
                    kind: ErrorKind::InvalidFunctionSignature(
                        "the `self` argynebt is only allowed in struct methods",
                    ),
                    span: arg_name.span,
                })?;

                if !args.is_empty() {
                    return Err(Error {
                        kind: ErrorKind::InvalidFunctionSignature(
                            "`self` must be the first argument",
                        ),
                        span: arg_name.span,
                    });
                }

                Ty {
                    kind: TyKind::Custom(self_name.value.clone()),
                    span: self_name.span,
                }
            } else {
                // :
                tokens.bump_expected(ctx, TokenKind::Colon)?;

                // type
                Ty::parse(ctx, tokens)?
            };

            // , or )
            let separator = tokens.bump_err(
                ctx,
                ErrorKind::InvalidFunctionSignature("expected end of function or other argument"),
            )?;

            let span = span.merge_with(separator.span);
            let arg = FnArg {
                name: arg_name,
                typ: arg_typ,
                attribute: public,
                span,
            };
            args.push(arg);

            match separator.kind {
                // (pub arg1: type1, arg2: type2)
                //                 ^
                TokenKind::Comma => (),
                // (pub arg1: type1, arg2: type2)
                //                              ^
                TokenKind::RightParen => break,
                _ => {
                    return Err(Error {
                        kind: ErrorKind::InvalidFunctionSignature(
                            "expected end of function or other argument",
                        ),
                        span: separator.span,
                    });
                }
            }
        }

        Ok(args)
    }

    pub fn parse_fn_return_type(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Option<Ty>> {
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

    pub fn parse_fn_body(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Vec<Stmt>> {
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
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self> {
        // ghetto way of getting the span of the function: get the span of the first token (name), then try to get the span of the last token
        let mut span = tokens
            .peek()
            .ok_or(Error {
                kind: ErrorKind::InvalidFunctionSignature("expected function name"),
                span: ctx.last_span(),
            })?
            .span;

        let name = FnNameDef::parse(ctx, tokens)?;
        let arguments = Self::parse_args(ctx, tokens, name.self_name.as_ref())?;
        let return_type = Self::parse_fn_return_type(ctx, tokens)?;
        let body = Self::parse_fn_body(ctx, tokens)?;

        // here's the last token, that is if the function is not empty (maybe we should disallow empty functions?)

        if let Some(t) = body.last() {
            span.1 = (t.span.0 + t.span.1) - span.0;
        } else {
            return Err(Error {
                kind: ErrorKind::InvalidFunctionSignature("expected function body"),
                span: ctx.last_span(),
            });
        }

        let func = Self {
            sig: FnSig {
                name,
                arguments,
                return_type,
            },
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
// ## Statements
//
//~ statement ::=
//~     | "let" ident "=" expr ";"
//~     | expr ";"
//~     | "return" expr ";"
//~
//~ where an expression is allowed only if it is a function call that does not return a value.
//~
//~ Actually currently we don't implement it this way.
//~ We don't expect an expression to be a statement,
//~ but a well defined function call:
//~
//~ fn_call ::= path "(" [ expr { "," expr } ] ")"
//~ path ::= ident { "::" ident }
//~

#[derive(Debug, Clone)]
pub struct Range {
    pub start: u32,
    pub end: u32,
    pub span: Span,
}

impl Range {
    pub fn range(&self) -> std::ops::Range<u32> {
        self.start..self.end
    }
}

#[derive(Debug, Clone)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum StmtKind {
    Assign {
        mutable: bool,
        lhs: Ident,
        rhs: Box<Expr>,
    },
    Expr(Box<Expr>),
    Return(Box<Expr>),
    Comment(String),
    For {
        var: Ident,
        range: Range,
        body: Vec<Stmt>,
    },
}

impl Stmt {
    /// Returns a list of statement parsed until seeing the end of a block (`}`).
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self> {
        match tokens.peek() {
            None => Err(Error {
                kind: ErrorKind::InvalidStatement,
                span: ctx.last_span(),
            }),
            // assignment
            Some(Token {
                kind: TokenKind::Keyword(Keyword::Let),
                span,
            }) => {
                let mut span = span;
                tokens.bump(ctx);

                // let mut x = 5;
                //     ^^^

                let mutable = if matches!(
                    tokens.peek(),
                    Some(Token {
                        kind: TokenKind::Keyword(Keyword::Mut),
                        ..
                    })
                ) {
                    tokens.bump(ctx);
                    true
                } else {
                    false
                };

                // let mut x = 5;
                //         ^
                let lhs = Ident::parse(ctx, tokens)?;

                // let mut x = 5;
                //           ^
                tokens.bump_expected(ctx, TokenKind::Equal)?;

                // let mut x = 5;
                //             ^
                let rhs = Box::new(Expr::parse(ctx, tokens)?);

                span.1 = rhs.span.1 + rhs.span.0 - span.0;

                // let mut x = 5;
                //              ^
                tokens.bump_expected(ctx, TokenKind::SemiColon)?;

                //
                Ok(Stmt {
                    kind: StmtKind::Assign { mutable, lhs, rhs },
                    span,
                })
            }

            // for loop
            Some(Token {
                kind: TokenKind::Keyword(Keyword::For),
                span,
            }) => {
                tokens.bump(ctx);

                // for i in 0..5 { ... }
                //     ^
                let var = Ident::parse(ctx, tokens)?;

                // for i in 0..5 { ... }
                //       ^^
                tokens.bump_expected(ctx, TokenKind::Keyword(Keyword::In))?;

                // for i in 0..5 { ... }
                //          ^
                let (start, start_span) = match tokens.bump(ctx) {
                    Some(Token {
                        kind: TokenKind::BigInt(n),
                        span,
                    }) => {
                        let start: u32 = n.parse().map_err(|_e| Error {
                            kind: ErrorKind::InvalidRangeSize,
                            span,
                        })?;
                        (start, span)
                    }
                    _ => {
                        return Err(Error {
                            kind: ErrorKind::ExpectedToken(TokenKind::BigInt("".to_string())),
                            span: ctx.last_span(),
                        })
                    }
                };

                // for i in 0..5 { ... }
                //           ^^
                tokens.bump_expected(ctx, TokenKind::DoubleDot)?;

                // for i in 0..5 { ... }
                //             ^
                let (end, end_span) = match tokens.bump(ctx) {
                    Some(Token {
                        kind: TokenKind::BigInt(n),
                        span,
                    }) => {
                        let end: u32 = n.parse().map_err(|_e| Error {
                            kind: ErrorKind::InvalidRangeSize,
                            span,
                        })?;
                        (end, span)
                    }
                    _ => {
                        return Err(Error {
                            kind: ErrorKind::ExpectedToken(TokenKind::BigInt("".to_string())),
                            span: ctx.last_span(),
                        })
                    }
                };

                let range = Range {
                    start,
                    end,
                    span: start_span.merge_with(end_span),
                };

                // for i in 0..5 { ... }
                //               ^
                tokens.bump_expected(ctx, TokenKind::LeftCurlyBracket)?;

                // for i in 0..5 { ... }
                //                 ^^^
                let mut body = vec![];

                loop {
                    // for i in 0..5 { ... }
                    //                     ^
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
                    // TODO: should we prevent `return` here?
                    // TODO: in general, do we prevent early returns atm?
                    let statement = Stmt::parse(ctx, tokens)?;
                    body.push(statement);
                }

                //
                Ok(Stmt {
                    kind: StmtKind::For { var, range, body },
                    span,
                })
            }

            // if/else
            Some(Token {
                kind: TokenKind::Keyword(Keyword::If),
                span: _,
            }) => {
                // TODO: wait, this should be implemented as an expresssion! not a statement
                todo!()
            }

            // return
            Some(Token {
                kind: TokenKind::Keyword(Keyword::Return),
                span,
            }) => {
                tokens.bump(ctx);

                // return xx;
                //        ^^
                let expr = Expr::parse(ctx, tokens)?;

                // return xx;
                //          ^
                tokens.bump_expected(ctx, TokenKind::SemiColon)?;

                Ok(Stmt {
                    kind: StmtKind::Return(Box::new(expr)),
                    span,
                })
            }

            // comment
            Some(Token {
                kind: TokenKind::Comment(c),
                span,
            }) => {
                tokens.bump(ctx);
                Ok(Stmt {
                    kind: StmtKind::Comment(c),
                    span,
                })
            }

            // statement expression (like function call)
            _ => {
                let expr = Expr::parse(ctx, tokens)?;
                let span = expr.span;

                tokens.bump_expected(ctx, TokenKind::SemiColon)?;

                Ok(Stmt {
                    kind: StmtKind::Expr(Box::new(expr)),
                    span,
                })
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
    pub span: Span,
}

#[derive(Debug)]
pub struct UsePath {
    pub module: Ident,
    pub submodule: Ident,
    pub span: Span,
}

impl UsePath {
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self> {
        let module = tokens.bump_ident(
            ctx,
            ErrorKind::InvalidPath("wrong path: expected a module (TODO: better error"),
        )?;
        let span = module.span;

        tokens.bump_expected(ctx, TokenKind::DoubleColon)?; // ::

        let submodule = tokens.bump_ident(
            ctx,
            ErrorKind::InvalidPath(
                "wrong path: expected a submodule after `::` (TODO: better error",
            ),
        )?;

        let span = span.merge_with(submodule.span);
        Ok(UsePath {
            module,
            submodule,
            span,
        })
    }
}

#[derive(Debug)]
pub enum RootKind {
    Use(UsePath),
    Function(Function),
    Comment(String),
    Struct(Struct),
    Const(Const),
}

//
// Const
//

#[derive(Debug)]
pub struct Const {
    pub name: Ident,
    pub value: Field,
    pub span: Span,
}

impl Const {
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self> {
        // const foo = 42;
        //       ^^^
        let name = Ident::parse(ctx, tokens)?;

        // const foo = 42;
        //           ^
        tokens.bump_expected(ctx, TokenKind::Equal)?;

        // const foo = 42;
        //             ^^
        let value = Expr::parse(ctx, tokens)?;
        let value = match &value.kind {
            ExprKind::BigInt(s) => s.parse().map_err(|_e| Error {
                kind: ErrorKind::InvalidField(s.clone()),
                span: value.span,
            })?,
            _ => {
                return Err(Error {
                    kind: ErrorKind::InvalidConstType,
                    span: value.span,
                });
            }
        };

        // const foo = 42;
        //               ^
        tokens.bump_expected(ctx, TokenKind::SemiColon)?;

        //
        let span = name.span;
        Ok(Const { name, value, span })
    }
}

//
// Custom Struct
//

#[derive(Debug)]
pub struct Struct {
    //pub attribute: Attribute,
    pub name: CustomType,
    pub fields: Vec<(Ident, Ty)>,
    pub span: Span,
}

impl Struct {
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self> {
        // ghetto way of getting the span of the function: get the span of the first token (name), then try to get the span of the last token
        let span = tokens
            .peek()
            .ok_or(Error {
                kind: ErrorKind::InvalidFunctionSignature("expected function name"),
                span: ctx.last_span(),
            })?
            .span;

        // struct Foo { a: Field, b: Field }
        //        ^^^

        let name = parse_type(ctx, tokens)?;

        // struct Foo { a: Field, b: Field }
        //            ^
        tokens.bump_expected(ctx, TokenKind::LeftCurlyBracket)?;

        let mut fields = vec![];
        loop {
            // struct Foo { a: Field, b: Field }
            //                                 ^
            if let Some(Token {
                kind: TokenKind::RightCurlyBracket,
                ..
            }) = tokens.peek()
            {
                tokens.bump(ctx);
                break;
            }
            // struct Foo { a: Field, b: Field }
            //              ^
            let field_name = Ident::parse(ctx, tokens)?;

            // struct Foo { a: Field, b: Field }
            //               ^
            tokens.bump_expected(ctx, TokenKind::Colon)?;

            // struct Foo { a: Field, b: Field }
            //                 ^^^^^
            let field_ty = Ty::parse(ctx, tokens)?;
            fields.push((field_name, field_ty));

            // struct Foo { a: Field, b: Field }
            //                      ^          ^
            match tokens.peek() {
                Some(Token {
                    kind: TokenKind::Comma,
                    ..
                }) => {
                    tokens.bump(ctx);
                }
                Some(Token {
                    kind: TokenKind::RightCurlyBracket,
                    ..
                }) => {
                    tokens.bump(ctx);
                    break;
                }
                _ => {
                    return Err(Error {
                        kind: ErrorKind::ExpectedToken(TokenKind::Comma),
                        span: ctx.last_span(),
                    })
                }
            }
        }

        // figure out the span
        let span = span.merge_with(ctx.last_span());

        //
        Ok(Struct { name, fields, span })
    }
}

//
// AST
//

#[derive(Debug, Default)]
pub struct AST(pub Vec<Root>);

impl AST {
    pub fn parse(mut tokens: Tokens) -> Result<AST> {
        let mut ast = vec![];
        let ctx = &mut ParserCtx::default();

        // use statements must appear first
        let mut function_observed = false;

        while let Some(token) = tokens.bump(ctx) {
            match &token.kind {
                // `use crypto::poseidon;`
                TokenKind::Keyword(Keyword::Use) => {
                    if function_observed {
                        return Err(Error {
                            kind: ErrorKind::UseAfterFn,
                            span: token.span,
                        });
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
                        return Err(Error {
                            kind: ErrorKind::InvalidEndOfLine,
                            span: token.span,
                        });
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
                    return Err(Error {
                        kind: ErrorKind::InvalidToken,
                        span: token.span,
                    });
                }
            }
        }

        Ok(Self(ast))
    }
}

//
// CustomType
//

#[derive(Debug)]
pub struct CustomType {
    pub value: String,
    pub span: Span,
}

// TODO: implement as impl CustomType
pub fn parse_type(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<CustomType> {
    let ty_name = tokens.bump_ident(ctx, ErrorKind::InvalidType)?;

    if !is_type(&ty_name.value) {
        panic!("type name should start with uppercase letter (TODO: better error");
    }

    // make sure that this type is allowed
    if !matches!(Ty::reserved_types(&ty_name.value), TyKind::Custom(_)) {
        return Err(Error {
            kind: ErrorKind::ReservedType(ty_name.value),
            span: ty_name.span,
        });
    }

    Ok(CustomType {
        value: ty_name.value,
        span: ty_name.span,
    })
}

//
// Tests
//

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fn_signature() {
        let code = r#"main(pub public_input: [Fel; 3], private_input: [Fel; 3]) -> [Fel; 3] { return public_input; }"#;
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
