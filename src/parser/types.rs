use educe::Educe;
use std::{
    fmt::Display,
    hash::{Hash, Hasher},
    str::FromStr,
};

use ark_ff::Field;
use serde::{Deserialize, Serialize};

use crate::{
    cli::packages::UserRepo,
    constants::Span,
    error::{ErrorKind, Result},
    lexer::{Keyword, Token, TokenKind, Tokens},
    stdlib::BUILTIN_FN_NAMES,
    syntax::is_type,
};

use super::{CustomType, Expr, ExprKind, ParserCtx, StructDef};

pub fn parse_type_declaration(
    ctx: &mut ParserCtx,
    tokens: &mut Tokens,
    ident: Ident,
) -> Result<Expr> {
    if !is_type(&ident.value) {
        return Err(ctx.error(
            ErrorKind::UnexpectedError(
                "this looks like a type declaration but not on a type (types start with an uppercase)",
            ), ident.span));
    }

    let mut span = ident.span;

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
        span = span.merge_with(field_value.span);
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
            _ => return Err(ctx.error(ErrorKind::InvalidEndOfLine, ctx.last_span())),
        };
    }

    Ok(Expr::new(
        ctx,
        ExprKind::CustomTypeDeclaration {
            custom: CustomType {
                module: ModulePath::Local,
                name: ident.value,
                span: ident.span,
            },
            fields,
        },
        span,
    ))
}

pub fn parse_fn_call_args(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<(Vec<Expr>, Span)> {
    let start = tokens.bump(ctx).expect("parser error: parse_fn_call_args"); // (
    let mut span = start.span;

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
                    let end = tokens.bump(ctx).unwrap();
                    span = span.merge_with(end.span);
                    break;
                }

                // an argument (as expression)
                _ => {
                    let arg = Expr::parse(ctx, tokens)?;

                    args.push(arg);
                }
            },

            None => {
                return Err(ctx.error(
                    ErrorKind::InvalidFnCall("unexpected end of function call"),
                    ctx.last_span(),
                ))
            }
        }
    }

    Ok((args, span))
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ty {
    pub kind: TyKind,
    pub span: Span,
}

/// The module preceding structs, functions, or variables.
#[derive(Default, Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum ModulePath {
    #[default]
    /// This is a local type, not imported from another module.
    Local,

    /// This is a type imported from another module.
    Alias(Ident),

    /// This is a type imported from another module,
    /// fully-qualified (as `user::repo`) thanks to the name resolution pass of the compiler.
    Absolute(UserRepo),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TyKind {
    /// The main primitive type. 'Nuf said.
    // TODO: Field { constant: bool },
    Field,

    /// Custom / user-defined types
    Custom { module: ModulePath, name: String },

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

impl TyKind {
    pub fn match_expected(&self, expected: &TyKind) -> bool {
        match (self, expected) {
            (TyKind::BigInt, TyKind::Field) => true,
            (TyKind::Array(lhs, lhs_size), TyKind::Array(rhs, rhs_size)) => {
                lhs_size == rhs_size && lhs.match_expected(rhs)
            }
            (
                TyKind::Custom { module, name },
                TyKind::Custom {
                    module: expected_module,
                    name: expected_name,
                },
            ) => module == expected_module && name == expected_name,
            (x, y) if x == y => true,
            _ => false,
        }
    }

    pub fn same_as(&self, other: &TyKind) -> bool {
        match (self, other) {
            (TyKind::BigInt, TyKind::Field) | (TyKind::Field, TyKind::BigInt) => true,
            (TyKind::Array(lhs, lhs_size), TyKind::Array(rhs, rhs_size)) => {
                lhs_size == rhs_size && lhs.match_expected(rhs)
            }
            (
                TyKind::Custom { module, name },
                TyKind::Custom {
                    module: expected_module,
                    name: expected_name,
                },
            ) => module == expected_module && name == expected_name,
            (x, y) if x == y => true,
            _ => false,
        }
    }
}

impl Display for TyKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TyKind::Custom { module, name } => match module {
                ModulePath::Absolute(user_repo) => write!(
                    f,
                    "a `{module}::{submodule}::{name}` struct",
                    name = name,
                    module = user_repo.user,
                    submodule = user_repo.repo
                ),
                ModulePath::Alias(module) => write!(
                    f,
                    "a `{module}::{name}` struct",
                    name = name,
                    module = module.value
                ),
                ModulePath::Local => write!(f, "a `{}` struct", name),
            },
            TyKind::Field => write!(f, "Field"),
            TyKind::BigInt => write!(f, "BigInt"),
            TyKind::Array(ty, size) => write!(f, "[{}; {}]", ty, size),
            TyKind::Bool => write!(f, "Bool"),
        }
    }
}

impl Ty {
    pub fn reserved_types(module: ModulePath, name: Ident) -> TyKind {
        match name.value.as_ref() {
            "Field" | "Bool" if !matches!(module, ModulePath::Local) => {
                panic!("reserved types cannot be in a module (TODO: better error)")
            }
            "Field" => TyKind::Field,
            "Bool" => TyKind::Bool,
            _ => TyKind::Custom {
                module,
                name: name.value,
            },
        }
    }

    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self> {
        let token = tokens.bump_err(ctx, ErrorKind::MissingType)?;

        match token.kind {
            // module::Type or Type
            // ^^^^^^^^^^^^    ^^^^
            TokenKind::Identifier(ty_name) => {
                let maybe_module = Ident::new(ty_name.clone(), token.span);
                let (module, name, _span) = if is_type(&ty_name) {
                    // Type
                    // ^^^^
                    (ModulePath::Local, maybe_module, token.span)
                } else {
                    // module::Type
                    //       ^^
                    tokens.bump_expected(ctx, TokenKind::DoubleColon)?;

                    // module::Type
                    //         ^^^^
                    let (name, span) = match tokens.bump(ctx) {
                        Some(Token {
                            kind: TokenKind::Identifier(name),
                            span,
                        }) => (name, span),
                        _ => return Err(ctx.error(ErrorKind::MissingType, ctx.last_span())),
                    };

                    let name = Ident::new(name, span);
                    let span = token.span.merge_with(span);

                    (ModulePath::Alias(maybe_module), name, span)
                };

                let ty_kind = Self::reserved_types(module, name);

                Ok(Self {
                    kind: ty_kind,
                    span: token.span,
                })
            }

            // array
            // [type; size]
            // ^
            TokenKind::LeftBracket => {
                let span = token.span;

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
                    TokenKind::BigInt(s) => s
                        .parse()
                        .map_err(|_e| ctx.error(ErrorKind::InvalidArraySize, siz.span))?,
                    _ => {
                        return Err(ctx.error(
                            ErrorKind::ExpectedToken(TokenKind::BigInt("".to_string())),
                            siz.span,
                        ));
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
            _ => Err(ctx.error(ErrorKind::InvalidType, token.span)),
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

impl FnSig {
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self> {
        let (name, kind) = FuncOrMethod::parse(ctx, tokens)?;

        let arguments = FunctionDef::parse_args(ctx, tokens, &kind)?;

        let return_type = FunctionDef::parse_fn_return_type(ctx, tokens)?;

        Ok(Self {
            kind,
            name,
            arguments,
            return_type,
        })
    }
}

/// Any kind of text that can represent a type, a variable, a function name, etc.
#[derive(Debug, Default, Clone, Eq, Serialize, Deserialize, Educe)]
#[educe(Hash)]
pub struct Ident {
    pub value: String,
    #[educe(Hash(ignore))]
    pub span: Span,
}

impl PartialEq for Ident {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Ident {
    pub fn new(value: String, span: Span) -> Self {
        Self { value, span }
    }

    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self> {
        let token = tokens.bump_err(ctx, ErrorKind::MissingToken)?;
        match token.kind {
            TokenKind::Identifier(ident) => Ok(Self {
                value: ident,
                span: token.span,
            }),

            _ => Err(ctx.error(
                ErrorKind::ExpectedToken(TokenKind::Identifier("".to_string())),
                token.span,
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AttributeKind {
    Pub,
    Const,
}

impl AttributeKind {
    pub fn is_public(&self) -> bool {
        matches!(self, Self::Pub)
    }

    pub fn is_constant(&self) -> bool {
        matches!(self, Self::Const)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attribute {
    pub kind: AttributeKind,
    pub span: Span,
}

impl Attribute {
    pub fn is_public(&self) -> bool {
        self.kind.is_public()
    }

    pub fn is_constant(&self) -> bool {
        self.kind.is_constant()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub sig: FnSig,
    pub body: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FuncOrMethod {
    /// Function.
    Function(
        /// Set during name resolution.
        ModulePath,
    ),
    /// Method defined on a custom type.
    Method(CustomType),
}

impl Default for FuncOrMethod {
    fn default() -> Self {
        unreachable!()
    }
}

// TODO: remove default here?
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FnSig {
    pub kind: FuncOrMethod,
    pub name: Ident,
    /// (pub, ident, type)
    pub arguments: Vec<FnArg>,
    pub return_type: Option<Ty>,
}

pub struct Method {
    pub sig: MethodSig,
    pub body: Vec<Stmt>,
    pub span: Span,
}

pub struct MethodSig {
    pub self_name: CustomType,
    pub name: Ident,
    /// (pub, ident, type)
    pub arguments: Vec<FnArg>,
    pub return_type: Option<Ty>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FnArg {
    pub name: Ident,
    pub typ: Ty,
    pub attribute: Option<Attribute>,
    pub span: Span,
}

impl FnArg {
    pub fn is_public(&self) -> bool {
        self.attribute
            .as_ref()
            .map(|attr| attr.is_public())
            .unwrap_or(false)
    }

    pub fn is_constant(&self) -> bool {
        self.attribute
            .as_ref()
            .map(|attr| attr.is_constant())
            .unwrap_or(false)
    }
}

impl FuncOrMethod {
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<(Ident, Self)> {
        // fn House.verify(   or   fn verify(
        //    ^^^^^                   ^^^^^
        let maybe_self_name = tokens.bump_ident(
            ctx,
            ErrorKind::InvalidFunctionSignature("expected function name"),
        )?;

        // fn House.verify(
        //    ^^^^^
        if is_type(&maybe_self_name.value) {
            let struct_name = maybe_self_name;
            // fn House.verify(
            //         ^
            tokens.bump_expected(ctx, TokenKind::Dot)?;

            // fn House.verify(
            //          ^^^^^^
            let name = tokens.bump_ident(
                ctx,
                ErrorKind::InvalidFunctionSignature("expected function name"),
            )?;

            Ok((
                name,
                FuncOrMethod::Method(CustomType {
                    module: ModulePath::Local,
                    name: struct_name.value,
                    span: struct_name.span,
                }),
            ))
        } else {
            // fn verify(
            //    ^^^^^^

            // check that it is not shadowing a builtin
            let fn_name = maybe_self_name;

            Ok((fn_name, FuncOrMethod::Function(ModulePath::Local)))
        }
    }
}

impl FunctionDef {
    pub fn is_main(&self) -> bool {
        self.sig.name.value == "main"
    }

    pub fn parse_args(
        ctx: &mut ParserCtx,
        tokens: &mut Tokens,
        fn_kind: &FuncOrMethod,
    ) -> Result<Vec<FnArg>> {
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
                ErrorKind::InvalidFunctionSignature("expected function arguments"),
            )?;

            let (attribute, arg_name) = match token.kind {
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
                // constant input
                TokenKind::Keyword(Keyword::Const) => {
                    let arg_name = Ident::parse(ctx, tokens)?;
                    (
                        Some(Attribute {
                            kind: AttributeKind::Const,
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
                    return Err(ctx.error(
                        ErrorKind::InvalidFunctionSignature("expected identifier"),
                        token.span,
                    ));
                }
            };

            // self takes no value
            let arg_typ = if arg_name.value == "self" {
                let self_name = match fn_kind {
                    FuncOrMethod::Function(_) => {
                        return Err(ctx.error(
                            ErrorKind::InvalidFunctionSignature(
                                "the `self` argument is only allowed in methods, not functions",
                            ),
                            arg_name.span,
                        ));
                    }
                    FuncOrMethod::Method(self_name) => self_name,
                };

                if !args.is_empty() {
                    return Err(ctx.error(
                        ErrorKind::InvalidFunctionSignature("`self` must be the first argument"),
                        arg_name.span,
                    ));
                }

                Ty {
                    kind: TyKind::Custom {
                        module: ModulePath::Local,
                        name: self_name.name.clone(),
                    },
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

            let span = if let Some(attr) = &attribute {
                if &arg_name.value == "self" {
                    return Err(ctx.error(ErrorKind::SelfHasAttribute, arg_name.span));
                } else {
                    attr.span.merge_with(arg_typ.span)
                }
            } else {
                if &arg_name.value == "self" {
                    arg_name.span
                } else {
                    arg_name.span.merge_with(arg_typ.span)
                }
            };

            let arg = FnArg {
                name: arg_name,
                typ: arg_typ,
                attribute,
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
                    return Err(ctx.error(
                        ErrorKind::InvalidFunctionSignature(
                            "expected end of function or other argument",
                        ),
                        separator.span,
                    ));
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
            .ok_or_else(|| {
                ctx.error(
                    ErrorKind::InvalidFunctionSignature("expected function name"),
                    ctx.last_span(),
                )
            })?
            .span;

        // parse signature
        let sig = FnSig::parse(ctx, tokens)?;

        // make sure that it doesn't shadow a builtin
        if BUILTIN_FN_NAMES.contains(&sig.name.value) {
            return Err(ctx.error(
                ErrorKind::ShadowingBuiltIn(sig.name.value.clone()),
                sig.name.span,
            ));
        }

        // parse body
        let body = Self::parse_fn_body(ctx, tokens)?;

        // here's the last token, that is if the function is not empty (maybe we should disallow empty functions?)

        if let Some(t) = body.last() {
            span = span.merge_with(t.span);
        } else {
            return Err(ctx.error(
                ErrorKind::InvalidFunctionSignature("expected function body"),
                ctx.last_span(),
            ));
        }

        let func = Self { sig, body, span };

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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StmtKind {
    Assign {
        mutable: bool,
        lhs: Ident,
        rhs: Box<Expr>,
    },
    Expr(Box<Expr>),
    Return(Box<Expr>),
    Comment(String),

    // `for var in 0..10 { <body> }`
    ForLoop {
        var: Ident,
        range: Range,
        body: Vec<Stmt>,
    },
}

impl Stmt {
    /// Returns a list of statement parsed until seeing the end of a block (`}`).
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self> {
        match tokens.peek() {
            None => Err(ctx.error(ErrorKind::InvalidStatement, ctx.last_span())),
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
                span = span.merge_with(rhs.span);

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
                        let start: u32 = n
                            .parse()
                            .map_err(|_e| ctx.error(ErrorKind::InvalidRangeSize, span))?;
                        (start, span)
                    }
                    _ => {
                        return Err(ctx.error(
                            ErrorKind::ExpectedToken(TokenKind::BigInt("".to_string())),
                            ctx.last_span(),
                        ))
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
                        let end: u32 = n
                            .parse()
                            .map_err(|_e| ctx.error(ErrorKind::InvalidRangeSize, span))?;
                        (end, span)
                    }
                    _ => {
                        return Err(ctx.error(
                            ErrorKind::ExpectedToken(TokenKind::BigInt("".to_string())),
                            ctx.last_span(),
                        ))
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
                    kind: StmtKind::ForLoop { var, range, body },
                    span,
                })
            }

            // if/else
            Some(Token {
                kind: TokenKind::Keyword(Keyword::If),
                span: _,
            }) => {
                // TODO: wait, this should be implemented as an expression! not a statement
                panic!("if statements are not implemented yet. Use if expressions instead (e.g. `x = if cond {{ 1 }} else {{ 2 }};`)");
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
pub struct Root<F>
where
    F: Field,
{
    pub kind: RootKind<F>,
    pub span: Span,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UsePath {
    pub module: Ident,
    pub submodule: Ident,
    pub span: Span,
}

impl From<&UsePath> for UserRepo {
    fn from(path: &UsePath) -> Self {
        UserRepo {
            user: path.module.value.clone(),
            repo: path.submodule.value.clone(),
        }
    }
}

impl Display for UsePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}::{}", self.module.value, self.submodule.value)
    }
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
pub enum RootKind<F: Field> {
    Use(UsePath),
    FunctionDef(FunctionDef),
    Comment(String),
    StructDef(StructDef),
    ConstDef(ConstDef<F>),
}

//
// Const
//

#[derive(Debug)]
pub struct ConstDef<F>
where
    F: Field,
{
    pub module: ModulePath, // name resolution
    pub name: Ident,
    pub value: F,
    pub span: Span,
}

impl<F: Field + FromStr> ConstDef<F> {
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
            ExprKind::BigInt(s) => s
                .parse()
                .map_err(|_e| ctx.error(ErrorKind::InvalidField(s.clone()), value.span))?,
            _ => {
                return Err(ctx.error(ErrorKind::InvalidConstType, value.span));
            }
        };

        // const foo = 42;
        //               ^
        tokens.bump_expected(ctx, TokenKind::SemiColon)?;

        //
        let span = name.span;
        Ok(ConstDef {
            module: ModulePath::Local,
            name,
            value,
            span,
        })
    }
}
