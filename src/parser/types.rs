use educe::Educe;
use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    hash::{Hash, Hasher},
    str::FromStr,
};

use ark_ff::{Field, Zero};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};

use crate::{
    cli::packages::UserRepo,
    constants::Span,
    error::{Error, ErrorKind, Result},
    lexer::{Keyword, Token, TokenKind, Tokens},
    mast::ExprMonoInfo,
    stdlib::builtins::BUILTIN_FN_NAMES,
    syntax::{is_generic_parameter, is_type},
};

use super::{CustomType, Expr, ExprKind, Op2, ParserCtx, StructDef};

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

pub fn is_numeric(typ: &TyKind) -> bool {
    matches!(typ, TyKind::Field { .. })
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
pub enum Symbolic {
    /// A literal number
    Concrete(u32),
    /// Point to a constant variable
    Constant(Ident),
    /// Generic parameter
    Generic(Ident),
    Add(Box<Symbolic>, Box<Symbolic>),
    Mul(Box<Symbolic>, Box<Symbolic>),
}

impl Display for Symbolic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Symbolic::Concrete(n) => write!(f, "{}", n),
            Symbolic::Constant(ident) => write!(f, "{}", ident.value),
            Symbolic::Generic(ident) => write!(f, "{}", ident.value),
            Symbolic::Add(lhs, rhs) => write!(f, "{} + {}", lhs, rhs),
            Symbolic::Mul(lhs, rhs) => write!(f, "{} * {}", lhs, rhs),
        }
    }
}

impl Symbolic {
    /// Extract all generic parameters.
    /// Since the function signature syntax doesn't support <N, M> to declare generics,
    /// we need to extract the implicit generic parameters from the function arguments.
    /// Then they will be attached to [FnSig]
    pub fn extract_generics(&self) -> HashSet<String> {
        let mut generics = HashSet::new();

        match self {
            Symbolic::Concrete(_) => (),
            Symbolic::Constant(ident) => {
                generics.insert(ident.value.clone());
            }
            Symbolic::Generic(ident) => {
                generics.insert(ident.value.clone());
            }
            Symbolic::Add(lhs, rhs) | Symbolic::Mul(lhs, rhs) => {
                generics.extend(lhs.extract_generics());
                generics.extend(rhs.extract_generics());
            }
        }

        generics
    }

    /// Parse from an expression node recursively.
    pub fn parse(node: &Expr) -> Result<Self> {
        match &node.kind {
            ExprKind::BigUInt(n) => Ok(Symbolic::Concrete(n.to_u32().unwrap())),
            ExprKind::Variable { module: _, name } => {
                if is_generic_parameter(&name.value) {
                    Ok(Symbolic::Generic(name.clone()))
                } else {
                    Ok(Symbolic::Constant(name.clone()))
                }
            }
            ExprKind::BinaryOp {
                op,
                lhs,
                rhs,
                protected: _,
            } => {
                let lhs = Symbolic::parse(lhs)?;
                let rhs = Symbolic::parse(rhs);

                match op {
                    Op2::Addition => Ok(Symbolic::Add(Box::new(lhs), Box::new(rhs?))),
                    Op2::Multiplication => Ok(Symbolic::Mul(Box::new(lhs), Box::new(rhs?))),
                    _ => Err(Error::new(
                        "mast",
                        ErrorKind::InvalidSymbolicSize,
                        node.span,
                    )),
                }
            }
            _ => Err(Error::new(
                "mast",
                ErrorKind::InvalidSymbolicSize,
                node.span,
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TyKind {
    /// The main primitive type. 'Nuf said.
    Field { constant: bool },

    /// Custom / user-defined types
    Custom { module: ModulePath, name: String },

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
    /// An array with symbolic size.
    /// This is an intermediate type.
    /// After monomorphized, it will be converted to `Array`.
    GenericSizedArray(Box<TyKind>, Symbolic),
}

impl TyKind {
    /// A less strict checks when comparing with generic types.
    pub fn match_expected(&self, expected: &TyKind) -> bool {
        match (self, expected) {
            (TyKind::Field { .. }, TyKind::Field { .. }) => true,
            (TyKind::Array(lhs, lhs_size), TyKind::Array(rhs, rhs_size)) => {
                lhs_size == rhs_size && lhs.match_expected(rhs)
            }
            // the checks on the generic arrays can be done in MAST
            (TyKind::GenericSizedArray(lhs, _), TyKind::GenericSizedArray(rhs, _))
            | (TyKind::Array(lhs, _), TyKind::GenericSizedArray(rhs, _))
            | (TyKind::GenericSizedArray(lhs, _), TyKind::Array(rhs, _)) => lhs.match_expected(rhs),
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

    /// An exact match check, assuming there is no generic type.
    pub fn same_as(&self, other: &TyKind) -> bool {
        match (self, other) {
            (TyKind::Field { .. }, TyKind::Field { .. }) => true,
            (TyKind::Array(lhs, lhs_size), TyKind::Array(rhs, rhs_size)) => {
                lhs_size == rhs_size && lhs.same_as(rhs)
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

    /// Recursively extract generic parameters from GenericArray type
    /// it should be able to extract generic parameter 'N' 'M' from [[Field; N], M]
    pub fn extract_generics(&self) -> HashSet<String> {
        let mut generics = HashSet::new();

        match self {
            TyKind::Field { .. } => (),
            TyKind::Bool => (),
            TyKind::Custom { .. } => (),
            // e.g [[Field; N], 3]
            TyKind::Array(ty, _) => {
                generics.extend(ty.extract_generics());
            }
            // e.g [[Field; N], M]
            TyKind::GenericSizedArray(ty, sym) => {
                generics.extend(ty.extract_generics());
                generics.extend(sym.extract_generics());
            }
        }

        generics
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
            TyKind::Field { constant } => {
                write!(
                    f,
                    "{}",
                    if *constant {
                        "a constant field element"
                    } else {
                        "a field element"
                    }
                )
            }
            TyKind::Array(ty, size) => write!(f, "[{}; {}]", ty, size),
            TyKind::Bool => write!(f, "Bool"),
            TyKind::GenericSizedArray(ty, size) => write!(f, "[{}; {}]", ty, size),
        }
    }
}

impl Ty {
    pub fn reserved_types(module: ModulePath, name: Ident) -> TyKind {
        match name.value.as_ref() {
            "Field" | "Bool" if !matches!(module, ModulePath::Local) => {
                panic!("reserved types cannot be in a module (TODO: better error)")
            }
            // Default the `constant` to false, as here has no context for const attribute.
            // For a function argument and it is with const attribute,
            // the `constant` will be corrected to true by the `FunctionDef::parse_args` parser.
            "Field" => TyKind::Field { constant: false },
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
                let siz_first = tokens.bump_err(ctx, ErrorKind::InvalidToken)?;

                // [type; size]
                //            ^
                let siz_second = tokens.bump_err(ctx, ErrorKind::InvalidToken)?;

                // return Array(ty, siz) if size is a number and right_paren is ]
                match (&siz_first.kind, &siz_second.kind) {
                    (TokenKind::BigUInt(b), TokenKind::RightBracket) => {
                        let siz: u32 = b
                            .try_into()
                            .map_err(|_e| ctx.error(ErrorKind::InvalidArraySize, siz_first.span))?;
                        let span = span.merge_with(siz_second.span);

                        Ok(Ty {
                            kind: TyKind::Array(Box::new(ty.kind), siz),
                            span,
                        })
                    }
                    // [Field; nn]
                    // [Field; NN]
                    //         ^^^
                    (TokenKind::Identifier(name), TokenKind::RightBracket) => {
                        let siz = Ident::new(name.to_string(), siz_first.span);
                        let span = span.merge_with(siz_second.span);
                        let sym = if is_generic_parameter(name) {
                            Symbolic::Generic(siz)
                        } else {
                            Symbolic::Constant(siz)
                        };

                        Ok(Ty {
                            kind: TyKind::GenericSizedArray(Box::new(ty.kind), sym),
                            span,
                        })
                    }
                    _ => Err(ctx.error(ErrorKind::InvalidSymbolicSize, siz_first.span)),
                }
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
//~ param ::= { "pub" | "const" } ident ":" type
//~

impl FnSig {
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self> {
        let (name, kind) = FuncOrMethod::parse(ctx, tokens)?;

        let arguments = FunctionDef::parse_args(ctx, tokens, &kind)?;

        // extract generic parameters from arguments
        let mut generics = GenericParameters::default();
        for arg in &arguments {
            match &arg.typ.kind {
                TyKind::Field { .. } => {
                    // extract from const argument
                    if is_generic_parameter(&arg.name.value) && arg.is_constant() {
                        generics.add(arg.name.value.to_string());
                    }
                }
                TyKind::Array(ty, _) => {
                    // recursively extract all generic parameters from the item type
                    let extracted = ty.extract_generics();

                    for name in extracted {
                        generics.add(name);
                    }
                }
                TyKind::GenericSizedArray(_, _) => {
                    // recursively extract all generic parameters from the symbolic size
                    let extracted = arg.typ.kind.extract_generics();

                    for name in extracted {
                        generics.add(name);
                    }
                }
                _ => (),
            }
        }

        let return_type = FunctionDef::parse_fn_return_type(ctx, tokens)?;

        Ok(Self {
            kind,
            name,
            generics,
            arguments,
            return_type,
        })
    }

    /// Recursively assign values to the generic parameters based on observed Array type argument
    fn resolve_generic_array(
        &mut self,
        sig_arg: &TyKind,
        observed: &TyKind,
        span: Span,
    ) -> Result<()> {
        match (sig_arg, observed) {
            // [[Field; NN]; MM]
            (TyKind::GenericSizedArray(ty, sym), TyKind::Array(observed_ty, observed_size)) => {
                // resolve the generic parameter
                match sym {
                    Symbolic::Generic(ident) => {
                        self.generics.assign(&ident.value, *observed_size, span)?;
                    }
                    _ => unreachable!("no operation allowed on symbolic size in function argument"),
                }

                // recursively resolve the generic parameter
                self.resolve_generic_array(ty, observed_ty, span)?;
            }
            // [[Field; NN]; 3]
            (TyKind::Array(ty, _), TyKind::Array(observed_ty, _)) => {
                // recursively resolve the generic parameter
                self.resolve_generic_array(ty, observed_ty, span)?;
            }
            _ => (),
        }

        Ok(())
    }

    /// Resolve generic values for each generic parameter
    pub fn resolve_generic_values(&mut self, observed: &[ExprMonoInfo]) -> Result<()> {
        for (sig_arg, observed_arg) in self.arguments.clone().iter().zip(observed) {
            let observed_ty = observed_arg.typ.clone().expect("expected type");
            match (&sig_arg.typ.kind, &observed_ty) {
                (TyKind::GenericSizedArray(_, _), TyKind::Array(_, _))
                | (TyKind::Array(_, _), TyKind::Array(_, _)) => {
                    self.resolve_generic_array(
                        &sig_arg.typ.kind,
                        &observed_ty,
                        observed_arg.expr.span,
                    )?;
                }
                // const NN: Field
                _ => {
                    let cst = observed_arg.constant;
                    if is_generic_parameter(sig_arg.name.value.as_str()) && cst.is_some() {
                        self.generics.assign(
                            &sig_arg.name.value,
                            cst.unwrap(),
                            observed_arg.expr.span,
                        )?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Returns true if the function signature contains generic parameters or generic array types.
    /// Either:
    /// - `const NN: Field` or `[[Field; NN]; MM]`
    /// - `[Field; cst]`, where cst is a constant variable. We also monomorphize generic array with a constant var as its size.
    pub fn require_monomorphization(&self) -> bool {
        let has_arg_cst = self
            .arguments
            .iter()
            .any(|arg| self.has_constant(&arg.typ.kind));

        let has_ret_cst = self.return_type.is_some()
            && self.has_constant(&self.return_type.as_ref().unwrap().kind);

        !self.generics.is_empty() || has_arg_cst || has_ret_cst
    }

    /// Recursively check if the generic array symbolic value contains constant variant
    fn has_constant(&self, typ: &TyKind) -> bool {
        match typ {
            TyKind::GenericSizedArray(ty, sym) => {
                match sym {
                    Symbolic::Constant(_) => return true,
                    _ => false,
                };

                self.has_constant(ty)
            }
            TyKind::Array(ty, _) => self.has_constant(ty),
            _ => false,
        }
    }

    /// Returns the monomorphized function name,
    /// using the patter: `fn_full_qualified_name#generic1=value1#generic2=value2`
    pub fn monomorphized_name(&self) -> Ident {
        let mut name = self.name.clone();

        if self.require_monomorphization() {
            let mut generics = self.generics.0.iter().collect::<Vec<_>>();
            generics.sort_by(|a, b| a.0.cmp(b.0));

            let generics = generics
                .iter()
                .map(|(name, value)| format!("{}={}", name, value.unwrap()))
                .collect::<Vec<_>>()
                .join("#");

            name.value.push_str(&format!("#{}", generics));
        }

        name
    }
}

/// Any kind of text that can represent a type, a variable, a function name, etc.
#[derive(Debug, Default, Clone, Eq, Serialize, Deserialize, Educe)]
#[educe(Hash, PartialEq)]
pub struct Ident {
    pub value: String,
    #[educe(Hash(ignore))]
    #[educe(PartialEq(ignore))]
    pub span: Span,
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

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct GenericParameters(HashMap<String, Option<u32>>);

impl GenericParameters {
    /// Return all generic parameter names
    pub fn names(&self) -> HashSet<String> {
        self.0.keys().cloned().collect()
    }

    /// Add an unbound generic parameter
    pub fn add(&mut self, name: String) {
        self.0.insert(name, None);
    }

    /// Get the value of a generic parameter
    pub fn get(&self, name: &str) -> u32 {
        self.0
            .get(name)
            .expect("generic parameter not found")
            .expect("generic value not assigned")
    }

    /// Returns whether the generic parameters are empty
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Bind a generic parameter to a value
    pub fn assign(&mut self, name: &String, value: u32, span: Span) -> Result<()> {
        let existing = self.0.get(name);
        match existing {
            Some(Some(v)) => {
                if *v == value {
                    return Ok(());
                }

                Err(Error::new(
                    "mast",
                    ErrorKind::ConflictGenericValue(name.to_string(), *v, value),
                    span,
                ))
            }
            Some(None) => {
                self.0.insert(name.to_string(), Some(value));
                Ok(())
            }
            None => Err(Error::new(
                "mast",
                ErrorKind::UnexpectedGenericParameter(name.to_string()),
                span,
            )),
        }
    }
}

// TODO: remove default here?
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FnSig {
    pub kind: FuncOrMethod,
    pub name: Ident,
    pub generics: GenericParameters,
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

            let mut arg = FnArg {
                name: arg_name,
                typ: arg_typ,
                attribute,
                span,
            };

            // if it is with const attribute, then converts it to a constant field.
            // this is because the parser doesn't know if a token has a corresponding attribute
            // until it has parsed the whole token.
            if arg.is_constant() {
                arg.typ.kind = TyKind::Field { constant: true };
            }

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
        if BUILTIN_FN_NAMES.contains(&sig.name.value.as_ref()) {
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
    pub start: Expr,
    pub end: Expr,
    pub span: Span,
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
                let start = Expr::parse(ctx, tokens)?;

                // for i in 0..5 { ... }
                //           ^^
                tokens.bump_expected(ctx, TokenKind::DoubleDot)?;

                // for i in 0..5 { ... }
                //             ^
                let end = Expr::parse(ctx, tokens)?;

                let start_span = start.span;
                let end_span = end.span;

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
                let end_token = tokens.bump_expected(ctx, TokenKind::SemiColon)?;

                Ok(Stmt {
                    kind: StmtKind::Return(Box::new(expr)),
                    span: span.merge_with(end_token.span),
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
            ExprKind::BigUInt(n) => n
                .to_string()
                .parse()
                .map_err(|_e| ctx.error(ErrorKind::InvalidField(n.to_string()), value.span))?,
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
