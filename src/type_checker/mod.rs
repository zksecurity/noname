use std::collections::HashMap;

use crate::{
    backends::Backend,
    cli::packages::UserRepo,
    constants::Span,
    error::{Error, ErrorKind, Result},
    imports::FnKind,
    name_resolution::NAST,
    parser::{
        types::{FuncOrMethod, FunctionDef, ModulePath, RootKind, Ty, TyKind},
        CustomType, Expr, StructDef,
    },
    stdlib::{builtin_fns, crypto::crypto_fns, QUALIFIED_BUILTINS},
};

use ark_ff::Field;
pub use checker::{FnInfo, StructInfo};
pub use fn_env::{TypeInfo, TypedFnEnv};

use serde::{Deserialize, Serialize};
use serde_with::serde_as;

pub mod checker;
pub mod fn_env;

const RESERVED_ARGS: [&str; 1] = ["public_output"];

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstInfo<F>
where
    F: Field,
{
    #[serde_as(as = "crate::serialization::SerdeAs")]
    pub value: Vec<F>,
    pub typ: Ty,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FullyQualified {
    /// Set to `None` if the function is defined in the main module.
    pub module: Option<UserRepo>,
    pub name: String,
}

impl FullyQualified {
    #[must_use]
    pub fn local(name: String) -> Self {
        Self { module: None, name }
    }

    #[must_use]
    // TODO: Pass in `String`, instead of `&str`, so we don't hide an
    // allocation within.
    pub fn new(module: &ModulePath, name: &str) -> Self {
        let module = match module {
            ModulePath::Local => None,
            ModulePath::Alias(_) => unreachable!(),
            ModulePath::Absolute(user_repo) => Some(user_repo.clone()),
        };
        Self {
            module,
            name: name.to_string(),
        }
    }
}

/// The environment we use to type check a noname program.
#[derive(Debug, Serialize, Deserialize)]
pub struct TypeChecker<B>
where
    B: Backend,
{
    /// the functions present in the scope
    /// contains at least the set of builtin functions (like `assert_eq`)
    functions: HashMap<FullyQualified, FnInfo<B>>,

    /// Custom structs type information and ASTs for methods.
    structs: HashMap<FullyQualified, StructInfo>,

    /// Constants declared in this module.
    constants: HashMap<FullyQualified, ConstInfo<B::Field>>,

    /// Mapping from node id to `TyKind`.
    /// This can be used by the circuit-writer when it needs type information.
    // TODO: I think we should get rid of this if we can
    node_types: HashMap<usize, TyKind>,
}

impl<B: Backend> TypeChecker<B> {
    pub(crate) fn expr_type(&self, expr: &Expr) -> Option<&TyKind> {
        self.node_types.get(&expr.node_id)
    }

    // TODO: can we get rid of this?
    pub(crate) fn node_type(&self, node_id: usize) -> Option<&TyKind> {
        self.node_types.get(&node_id)
    }

    pub(crate) fn struct_info(&self, qualified: &FullyQualified) -> Option<&StructInfo> {
        self.structs.get(qualified)
    }

    pub(crate) fn fn_info(&self, qualified: &FullyQualified) -> Option<&FnInfo<B>> {
        self.functions.get(qualified)
    }

    pub(crate) fn const_info(&self, qualified: &FullyQualified) -> Option<&ConstInfo<B::Field>> {
        self.constants.get(qualified)
    }

    /// Returns the number of field elements contained in the given type.
    // TODO: might want to memoize that at some point
    pub(crate) fn size_of(&self, typ: &TyKind) -> usize {
        match typ {
            TyKind::Field => 1,
            TyKind::Custom { module, name } => {
                let qualified = FullyQualified::new(module, name);
                let struct_info = self
                    .struct_info(&qualified)
                    .expect("bug in the type checker: cannot find struct info");

                let mut sum = 0;

                for (_, t) in &struct_info.fields {
                    sum += self.size_of(t);
                }

                sum
            }
            TyKind::BigInt => 1,
            TyKind::Array(typ, len) => (*len as usize) * self.size_of(typ),
            TyKind::Bool => 1,
        }
    }
}

impl<B: Backend> Default for TypeChecker<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> TypeChecker<B> {
    // TODO: we can probably lazy const this
    #[must_use]
    pub fn new() -> Self {
        let mut type_checker = Self {
            functions: HashMap::new(),
            structs: HashMap::new(),
            constants: HashMap::new(),
            node_types: HashMap::new(),
        };

        // initialize it with the builtins
        let builtin_module = ModulePath::Absolute(UserRepo::new(QUALIFIED_BUILTINS));
        for fn_info in builtin_fns() {
            let qualified = FullyQualified::new(&builtin_module, &fn_info.sig().name.value);
            if type_checker
                .functions
                .insert(qualified, fn_info.clone())
                .is_some()
            {
                panic!("type-checker bug: global imports conflict");
            }
        }

        // initialize it with the standard library
        let crypto_module = ModulePath::Absolute(UserRepo::new("std/crypto"));
        for fn_info in crypto_fns() {
            let qualified = FullyQualified::new(&crypto_module, &fn_info.sig().name.value);
            if type_checker
                .functions
                .insert(qualified, fn_info.clone())
                .is_some()
            {
                panic!("type-checker bug: global imports conflict");
            }
        }

        //
        type_checker
    }

    #[must_use]
    pub fn error(&self, kind: ErrorKind, span: Span) -> Error {
        Error::new("type-checker", kind, span)
    }

    /// This takes the AST produced by the parser, and performs two things:
    /// - resolves imports
    /// - type checks
    pub fn analyze(&mut self, nast: NAST<B>, is_lib: bool) -> Result<()> {
        //
        // Process constants
        //

        // we detect struct or function definition
        let mut abort = None;

        for root in &nast.ast.0 {
            match &root.kind {
                RootKind::ConstDef(cst) => {
                    // important: no struct or function definition must appear before a constant declaration
                    if let Some(span) = abort {
                        return Err(Error::new(
                            "type-checker",
                            ErrorKind::OrderOfConstDeclaration,
                            span,
                        ));
                    }

                    let qualified = FullyQualified::new(&cst.module, &cst.name.value);

                    if self
                        .constants
                        .insert(
                            qualified,
                            ConstInfo {
                                value: vec![cst.value],
                                typ: Ty {
                                    kind: TyKind::Field,
                                    span: cst.span,
                                },
                            },
                        )
                        .is_some()
                    {
                        return Err(Error::new(
                            "type-checker",
                            ErrorKind::DuplicateDefinition(cst.name.value.clone()),
                            cst.name.span,
                        ));
                    }
                }

                RootKind::FunctionDef(FunctionDef { span, .. })
                | RootKind::StructDef(StructDef { span, .. }) => abort = Some(*span),

                RootKind::Use(_) | RootKind::Comment(_) => (),
            }
        }

        //
        // Type check structs
        //

        for root in &nast.ast.0 {
            match &root.kind {
                // `use user::repo;`
                RootKind::StructDef(struct_def) => {
                    let StructDef {
                        module,
                        name,
                        fields,
                        ..
                    } = struct_def;

                    let fields: Vec<_> = fields
                        .iter()
                        .map(|field| {
                            let (name, typ) = field;
                            (name.value.clone(), typ.kind.clone())
                        })
                        .collect();

                    let struct_info = StructInfo {
                        name: name.name.clone(),
                        fields,
                        methods: HashMap::new(),
                    };

                    let qualified = FullyQualified::new(module, &name.name);
                    self.structs.insert(qualified, struct_info);
                }

                RootKind::ConstDef(_)
                | RootKind::Use(_)
                | RootKind::FunctionDef(_)
                | RootKind::Comment(_) => (),
            }
        }

        //
        // Type check functions and methods
        //

        for root in &nast.ast.0 {
            match &root.kind {
                // `fn main() { ... }`
                RootKind::FunctionDef(function) => {
                    // create a new typed fn environment to type check the function
                    let mut typed_fn_env = TypedFnEnv::default();

                    // if we're expecting a library, this should not be the main function
                    let is_main = function.is_main();
                    if is_main && is_lib {
                        return Err(Error::new(
                            "type-checker",
                            ErrorKind::MainFunctionInLib,
                            function.span,
                        ));
                    }

                    // if this is the main function check that it has arguments
                    if is_main && function.sig.arguments.is_empty() {
                        return Err(Error::new(
                            "type-checker",
                            ErrorKind::NoArgsInMain,
                            function.span,
                        ));
                    }

                    // save the function in the typed global env
                    let fn_kind = FnKind::<B>::Native(function.clone());
                    let fn_info = FnInfo {
                        kind: fn_kind,
                        span: function.span,
                    };

                    match &function.sig.kind {
                        FuncOrMethod::Method(custom) => {
                            let CustomType {
                                module,
                                name,
                                span: _,
                            } = custom;
                            let qualified = FullyQualified::new(module, name);
                            let struct_info = self
                                .structs
                                .get_mut(&qualified)
                                .expect("couldn't find the struct for storing the method");

                            struct_info
                                .methods
                                .insert(function.sig.name.value.clone(), function.clone());
                        }
                        FuncOrMethod::Function(module) => {
                            let qualified = FullyQualified::new(module, &function.sig.name.value);
                            self.functions.insert(qualified, fn_info);
                        }
                    };

                    // store variables and their types in the fn_env
                    for arg in &function.sig.arguments {
                        // public_output is a reserved name,
                        // associated automatically to the public output of the main function
                        if RESERVED_ARGS.contains(&arg.name.value.as_str()) {
                            return Err(Error::new(
                                "type-checker",
                                ErrorKind::PublicOutputReserved(arg.name.value.to_string()),
                                arg.name.span,
                            ));
                        }

                        // `pub` arguments are only for the main function
                        if !is_main && arg.is_public() {
                            return Err(Error::new(
                                "type-checker",
                                ErrorKind::PubArgumentOutsideMain,
                                arg.attribute.as_ref().unwrap().span,
                            ));
                        }

                        // `const` arguments are only for non-main functions
                        if is_main && arg.is_constant() {
                            return Err(Error::new(
                                "type-checker",
                                ErrorKind::ConstArgumentNotForMain,
                                arg.name.span,
                            ));
                        }

                        // store the args' type in the fn environment
                        let arg_typ = arg.typ.kind.clone();

                        if arg.is_constant() {
                            typed_fn_env.store_type(
                                arg.name.value.clone(),
                                TypeInfo::new_cst(arg_typ, arg.span),
                            )?;
                        } else {
                            typed_fn_env.store_type(
                                arg.name.value.clone(),
                                TypeInfo::new(arg_typ, arg.span),
                            )?;
                        }
                    }

                    // the output value returned by the main function is also a main_args with a special name (public_output)
                    if let Some(typ) = &function.sig.return_type {
                        if is_main {
                            if !matches!(typ.kind, TyKind::Field) {
                                unimplemented!();
                            }

                            typed_fn_env.store_type(
                                "public_output".to_string(),
                                TypeInfo::new_mut(typ.kind.clone(), typ.span),
                            )?;
                        }
                    }

                    // type system pass on the function body
                    self.check_block(
                        &mut typed_fn_env,
                        &function.body,
                        function.sig.return_type.as_ref(),
                    )?;
                }

                RootKind::Use(_)
                | RootKind::ConstDef(_)
                | RootKind::StructDef(_)
                | RootKind::Comment(_) => (),
            };
        }

        Ok(())
    }
}
