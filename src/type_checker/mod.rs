use std::collections::HashMap;

use crate::{
    constants::Field,
    error::{Error, ErrorKind, Result},
    imports::FnKind,
    parser::{
        types::{Const, Function, RootKind, Struct, Ty, TyKind, UsePath},
        AST,
    },
};

pub use checker::{FnInfo, StructInfo};
pub use dependencies::Dependencies;
pub use fn_env::{TypeInfo, TypedFnEnv};

use miette::NamedSource;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

pub mod checker;
pub mod dependencies;
pub mod fn_env;

const RESERVED_ARGS: [&str; 1] = ["public_output"];

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstInfo {
    #[serde_as(as = "crate::serialization::SerdeAs")]
    pub value: Field,
    pub typ: Ty,
}

/// The environment we use to type check a noname program.
#[derive(Debug, Serialize, Deserialize)]
pub struct TypeChecker {
    /// The filename containining the code
    pub filename: String,

    /// The source code
    pub src: String,

    /// the functions present in the scope
    /// contains at least the set of builtin functions (like assert_eq)
    pub functions: HashMap<String, FnInfo>,

    /// maps `module` to its original `use a::module`
    pub modules: HashMap<String, UsePath>,

    /// Custom structs type information and ASTs for methods.
    pub structs: HashMap<String, StructInfo>,

    /// Constants declared in this module.
    pub constants: HashMap<String, ConstInfo>,

    /// Mapping from node id to TyKind.
    /// This can be used by the circuit-writer when it needs type information.
    pub node_types: HashMap<usize, TyKind>,
}

impl TypeChecker {
    fn new(filename: String, src: String) -> Self {
        Self {
            filename,
            src,
            functions: HashMap::new(),
            modules: HashMap::new(),
            structs: HashMap::new(),
            constants: HashMap::new(),
            node_types: HashMap::new(),
        }
    }

    pub fn analyze(filename: String, code: String, ast: AST, deps: &Dependencies) -> Result<Self> {
        let res = Self::analyze_inner(filename.clone(), code.clone(), ast, deps);

        res.map_err(|mut err| {
            err.src = NamedSource::new(filename, code);
            err
        })
    }

    /// This takes the AST produced by the parser, and performs two things:
    /// - resolves imports
    /// - type checks
    fn analyze_inner(
        filename: String,
        code: String,
        ast: AST,
        deps: &Dependencies,
    ) -> Result<Self> {
        //
        // inject some utility builtin functions in the scope
        //

        let mut type_checker = TypeChecker::new(filename, code);

        // TODO: should we really import them by default?
        type_checker.resolve_global_imports()?;

        //
        // Resolve imports
        //

        let mut abort = None;

        for root in &ast.0 {
            match &root.kind {
                // `use crypto::poseidon;`
                RootKind::Use(path) => {
                    // important: no struct or function definition must appear before a use declaration
                    if let Some(span) = abort {
                        return Err(Error::new(
                            "type-checker",
                            ErrorKind::OrderOfUseDeclaration,
                            span,
                        ));
                    }
                    type_checker.import(path)?
                }
                RootKind::Function(Function { span, .. })
                | RootKind::Struct(Struct { span, .. })
                | RootKind::Const(Const { span, .. }) => abort = Some(*span),
                RootKind::Comment(_) => (),
            }
        }

        //
        // Process constants
        //

        // we detect struct or function definition
        let mut abort = None;

        for root in &ast.0 {
            match &root.kind {
                RootKind::Const(cst) => {
                    // important: no struct or function definition must appear before a constant declaration
                    if let Some(span) = abort {
                        return Err(Error::new(
                            "type-checker",
                            ErrorKind::OrderOfConstDeclaration,
                            span,
                        ));
                    }

                    if type_checker
                        .constants
                        .insert(
                            cst.name.value.clone(),
                            ConstInfo {
                                value: cst.value,
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

                RootKind::Function(Function { span, .. })
                | RootKind::Struct(Struct { span, .. }) => abort = Some(*span),

                RootKind::Use(_) | RootKind::Comment(_) => (),
            }
        }

        //
        // Type check structs
        //

        for root in &ast.0 {
            match &root.kind {
                // `use crypto::poseidon;`
                RootKind::Struct(struct_) => {
                    let Struct { name, fields, .. } = struct_;

                    let fields: Vec<_> = fields
                        .iter()
                        .map(|field| {
                            let (name, typ) = field;
                            (name.value.clone(), typ.kind.clone())
                        })
                        .collect();

                    let struct_info = StructInfo {
                        name: name.value.clone(),
                        fields,
                        methods: HashMap::new(),
                    };

                    type_checker.structs.insert(name.value.clone(), struct_info);
                }

                RootKind::Const(_)
                | RootKind::Use(_)
                | RootKind::Function(_)
                | RootKind::Comment(_) => (),
            }
        }

        //
        // Semantic analysis includes:
        // - type checking
        // - ?
        //

        for root in &ast.0 {
            match &root.kind {
                // `fn main() { ... }`
                RootKind::Function(function) => {
                    // create a new typed fn environment to type check the function
                    let mut typed_fn_env = TypedFnEnv::default();

                    // if this is the main function check that it has arguments
                    let is_main = function.is_main();
                    if is_main && function.sig.arguments.is_empty() {
                        return Err(Error::new(
                            "type-checker",
                            ErrorKind::NoArgsInMain,
                            function.span,
                        ));
                    }

                    // save the function in the typed global env
                    let fn_kind = FnKind::Native(function.clone());
                    let fn_info = FnInfo {
                        kind: fn_kind,
                        span: function.span,
                    };

                    if let Some(self_name) = &function.sig.name.self_name {
                        let struct_info = type_checker
                            .structs
                            .get_mut(&self_name.value)
                            .expect("couldn't find the struct for storing the method");

                        struct_info
                            .methods
                            .insert(function.sig.name.name.value.clone(), function.clone());
                    } else {
                        type_checker
                            .functions
                            .insert(function.sig.name.name.value.clone(), fn_info);
                    }

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

                    // type system pass
                    type_checker.check_block(
                        &mut typed_fn_env,
                        deps,
                        &function.body,
                        function.sig.return_type.as_ref(),
                    )?;
                }

                RootKind::Use(_)
                | RootKind::Const(_)
                | RootKind::Struct(_)
                | RootKind::Comment(_) => (),
            }
        }

        Ok(type_checker)
    }
}
