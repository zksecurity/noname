use std::collections::HashMap;

use crate::{
    constants::Span,
    error::{Error, ErrorKind, Result},
    imports::{resolve_builtin_functions, resolve_imports, FnKind},
    parser::{
        Expr, ExprKind, FnSig, Function, Op2, Path, RootKind, Stmt, StmtKind, Struct, Ty, TyKind,
        UsePath, AST,
    },
    stdlib::ImportedModule,
    syntax::is_type,
};

//
// Consts
//

const RESERVED_ARGS: [&str; 1] = ["public_output"];

//
// Data Structures
//

/// Some type information on local variables that we want to track.
#[derive(Debug, Clone)]
pub struct TypeInfo {
    /// If the variable can be mutated or not.
    pub mutable: bool,

    /// A variable becomes disabled once we exit its scope.
    /// We do this instead of deleting a variable to detect shadowing.
    pub disabled: bool,

    /// Some type information.
    pub typ: TyKind,

    /// The span of the variable declaration.
    pub span: Span,
}

impl TypeInfo {
    pub fn new(typ: TyKind, span: Span) -> Self {
        Self {
            mutable: false,
            disabled: false,
            typ,
            span,
        }
    }

    pub fn new_mut(typ: TyKind, span: Span) -> Self {
        Self {
            mutable: true,
            ..Self::new(typ, span)
        }
    }
}

#[derive(Default, Debug)]
pub struct StructInfo {
    name: String,
    fields: Vec<(String, TyKind)>,
    pub methods: HashMap<String, Function>,
}

/// The environment we use to type check a noname program.
#[derive(Default, Debug)]
pub struct TypedGlobalEnv {
    /// the functions present in the scope
    /// contains at least the set of builtin functions (like assert_eq)
    pub functions: HashMap<String, FnInfo>,

    /// stores the imported modules
    pub modules: HashMap<String, ImportedModule>,

    /// If there's a main function in this module, then this is true
    /// (in other words, it's not a library)
    pub has_main: bool,

    /// Custom structs type information and ASTs for methods.
    structs: HashMap<String, StructInfo>,

    /// Constants declared in this module.
    constants: HashMap<String, Ty>,

    /// Mapping from node id to TyKind.
    /// This can be used by the circuit-writer when it needs type information.
    node_types: HashMap<usize, TyKind>,
}

impl TypedGlobalEnv {
    pub fn expr_type(&self, expr: &Expr) -> Option<&TyKind> {
        self.node_types.get(&expr.node_id)
    }

    pub fn node_type(&self, node_id: usize) -> Option<&TyKind> {
        self.node_types.get(&node_id)
    }

    pub fn struct_info(&self, name: &str) -> Option<&StructInfo> {
        self.structs.get(name)
    }
}

#[derive(Debug, Clone)]
pub struct FnInfo {
    pub kind: FnKind,
    pub span: Span,
}

impl FnInfo {
    pub fn sig(&self) -> &FnSig {
        match &self.kind {
            FnKind::BuiltIn(sig, _) => sig,
            FnKind::Native(func) => &func.sig,
            FnKind::Main(sig) => sig,
        }
    }
}

impl TypedGlobalEnv {
    pub fn resolve_global_imports(&mut self) -> Result<()> {
        let builtin_functions = resolve_builtin_functions();
        for fn_info in builtin_functions {
            if self
                .functions
                .insert(fn_info.sig().name.name.value.clone(), fn_info)
                .is_some()
            {
                panic!("global imports conflict (TODO: better error)");
            }
        }

        Ok(())
    }

    pub fn import(&mut self, path: &UsePath) -> Result<()> {
        let module = resolve_imports(path)?;

        if self
            .modules
            .insert(module.name.clone(), module.clone())
            .is_some()
        {
            return Err(Error {
                kind: ErrorKind::DuplicateModule(module.name.clone()),
                span: module.span,
            });
        }

        Ok(())
    }
}

/// The environment we use to type check functions.
#[derive(Default, Debug, Clone)]
pub struct TypedFnEnv {
    /// The current nesting level.
    /// Starting at 0 (top level), and increasing as we go into a block.
    current_scope: usize,

    /// Vars local to their scope.
    /// This needs to be garbage collected when we exit a scope.
    // TODO: there's an output_type field that's a reserved keyword?
    vars: HashMap<String, (usize, TypeInfo)>,
}

impl TypedFnEnv {
    /// Creates a new TypeEnv
    pub fn new() -> Self {
        Self::default()
    }

    /// Enters a scoped block.
    pub fn nest(&mut self) {
        self.current_scope += 1;
    }

    /// Exits a scoped block.
    pub fn pop(&mut self) {
        self.current_scope.checked_sub(1).expect("scope bug");

        // disable variables as we exit the scope
        for (_name, (scope, type_info)) in self.vars.iter_mut() {
            if *scope > self.current_scope {
                type_info.disabled = true;
            }
        }
    }

    /// Returns true if a scope is a prefix of our scope.
    pub fn is_in_scope(&self, prefix_scope: usize) -> bool {
        self.current_scope >= prefix_scope
    }

    /// Stores type information about a local variable.
    /// Note that we forbid shadowing at all scopes.
    pub fn store_type(&mut self, ident: String, type_info: TypeInfo) -> Result<()> {
        match self
            .vars
            .insert(ident.clone(), (self.current_scope, type_info.clone()))
        {
            Some(_) => Err(Error {
                kind: ErrorKind::DuplicateDefinition(ident),
                span: type_info.span,
            }),
            None => Ok(()),
        }
    }

    pub fn get_type(&self, ident: &str) -> Option<&TyKind> {
        self.get_type_info(ident).map(|type_info| &type_info.typ)
    }

    pub fn mutable(&self, ident: &str) -> Option<bool> {
        self.get_type_info(ident).map(|type_info| type_info.mutable)
    }

    /// Retrieves type information on a variable, given a name.
    /// If the variable is not in scope, return false.
    // TODO: return an error no?
    pub fn get_type_info(&self, ident: &str) -> Option<&TypeInfo> {
        if let Some((scope, type_info)) = self.vars.get(ident) {
            if self.is_in_scope(*scope) && !type_info.disabled {
                Some(type_info)
            } else {
                None
            }
        } else {
            None
        }
    }
}

//
// Expr
//

/// Type checks a structure access. For example:
/// `some_struct.some_field.some_other_field`
fn check_struct_access(
    typed_global_env: &mut TypedGlobalEnv,
    typed_fn_env: &mut TypedFnEnv,
    path: &Path,
) -> Result<TyKind> {
    // first element in the path is a struct
    let mut items = path.name.iter();
    let struct_var = items.next().expect("path should have at least one element");

    // get the type of the struct the item refers to
    let mut struct_typ = typed_fn_env
        .get_type(&struct_var.value)
        .expect("couldn't find the type of the struct (TODO: better error)");

    // go through the path
    for item in items {
        // get the name of that custom type
        let struct_name = match struct_typ {
            TyKind::Custom(name) => name,
            _ => panic!("cannot access a field on a non-struct variable (TODO: better error)"),
        };

        // get type information on that custom type
        let struct_info = typed_global_env
            .structs
            .get(struct_name)
            .expect("could not find struct in scope (TODO: better error)");

        // look for the field in the struct
        let struct_fields = &struct_info.fields;

        let field_type = struct_fields
            .iter()
            .find(|(name, _)| name == &item.value)
            .expect("couldn't find the field in the struct (TODO: better error)");

        struct_typ = &field_type.1;
    }

    Ok(struct_typ.clone())
}

/// Same as check_path, except that the last element is known to be a method call.
fn check_path_to_method(
    typed_global_env: &mut TypedGlobalEnv,
    typed_fn_env: &mut TypedFnEnv,
    path: &Path,
) -> Result<FnSig> {
    // at the very least our path should look like this:
    // `struct_var.method_name`
    assert!(path.name.len() > 1);

    // first element in the path is a struct
    let mut items = path.name.iter();
    let struct_var = items.next().expect("expected struct var");

    // get the type of the struct the item refers to
    let mut struct_typ = typed_fn_env.get_type(&struct_var.value);

    // go through the path
    let last_item = items.len() - 1;
    for (idx, item) in items.enumerate() {
        // get the name of that custom type
        let struct_name = match struct_typ {
            Some(TyKind::Custom(name)) => name,
            _ => panic!("cannot access a field on a non-struct variable (TODO: better error)"),
        };

        // get type information on that custom type
        let struct_info = typed_global_env
            .structs
            .get(struct_name)
            .expect("could not find struct in scope (TODO: better error)");

        // if it's the last item, we're looking for a method
        if idx == last_item {
            let method_type = struct_info
                .methods
                .get(&item.value)
                .expect("method not found (TODO: better error)");

            return Ok(method_type.sig.clone());
        } else {
            // look for the field in the struct
            let struct_fields = &struct_info.fields;

            let field_type = struct_fields
                .iter()
                .find(|(name, _)| name == &item.value)
                .map(|(_, typ)| typ);

            struct_typ = field_type;
        }
    }

    unreachable!();
}

impl Expr {
    pub fn compute_type(
        &self,
        typed_global_env: &mut TypedGlobalEnv,
        typed_fn_env: &mut TypedFnEnv,
    ) -> Result<Option<TyKind>> {
        let typ = match &self.kind {
            ExprKind::FnCall { path, args } => {
                check_fn_call(typed_global_env, typed_fn_env, path, args, self.span)?
            }

            ExprKind::Assignment { lhs, rhs } => {
                // lhs can be a local variable or a path to an array
                let path = match &lhs.kind {
                    ExprKind::Variable(var) => var,
                    ExprKind::ArrayAccess { .. } => todo!(),
                    _ => panic!("bad expression assignment (TODO: replace with error)"),
                };

                // we don't support path for now
                if path.name.len() > 1 {
                    unimplemented!();
                }

                let name = &path.name[0];

                // but not an external variable or something
                if path.module.is_some() {
                    panic!("can't assign to an external variable (TODO: replace with error)");
                }

                // check that the var exists locally
                let lhs_info = typed_fn_env
                    .get_type_info(&name.value)
                    .expect("variable not found (TODO: replace with error")
                    .clone();

                // and is mutable
                if !lhs_info.mutable {
                    panic!("variable is not mutable (TODO: replace with error)");
                }

                // and is of the same type as the rhs
                let rhs_typ = rhs.compute_type(typed_global_env, typed_fn_env)?.unwrap();
                if lhs_info.typ != rhs_typ {
                    panic!("lhs type doesn't match rhs type (TODO: replace with error)");
                }

                None
            }

            ExprKind::Op(op, lhs, rhs) => {
                let lhs_typ = lhs.compute_type(typed_global_env, typed_fn_env)?.unwrap();
                let rhs_typ = rhs.compute_type(typed_global_env, typed_fn_env)?.unwrap();

                if lhs_typ != rhs_typ {
                    // only allow bigint mixed with field
                    match (&lhs_typ, &rhs_typ) {
                        (TyKind::BigInt, TyKind::Field) | (TyKind::Field, TyKind::BigInt) => (),
                        _ => {
                            return Err(Error {
                                kind: ErrorKind::MismatchType(lhs_typ.clone(), rhs_typ.clone()),
                                span: self.span,
                            })
                        }
                    }
                }

                match op {
                    Op2::Equality => Some(TyKind::Bool),
                    Op2::Addition
                    | Op2::Subtraction
                    | Op2::Multiplication
                    | Op2::Division
                    | Op2::BoolAnd
                    | Op2::BoolOr
                    | Op2::BoolNot => Some(lhs_typ),
                }
            }

            ExprKind::Negated(inner) => {
                let inner_typ = inner.compute_type(typed_global_env, typed_fn_env)?.unwrap();
                if !matches!(inner_typ, TyKind::Bool) {
                    return Err(Error {
                        kind: ErrorKind::MismatchType(TyKind::Bool, inner_typ),
                        span: self.span,
                    });
                }

                Some(TyKind::Bool)
            }

            ExprKind::BigInt(_) => Some(TyKind::BigInt),

            ExprKind::Bool(_) => Some(TyKind::Bool),

            // mod::path.of.var
            ExprKind::Variable(path) => {
                // we don't support module variables for now
                if path.module.is_some() {
                    unimplemented!();
                }

                // a path means we're accessing the field of a struct
                if path.name.len() > 1 {
                    let last_item_typ = check_struct_access(typed_global_env, typed_fn_env, path)?;

                    Some(last_item_typ)
                } else if path.name.len() == 1 {
                    let name = &path.name[0];

                    // check if it's a constant first
                    let typ = if let Some(typ) = typed_global_env.constants.get(&name.value) {
                        typ.kind.clone()
                    } else {
                        // otherwise it's a local variable
                        typed_fn_env
                            .get_type(&name.value)
                            .ok_or(Error {
                                kind: ErrorKind::UndefinedVariable,
                                span: name.span,
                            })?
                            .clone()
                    };

                    Some(typ)
                } else {
                    panic!("empty path detected");
                }
            }

            ExprKind::ArrayAccess { path, idx } => {
                // only support scoped variable for now
                if path.module.is_some() {
                    unimplemented!();
                }

                // we don't support paths for now
                if path.name.len() > 1 {
                    unimplemented!();
                }
                let name = &path.name[0];

                // figure out if variable is in scope
                let typ = typed_fn_env
                    .get_type(&name.value)
                    .ok_or(Error {
                        kind: ErrorKind::UndefinedVariable,
                        span: self.span,
                    })?
                    .clone();

                // check that expression is a bigint
                match idx.compute_type(typed_global_env, typed_fn_env)? {
                    Some(TyKind::BigInt) => (),
                    _ => {
                        return Err(Error {
                            kind: ErrorKind::ExpectedConstant,
                            span: self.span,
                        })
                    }
                };

                //
                match typ {
                    TyKind::Array(typkind, _) => Some(*typkind),
                    _ => panic!("not an array"),
                }
            }

            ExprKind::ArrayDeclaration(items) => {
                let len: u32 = items.len().try_into().expect("array too large");

                let mut tykind: Option<TyKind> = None;

                for item in items {
                    let item_typ = item
                        .compute_type(typed_global_env, typed_fn_env)?
                        .expect("expected a value");

                    if let Some(tykind) = &tykind {
                        if tykind != &item_typ {
                            return Err(Error {
                                kind: ErrorKind::MismatchType(tykind.clone(), item_typ),
                                span: self.span,
                            });
                        }
                    } else {
                        tykind = Some(item_typ);
                    }
                }

                let tykind = tykind.expect("empty array declaration?");

                match tykind {
                    TyKind::Field | TyKind::BigInt => (),
                    _ => panic!("arrays can only be of field or bigint"),
                };

                Some(TyKind::Array(Box::new(tykind), len))
            }

            ExprKind::CustomTypeDeclaration(name, fields) => {
                let name = &name.value;
                let struct_info = typed_global_env.structs.get(name).ok_or(Error {
                    kind: ErrorKind::UndefinedStruct(name.clone()),
                    span: self.span,
                })?;

                let defined_fields = &struct_info.fields.clone();

                if defined_fields.len() != fields.len() {
                    return Err(Error {
                        kind: ErrorKind::MismatchStructFields(name.clone()),
                        span: self.span,
                    });
                }

                for (defined, observed) in defined_fields.into_iter().zip(fields) {
                    if defined.0 != observed.0.value {
                        return Err(Error {
                            kind: ErrorKind::InvalidStructField(
                                defined.0.clone(),
                                observed.0.value.clone(),
                            ),
                            span: self.span,
                        });
                    }

                    let observed_typ = observed
                        .1
                        .compute_type(typed_global_env, typed_fn_env)?
                        .expect("expected a value (TODO: better error)");

                    if defined.1 != observed_typ {
                        match (&defined.1, &observed_typ) {
                            (TyKind::Field, TyKind::BigInt) | (TyKind::BigInt, TyKind::Field) => (),
                            _ => {
                                return Err(Error {
                                    kind: ErrorKind::InvalidStructFieldType(
                                        defined.1.clone(),
                                        observed_typ,
                                    ),
                                    span: self.span,
                                });
                            }
                        };
                    }
                }

                Some(TyKind::Custom(name.clone()))
            }
        };

        // save the type of that expression in our typed global env
        if let Some(typ) = &typ {
            typed_global_env
                .node_types
                .insert(self.node_id, typ.clone());
        }

        // return the type to the caller
        Ok(typ)
    }
}

//
// Type checking
//

/// TAST for Typed-AST. Not sure how else to call this,
/// this is to make sure we call this compilation phase before the actual compilation.
pub struct TAST {
    pub ast: AST,
    pub typed_global_env: TypedGlobalEnv,
}

impl TAST {
    /// This takes the AST produced by the parser, and performs two things:
    /// - resolves imports
    /// - type checks
    pub fn analyze(ast: AST) -> Result<TAST> {
        //
        // inject some utility builtin functions in the scope
        //

        let mut typed_global_env = TypedGlobalEnv::default();

        // TODO: should we really import them by default?
        typed_global_env.resolve_global_imports()?;

        //
        // Resolve imports
        //

        for root in &ast.0 {
            match &root.kind {
                // `use crypto::poseidon;`
                RootKind::Use(path) => typed_global_env.import(path)?,
                RootKind::Function(_) => (),
                RootKind::Struct(_) => (),
                RootKind::Comment(_) => (),
                RootKind::Const(_) => (),
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

                    typed_global_env
                        .structs
                        .insert(name.value.clone(), struct_info);
                }

                RootKind::Const(cst) => {
                    typed_global_env.constants.insert(
                        cst.name.value.clone(),
                        Ty {
                            kind: TyKind::Field,
                            span: cst.span,
                        },
                    );
                }

                RootKind::Use(_) | RootKind::Function(_) | RootKind::Comment(_) => (),
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

                    // if this is main, witness it
                    let is_main = function.is_main();
                    if is_main {
                        typed_global_env.has_main = true;
                    }

                    // save the function in the typed global env
                    let fn_kind = if is_main {
                        FnKind::Main(function.sig.clone())
                    } else {
                        FnKind::Native(function.clone())
                    };
                    let fn_info = FnInfo {
                        kind: fn_kind,
                        span: function.span,
                    };

                    if let Some(self_name) = &function.sig.name.self_name {
                        let struct_info = typed_global_env
                            .structs
                            .get_mut(&self_name.value)
                            .expect("couldn't find the struct for storing the method");

                        struct_info
                            .methods
                            .insert(function.sig.name.name.value.clone(), function.clone());
                    } else {
                        typed_global_env
                            .functions
                            .insert(function.sig.name.name.value.clone(), fn_info);
                    }

                    // store variables and their types in the fn_env
                    for arg in &function.sig.arguments {
                        // public_output is a reserved name,
                        // associated automatically to the public output of the main function
                        if RESERVED_ARGS.contains(&arg.name.value.as_str()) {
                            return Err(Error {
                                kind: ErrorKind::PublicOutputReserved(arg.name.value.to_string()),
                                span: arg.name.span,
                            });
                        }

                        // `pub` arguments are only for the main function
                        if !is_main && arg.is_public() {
                            return Err(Error {
                                kind: ErrorKind::PubArgumentOutsideMain,
                                span: arg.name.span,
                            });
                        }

                        match &arg.typ.kind {
                            TyKind::Field => {
                                typed_fn_env.store_type(
                                    arg.name.value.clone(),
                                    TypeInfo::new(arg.typ.kind.clone(), arg.span),
                                )?;
                            }

                            TyKind::Array(..) => {
                                typed_fn_env.store_type(
                                    arg.name.value.clone(),
                                    TypeInfo::new(arg.typ.kind.clone(), arg.span),
                                )?;
                            }

                            TyKind::Bool => {
                                typed_fn_env.store_type(
                                    arg.name.value.clone(),
                                    TypeInfo::new(arg.typ.kind.clone(), arg.span),
                                )?;
                            }

                            TyKind::Custom(custom) => {
                                typed_fn_env.store_type(
                                    "self".to_string(),
                                    TypeInfo::new(TyKind::Custom(custom.clone()), arg.span),
                                )?;
                            }

                            t => panic!("unimplemented type {:?}", t),
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
                    Self::check_block(
                        &mut typed_global_env,
                        &mut typed_fn_env,
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

        Ok(TAST {
            ast,
            typed_global_env,
        })
    }

    pub fn check_block(
        typed_global_env: &mut TypedGlobalEnv,
        typed_fn_env: &mut TypedFnEnv,
        stmts: &[Stmt],
        expected_return: Option<&Ty>,
    ) -> Result<()> {
        // enter the scope
        typed_fn_env.nest();

        let mut return_typ = None;

        for stmt in stmts {
            if return_typ.is_some() {
                panic!("early return detected: we don't allow that for now (TODO: return error");
            }

            return_typ = Self::check_stmt(typed_global_env, typed_fn_env, stmt)?;
        }

        // check the return
        if let Some(expected) = expected_return {
            let observed = match return_typ {
                None => {
                    return Err(Error {
                        kind: ErrorKind::MissingPublicOutput,
                        span: expected.span,
                    })
                }
                Some(e) => e,
            };

            if expected.kind != observed {
                panic!(
                    "returned type is not the same as expected return type (TODO: return an error)"
                );
            }
        }

        // exit the scope
        typed_fn_env.pop();

        Ok(())
    }

    pub fn check_stmt(
        typed_global_env: &mut TypedGlobalEnv,
        typed_fn_env: &mut TypedFnEnv,
        stmt: &Stmt,
    ) -> Result<Option<TyKind>> {
        match &stmt.kind {
            StmtKind::Assign { mutable, lhs, rhs } => {
                // inferance can be easy: we can do it the Golang way and just use the type that rhs has (in `let` assignments)

                // but first we need to compute the type of the rhs expression
                let typ = rhs.compute_type(typed_global_env, typed_fn_env)?.unwrap();

                let type_info = if *mutable {
                    TypeInfo::new_mut(typ, lhs.span)
                } else {
                    TypeInfo::new(typ, lhs.span)
                };

                // store the type of lhs in the env
                typed_fn_env.store_type(lhs.value.clone(), type_info)?;
            }
            StmtKind::For { var, range, body } => {
                // enter a new scope
                typed_fn_env.nest();

                // create var (for now it's always a bigint)
                typed_fn_env
                    .store_type(var.value.clone(), TypeInfo::new(TyKind::BigInt, var.span))?;

                // ensure start..end makes sense
                if range.end < range.start {
                    panic!("end can't be smaller than start (TODO: better error)");
                }

                // check block
                Self::check_block(typed_global_env, typed_fn_env, body, None)?;

                // exit the scope
                typed_fn_env.pop();
            }
            StmtKind::Expr(expr) => {
                // make sure the expression does not return any type
                // (it's a statement expression, it should only work via side effect)

                let typ = expr.compute_type(typed_global_env, typed_fn_env)?;
                if typ.is_some() {
                    return Err(Error {
                        kind: ErrorKind::UnusedReturnValue,
                        span: expr.span,
                    });
                }
            }
            StmtKind::Return(res) => {
                let typ = res.compute_type(typed_global_env, typed_fn_env)?.unwrap();

                return Ok(Some(typ));
            }
            StmtKind::Comment(_) => (),
        }

        Ok(None)
    }
}

/// type checks a function call.
/// Note that this can be a method call as well.
/// (A method is defined on a custom type.)
pub fn check_fn_call(
    typed_global_env: &mut TypedGlobalEnv,
    typed_fn_env: &mut TypedFnEnv,
    path: &Path,
    args: &[Expr],
    span: Span,
) -> Result<Option<TyKind>> {
    // retrieve the function sig in the env
    let sig: FnSig = if let Some(module) = &path.module {
        // get name of function
        assert_eq!(path.name.len(), 1);
        let fn_name = &path.name[0].value;
        let fn_name_span = path.name[0].span;

        // check module present in the scope
        let module_val = &module.value;
        let imported_module = typed_global_env.modules.get(module_val).ok_or(Error {
            kind: ErrorKind::UndefinedModule(module_val.clone()),
            span: module.span,
        })?;
        let fn_info = imported_module.functions.get(fn_name).ok_or(Error {
            kind: ErrorKind::UndefinedFunction(fn_name.clone()),
            span: fn_name_span,
        })?;
        fn_info.sig().clone()
    } else if path.name.len() > 1 {
        // method call detected
        check_path_to_method(typed_global_env, typed_fn_env, path)?
    } else {
        // get name of function
        assert_eq!(path.name.len(), 1);
        let fn_name = &path.name[0].value;
        let fn_name_span = path.name[0].span;

        // functions present in the scope
        let fn_info = typed_global_env.functions.get(fn_name).ok_or(Error {
            kind: ErrorKind::UndefinedFunction(fn_name.clone()),
            span: fn_name_span,
        })?;
        fn_info.sig().clone()
    };

    // canonicalize the arguments depending on method call or not
    let expected: Vec<_> = if path.name.len() > 1 {
        sig.arguments
            .iter()
            .filter(|arg| arg.name.value != "self")
            .collect()
    } else {
        sig.arguments.iter().collect()
    };

    // compute the observed arguments types
    let mut observed = Vec::with_capacity(args.len());
    for arg in args {
        if let Some(typ) = arg.compute_type(typed_global_env, typed_fn_env)? {
            observed.push((typ.clone(), arg.span));
        } else {
            return Err(Error {
                kind: ErrorKind::CannotComputeExpression,
                span: arg.span,
            });
        }
    }

    // check argument length
    if expected.len() != observed.len() {
        return Err(Error {
            kind: ErrorKind::MismatchFunctionArguments(observed.len(), expected.len()),
            span,
        });
    }

    // compare argument types with the function signature
    for (sig_arg, (typ, span)) in expected.iter().zip(observed) {
        if sig_arg.typ.kind != typ {
            // it's ok if a bigint is supposed to be a field no?
            // TODO: replace bigint -> constant?
            if matches!(
                (&sig_arg.typ.kind, &typ),
                (TyKind::Field, TyKind::BigInt) | (TyKind::BigInt, TyKind::Field)
            ) {
                continue;
            }

            return Err(Error {
                kind: ErrorKind::ArgumentTypeMismatch(sig_arg.typ.kind.clone(), typ),
                span,
            });
        }
    }

    Ok(sig.return_type.as_ref().map(|ty| ty.kind.clone()))
}
