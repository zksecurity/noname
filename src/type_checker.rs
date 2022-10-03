use std::collections::HashMap;

use crate::{
    constants::Span,
    error::{Error, ErrorKind, Result},
    imports::{resolve_builtin_functions, resolve_imports, FnKind},
    parser::{
        Expr, ExprKind, FnSig, Function, Op2, RootKind, Stmt, StmtKind, Struct, Ty, TyKind,
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

    /// If the variable is a constant or not.
    pub constant: bool,

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
            constant: false,
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

    pub fn new_cst(typ: TyKind, span: Span) -> Self {
        Self {
            constant: true,
            ..Self::new(typ, span)
        }
    }
}

#[derive(Default, Debug)]
pub struct StructInfo {
    pub name: String,
    pub fields: Vec<(String, TyKind)>,
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

    /// Returns the number of field elements contained in the given type.
    // TODO: might want to memoize that at some point
    pub fn size_of(&self, typ: &TyKind) -> usize {
        match typ {
            TyKind::Field => 1,
            TyKind::Custom(c) => {
                let struct_info = self
                    .struct_info(c)
                    .expect("couldn't find struct info of {c}");
                struct_info
                    .fields
                    .iter()
                    .map(|(_, t)| self.size_of(t))
                    .sum()
            }
            TyKind::BigInt => 1,
            TyKind::Array(typ, len) => (*len as usize) * self.size_of(typ),
            TyKind::Bool => 1,
        }
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
            return Err(Error::new(
                ErrorKind::DuplicateModule(module.name.clone()),
                module.span,
            ));
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
            Some(_) => Err(Error::new(
                ErrorKind::DuplicateDefinition(ident),
                type_info.span,
            )),
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

impl Expr {
    // TODO: implement this on typed_global_env, and modify the name of typed_global_env to type_checker?
    pub fn compute_type(
        &self,
        typed_global_env: &mut TypedGlobalEnv,
        typed_fn_env: &mut TypedFnEnv,
    ) -> Result<Option<TyKind>> {
        let typ = match &self.kind {
            ExprKind::FieldAccess { lhs, rhs } => {
                // compute type of left-hand side
                let lhs_type = lhs.compute_type(typed_global_env, typed_fn_env)?;

                // obtain the type of the field
                let struct_name = match lhs_type {
                    Some(TyKind::Custom(name)) => name,
                    _ => panic!("field access must be done on a custom struct"),
                };

                // get struct info
                let struct_info = typed_global_env
                    .struct_info(&struct_name)
                    .expect("this struct is not defined");

                // find field type
                let res = struct_info
                    .fields
                    .iter()
                    .find(|(name, _)| name == &rhs.value)
                    .map(|(_, typ)| typ.clone())
                    .expect("could not find field");

                Some(res)
            }

            // `module::fn_name(args)`
            ExprKind::FnCall {
                module,
                fn_name,
                args,
            } => {
                // retrieve the function signature
                let fn_sig: FnSig = if let Some(module) = module {
                    // check module present in the scope
                    let module_val = &module.value;
                    let imported_module =
                        typed_global_env.modules.get(module_val).ok_or_else(|| {
                            Error::new(ErrorKind::UndefinedModule(module_val.clone()), module.span)
                        })?;
                    let fn_info =
                        imported_module
                            .functions
                            .get(&fn_name.value)
                            .ok_or_else(|| {
                                Error::new(
                                    ErrorKind::UndefinedFunction(fn_name.value.clone()),
                                    fn_name.span,
                                )
                            })?;
                    fn_info.sig().clone()
                } else {
                    // functions present in the scope
                    let fn_info =
                        typed_global_env
                            .functions
                            .get(&fn_name.value)
                            .ok_or_else(|| {
                                Error::new(
                                    ErrorKind::UndefinedFunction(fn_name.value.clone()),
                                    fn_name.span,
                                )
                            })?;
                    fn_info.sig().clone()
                };

                // type check the function call
                let method_call = false;
                check_fn_call(
                    typed_global_env,
                    typed_fn_env,
                    method_call,
                    fn_sig,
                    args,
                    self.span,
                )?
            }

            // `lhs.method_name(args)`
            ExprKind::MethodCall {
                lhs,
                method_name,
                args,
            } => {
                // retrieve struct name on the lhs
                let lhs_type = lhs.compute_type(typed_global_env, typed_fn_env)?;
                let struct_name = match lhs_type {
                    Some(TyKind::Custom(name)) => name,
                    _ => panic!("method call can only be applied on custom structs"),
                };

                // get struct info
                let struct_info = typed_global_env
                    .structs
                    .get(&struct_name)
                    .expect("could not find struct in scope (TODO: better error)");

                // get method info
                let method_type = struct_info
                    .methods
                    .get(&method_name.value)
                    .expect("method not found on custom struct (TODO: better error)");

                // type check the method call
                let method_call = true;
                check_fn_call(
                    typed_global_env,
                    typed_fn_env,
                    method_call,
                    method_type.sig.clone(),
                    args,
                    self.span,
                )?
            }

            ExprKind::Assignment { lhs, rhs } => {
                // lhs can be a local variable or a path to an array
                let lhs_name = match &lhs.kind {
                    ExprKind::Variable { module, name } => {
                        if module.is_some() {
                            panic!("cannot assign to an external variable");
                        }
                        name
                    }
                    ExprKind::ArrayAccess { .. } => todo!(),
                    ExprKind::FieldAccess { .. } => {
                        todo!()
                    }
                    _ => panic!("bad expression assignment (TODO: replace with error)"),
                };

                // check that the var exists locally
                let lhs_info = typed_fn_env
                    .get_type_info(&lhs_name.value)
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

            ExprKind::Op { op, lhs, rhs } => {
                let lhs_typ = lhs.compute_type(typed_global_env, typed_fn_env)?.unwrap();
                let rhs_typ = rhs.compute_type(typed_global_env, typed_fn_env)?.unwrap();

                if lhs_typ != rhs_typ {
                    // only allow bigint mixed with field
                    match (&lhs_typ, &rhs_typ) {
                        (TyKind::BigInt, TyKind::Field) | (TyKind::Field, TyKind::BigInt) => (),
                        _ => {
                            return Err(Error::new(
                                ErrorKind::MismatchType(lhs_typ.clone(), rhs_typ.clone()),
                                self.span,
                            ))
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
                    return Err(Error::new(
                        ErrorKind::MismatchType(TyKind::Bool, inner_typ),
                        self.span,
                    ));
                }

                Some(TyKind::Bool)
            }

            ExprKind::BigInt(_) => Some(TyKind::BigInt),

            ExprKind::Bool(_) => Some(TyKind::Bool),

            // mod::path.of.var
            ExprKind::Variable { module, name } => {
                // sanitize
                if module.is_some() {
                    panic!("we don't support module variables for now");
                }

                if is_type(&name.value) {
                    // if the variable is a type, make sure it exists
                    let _struct_info = typed_global_env
                        .struct_info(&name.value)
                        .expect("custom type does not exist");

                    // and return its type
                    Some(TyKind::Custom(name.value.clone()))
                } else {
                    // if it's a variable,
                    // check if it's a constant first
                    let typ = if let Some(typ) = typed_global_env.constants.get(&name.value) {
                        // if it's a field, we need to convert it to a bigint
                        if matches!(typ.kind, TyKind::Field) {
                            TyKind::BigInt
                        } else {
                            typ.kind.clone()
                        }
                    } else {
                        // otherwise it's a local variable
                        let typ = typed_fn_env
                            .get_type(&name.value)
                            .ok_or_else(|| Error::new(ErrorKind::UndefinedVariable, name.span))?
                            .clone();
                        // if it's a field, we need to convert it to a bigint
                        if matches!(typ, TyKind::Field) {
                            TyKind::BigInt
                        } else {
                            typ
                        }
                    };

                    Some(typ)
                }
            }

            ExprKind::ArrayAccess { array, idx } => {
                // get type of lhs
                let typ = array.compute_type(typed_global_env, typed_fn_env)?.unwrap();

                // check that it is an array
                if !matches!(typ, TyKind::Array(..)) {
                    panic!("array access can only be performed on arrays");
                }

                // check that expression is a bigint
                let idx_typ = idx.compute_type(typed_global_env, typed_fn_env)?;
                match idx_typ {
                    Some(TyKind::BigInt) => (),
                    _ => return Err(Error::new(ErrorKind::ExpectedConstant, self.span)),
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
                            return Err(Error::new(
                                ErrorKind::MismatchType(tykind.clone(), item_typ),
                                self.span,
                            ));
                        }
                    } else {
                        tykind = Some(item_typ);
                    }
                }

                let tykind = tykind.expect("empty array declaration?");

                Some(TyKind::Array(Box::new(tykind), len))
            }

            ExprKind::CustomTypeDeclaration {
                struct_name: name,
                fields,
            } => {
                let name = &name.value;
                let struct_info = typed_global_env.structs.get(name).ok_or_else(|| {
                    Error::new(ErrorKind::UndefinedStruct(name.clone()), self.span)
                })?;

                let defined_fields = &struct_info.fields.clone();

                if defined_fields.len() != fields.len() {
                    return Err(Error::new(
                        ErrorKind::MismatchStructFields(name.clone()),
                        self.span,
                    ));
                }

                for (defined, observed) in defined_fields.iter().zip(fields) {
                    if defined.0 != observed.0.value {
                        return Err(Error::new(
                            ErrorKind::InvalidStructField(
                                defined.0.clone(),
                                observed.0.value.clone(),
                            ),
                            self.span,
                        ));
                    }

                    let observed_typ = observed
                        .1
                        .compute_type(typed_global_env, typed_fn_env)?
                        .expect("expected a value (TODO: better error)");

                    if defined.1 != observed_typ {
                        // we accept constants as `Field` types.
                        if !matches!((&defined.1, &observed_typ), (TyKind::Field, TyKind::BigInt)) {
                            return Err(Error::new(
                                ErrorKind::InvalidStructFieldType(defined.1.clone(), observed_typ),
                                self.span,
                            ));
                        }
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
                            return Err(Error::new(
                                ErrorKind::PublicOutputReserved(arg.name.value.to_string()),
                                arg.name.span,
                            ));
                        }

                        // `pub` arguments are only for the main function
                        if !is_main && arg.is_public() {
                            return Err(Error::new(
                                ErrorKind::PubArgumentOutsideMain,
                                arg.attribute.as_ref().unwrap().span,
                            ));
                        }

                        // `const` arguments are only for non-main functions
                        if is_main && arg.is_constant() {
                            return Err(Error::new(
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
                None => return Err(Error::new(ErrorKind::MissingPublicOutput, expected.span)),
                Some(e) => e,
            };

            if expected.kind != observed
                && !matches!(
                    (&expected.kind, observed),
                    (TyKind::Field, TyKind::BigInt) | (TyKind::BigInt, TyKind::Field)
                )
            {
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
                    return Err(Error::new(ErrorKind::UnusedReturnValue, expr.span));
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
/// Note that this can also be a method call.
pub fn check_fn_call(
    typed_global_env: &mut TypedGlobalEnv,
    typed_fn_env: &mut TypedFnEnv,
    method_call: bool, // indicates if it's a fn call or a method call
    fn_sig: FnSig,
    args: &[Expr],
    span: Span,
) -> Result<Option<TyKind>> {
    // canonicalize the arguments depending on method call or not
    let expected: Vec<_> = if method_call {
        fn_sig
            .arguments
            .iter()
            .filter(|arg| arg.name.value != "self")
            .collect()
    } else {
        fn_sig.arguments.iter().collect()
    };

    // compute the observed arguments types
    let mut observed = Vec::with_capacity(args.len());
    for arg in args {
        if let Some(typ) = arg.compute_type(typed_global_env, typed_fn_env)? {
            observed.push((typ.clone(), arg.span));
        } else {
            return Err(Error::new(ErrorKind::CannotComputeExpression, arg.span));
        }
    }

    // check argument length
    if expected.len() != observed.len() {
        return Err(Error::new(
            ErrorKind::MismatchFunctionArguments(observed.len(), expected.len()),
            span,
        ));
    }

    // compare argument types with the function signature
    for (sig_arg, (typ, span)) in expected.iter().zip(observed) {
        if sig_arg.typ.kind != typ {
            // we accept constants as [Field] types
            if matches!((&sig_arg.typ.kind, &typ), (TyKind::Field, TyKind::BigInt)) {
                continue;
            }

            return Err(Error::new(
                ErrorKind::ArgumentTypeMismatch(sig_arg.typ.kind.clone(), typ),
                span,
            ));
        }
    }

    // return the return type of the function
    Ok(fn_sig.return_type.as_ref().map(|ty| ty.kind.clone()))
}
