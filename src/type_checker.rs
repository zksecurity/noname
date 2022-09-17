use std::collections::HashMap;

use crate::{
    constants::Span,
    error::{Error, ErrorKind, Result},
    imports::{resolve_builtin_functions, resolve_imports, FnKind},
    parser::{
        Expr, ExprKind, FnArg, FnSig, Op2, Path, RootKind, Stmt, StmtKind, Struct, Ty, TyKind, AST,
    },
    stdlib::ImportedModule,
    syntax::is_const,
};

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

/// The environment we use to type check a noname program.
#[derive(Default, Debug)]
pub struct TypedGlobalEnv {
    /// the functions present in the scope
    /// contains at least the set of builtin functions (like assert_eq)
    pub functions: HashMap<String, FnInfo>,

    /// stores the imported modules
    pub modules: HashMap<String, ImportedModule>,

    /// the arguments expected by main
    pub main_args: (HashMap<String, FnArg>, Span),

    /// If there's a main function in this module, then this is true
    /// (in other words, it's not a library)
    pub has_main: bool,

    /// Custom structs
    structs: HashMap<String, Vec<(String, TyKind)>>,
}

#[derive(Debug, Clone)]
pub struct FnInfo {
    pub sig: FnSig,
    pub kind: FnKind,
    pub span: Span,
}

impl TypedGlobalEnv {
    pub fn resolve_global_imports(&mut self) -> Result<()> {
        let builtin_functions = resolve_builtin_functions();
        for fn_info in builtin_functions {
            if self
                .functions
                .insert(fn_info.sig.name.value.clone(), fn_info)
                .is_some()
            {
                panic!("global imports conflict (TODO: better error)");
            }
        }

        Ok(())
    }

    pub fn import(&mut self, path: &Path) -> Result<()> {
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

impl Expr {
    pub fn compute_type(
        &self,
        typed_global_env: &TypedGlobalEnv,
        typed_fn_env: &mut TypedFnEnv,
    ) -> Result<Option<TyKind>> {
        match &self.kind {
            ExprKind::FnCall { name, args } => {
                typecheck_fn_call(typed_global_env, typed_fn_env, name, args, self.span)
            }
            ExprKind::Assignment { lhs, rhs } => {
                // lhs can be a local variable or a path to an array
                let name = match &lhs.kind {
                    ExprKind::Identifier(var) => var,
                    ExprKind::ArrayAccess(_, _) => todo!(),
                    _ => panic!("bad expression assignment (TODO: replace with error)"),
                };

                // check that the var exists locally
                let lhs_info = typed_fn_env
                    .get_type_info(name)
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

                Ok(None)
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
                    Op2::Equality => Ok(Some(TyKind::Bool)),
                    Op2::Addition
                    | Op2::Subtraction
                    | Op2::Multiplication
                    | Op2::Division
                    | Op2::BoolAnd
                    | Op2::BoolOr
                    | Op2::BoolNot => Ok(Some(lhs_typ)),
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

                Ok(Some(TyKind::Bool))
            }
            ExprKind::BigInt(_) => Ok(Some(TyKind::BigInt)),
            ExprKind::Bool(_) => Ok(Some(TyKind::Bool)),
            ExprKind::Identifier(ident) => {
                let typ = if is_const(ident) {
                    TyKind::Field
                } else {
                    typed_fn_env
                        .get_type(ident)
                        .ok_or(Error {
                            kind: ErrorKind::UndefinedVariable,
                            span: self.span,
                        })?
                        .clone()
                };

                Ok(Some(typ))
            }
            ExprKind::ArrayAccess(path, expr) => {
                // only support scoped variable for now
                if path.len() != 1 {
                    unimplemented!();
                }

                // figure out if variable is in scope
                let name = &path.path[0].value;
                let typ = typed_fn_env
                    .get_type(name)
                    .ok_or(Error {
                        kind: ErrorKind::UndefinedVariable,
                        span: self.span,
                    })?
                    .clone();

                // check that expression is a bigint
                match expr.compute_type(typed_global_env, typed_fn_env)? {
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
                    TyKind::Array(typkind, _) => Ok(Some(*typkind)),
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

                Ok(Some(TyKind::Array(Box::new(tykind), len)))
            }
            ExprKind::CustomTypeDeclaration(name, fields) => {
                let defined_fields = typed_global_env.structs.get(name).cloned().ok_or(Error {
                    kind: ErrorKind::UndefinedStruct(name.to_string()),
                    span: self.span,
                })?;

                if defined_fields.len() != fields.len() {
                    return Err(Error {
                        kind: ErrorKind::MismatchStructFields(name.to_string()),
                        span: self.span,
                    });
                }

                for (defined, observed) in defined_fields.iter().zip(fields) {
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

                Ok(Some(TyKind::Custom(name.to_string())))
            }
            ExprKind::StructAccess(name, field) => {
                let struct_name = typed_fn_env
                    .get_type(&name.value)
                    .expect("could not find variable in scope (TODO: better error)");

                let struct_name = match struct_name {
                    TyKind::Custom(name) => name,
                    _ => panic!(
                        "cannot access a field on a non-struct variable (TODO: better error)"
                    ),
                };

                let struct_type = typed_global_env
                    .structs
                    .get(struct_name)
                    .expect("could not find struct in scope (TODO: better error)");

                let field_type = struct_type
                    .iter()
                    .find(|(name, _)| name == &field.value)
                    .map(|(_, typ)| typ)
                    .expect("could not find given field in custom struct (TODO: better error");

                Ok(Some(field_type.clone()))
            }
        }
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

                    let fields = fields
                        .iter()
                        .map(|field| {
                            let (name, typ) = field;
                            (name.value.clone(), typ.kind.clone())
                        })
                        .collect();
                    typed_global_env.structs.insert(name.value.clone(), fields);
                }
                RootKind::Use(_)
                | RootKind::Function(_)
                | RootKind::Comment(_)
                | RootKind::Const(_) => (),
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
                    let mut typed_fn_env = TypedFnEnv::default();

                    // if this is main, witness it
                    if function.is_main() {
                        typed_global_env.has_main = true;
                        typed_global_env.main_args.1 = function.span;
                    }

                    // store variables and their types in the env
                    for arg in &function.arguments {
                        // public_output is a reserved name,
                        // associated automatically to the public output of the main function
                        if arg.name.value == "public_output" {
                            return Err(Error {
                                kind: ErrorKind::PublicOutputReserved,
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

                            t => panic!("unimplemented type {:?}", t),
                        }

                        // when we witness main, we do some stuff
                        if function.is_main() {
                            typed_global_env
                                .main_args
                                .0
                                .insert(arg.name.value.clone(), arg.clone());
                        }
                    }

                    // the output value returned by the main function is also a main_args with a special name (public_output)
                    if let Some(typ) = &function.return_type {
                        if !matches!(typ.kind, TyKind::Field) {
                            unimplemented!();
                        }

                        let name = "public_output";

                        typed_fn_env.store_type(
                            name.to_string(),
                            TypeInfo::new_mut(typ.kind.clone(), typ.span),
                        )?;
                    }

                    // type system pass
                    Self::check_block(
                        &typed_global_env,
                        &mut typed_fn_env,
                        &function.body,
                        function.return_type.as_ref(),
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
        typed_global_env: &TypedGlobalEnv,
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
        typed_global_env: &TypedGlobalEnv,
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
                        kind: ErrorKind::ExpectedUnitExpr,
                        span: expr.span,
                    });
                }
            }
            StmtKind::Return(res) => {
                let typ = res.compute_type(typed_global_env, typed_fn_env)?.unwrap();

                let expected = match typed_fn_env.get_type("public_output") {
                    Some(t) => t,
                    None => panic!("return statement when function signature doesn't have a return value (TODO: replace by error)"),
                };

                if expected != &typ {
                    return Err(Error {
                        kind: ErrorKind::ReturnTypeMismatch(expected.clone(), typ),
                        span: stmt.span,
                    });
                }

                return Ok(Some(typ));
            }
            StmtKind::Comment(_) => (),
        }

        Ok(None)
    }
}

pub fn typecheck_fn_call(
    typed_global_env: &TypedGlobalEnv,
    typed_fn_env: &mut TypedFnEnv,
    name: &Path,
    args: &[Expr],
    span: Span,
) -> Result<Option<TyKind>> {
    // retrieve the function sig in the env
    let path_len = name.path.len();
    let sig: FnSig = if path_len == 1 {
        // functions present in the scope
        let fn_name = &name.path[0].value;
        let fn_info = typed_global_env.functions.get(fn_name).ok_or(Error {
            kind: ErrorKind::UndefinedFunction(fn_name.clone()),
            span: name.span,
        })?;
        fn_info.sig.clone()
    } else if path_len == 2 {
        // check module present in the scope
        let module = &name.path[0];
        let fn_name = &name.path[1];
        let module = typed_global_env.modules.get(&module.value).ok_or(Error {
            kind: ErrorKind::UndefinedModule(module.value.clone()),
            span: module.span,
        })?;
        let fn_info = module.functions.get(&fn_name.value).ok_or(Error {
            kind: ErrorKind::UndefinedFunction(fn_name.value.clone()),
            span: fn_name.span,
        })?;
        fn_info.sig.clone()
    } else {
        return Err(Error {
            kind: ErrorKind::InvalidFnCall("sub-sub modules unsupported"),
            span: name.span,
        });
    };

    // compute the arguments
    let mut typs = Vec::with_capacity(args.len());
    for arg in args {
        if let Some(typ) = arg.compute_type(typed_global_env, typed_fn_env)? {
            typs.push((typ.clone(), arg.span));
        } else {
            return Err(Error {
                kind: ErrorKind::CannotComputeExpression,
                span: arg.span,
            });
        }
    }

    // argument length
    if sig.arguments.len() != typs.len() {
        return Err(Error {
            kind: ErrorKind::WrongNumberOfArguments {
                expected_args: sig.arguments.len(),
                observed_args: typs.len(),
            },
            span,
        });
    }

    // compare argument types with the function signature
    for (sig_arg, (typ, span)) in sig.arguments.iter().zip(typs) {
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
