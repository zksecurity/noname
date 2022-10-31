use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    cli::packages::UserRepo,
    constants::Span,
    error::{Error, ErrorKind, Result},
    imports::{resolve_builtin_functions, FnKind, Module},
    parser::types::{Expr, ExprKind, FnSig, Function, Op2, Stmt, StmtKind, Ty, TyKind, UsePath},
    syntax::is_type,
};

use super::{Dependencies, TypeChecker, TypeInfo, TypedFnEnv};

/// Keeps track of the signature of a user-defined function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FnInfo {
    pub kind: FnKind,
    pub span: Span,
}

impl FnInfo {
    pub fn sig(&self) -> &FnSig {
        match &self.kind {
            FnKind::BuiltIn(sig, _) => sig,
            FnKind::Native(func) => &func.sig,
        }
    }
}

/// Keeps track of the signature of a user-defined struct.
#[derive(Deserialize, Serialize, Default, Debug, Clone)]
pub struct StructInfo {
    pub name: String,
    pub fields: Vec<(String, TyKind)>,
    pub methods: HashMap<String, Function>,
}

/// Information that we need to pass around between expression nodes when type checking.
struct ExprTyInfo {
    /// This is needed to obtain information on variables.
    /// For example, parsing of an assignment expression needs to know if the lhs variable is mutable.
    var_name: Option<String>,

    /// The type of the expression node.
    typ: TyKind,
}

impl ExprTyInfo {
    fn new(var_name: Option<String>, typ: TyKind) -> Self {
        Self { var_name, typ }
    }

    fn new_var(var_name: String, typ: TyKind) -> Self {
        Self {
            var_name: Some(var_name),
            typ,
        }
    }

    fn new_anon(typ: TyKind) -> Self {
        Self {
            var_name: None,
            typ,
        }
    }
}

impl TypeChecker {
    pub fn expr_type(&self, expr: &Expr) -> Option<&TyKind> {
        self.node_types.get(&expr.node_id)
    }

    pub fn node_type(&self, node_id: usize) -> Option<&TyKind> {
        self.node_types.get(&node_id)
    }

    pub fn struct_info(&self, name: &str) -> Option<&StructInfo> {
        self.structs.get(name)
    }

    pub fn fn_info(&self, name: &str) -> Option<&FnInfo> {
        self.functions.get(name)
    }

    /// Returns the number of field elements contained in the given type.
    // TODO: might want to memoize that at some point
    pub fn size_of(&self, deps: &Dependencies, typ: &TyKind) -> Result<usize> {
        let res = match typ {
            TyKind::Field => 1,
            TyKind::Custom { module, name } => {
                let struct_name = &name.value;
                let struct_info = if let Some(module) = module {
                    // check module present in the scope
                    let module_val = &module.value;
                    let imported_module = self.modules.get(module_val).ok_or_else(|| {
                        Error::new(ErrorKind::UndefinedModule(module_val.clone()), module.span)
                    })?;

                    deps.get_struct(imported_module, name)?
                } else {
                    self.struct_info(struct_name)
                        .ok_or(Error::new(
                            ErrorKind::UndefinedStruct(struct_name.clone()),
                            name.span,
                        ))?
                        .clone()
                };

                let mut sum = 0;

                for (_, t) in &struct_info.fields {
                    sum += self.size_of(deps, t)?;
                }

                sum
            }
            TyKind::BigInt => 1,
            TyKind::Array(typ, len) => (*len as usize) * self.size_of(deps, typ)?,
            TyKind::Bool => 1,
        };
        Ok(res)
    }

    pub fn resolve_global_imports(&mut self) -> Result<()> {
        let builtin_functions = resolve_builtin_functions();
        for (fn_name, fn_info) in builtin_functions {
            if self.functions.insert(fn_name, fn_info).is_some() {
                panic!("global imports conflict (TODO: better error)");
            }
        }

        Ok(())
    }

    pub fn import(&mut self, path: &UsePath) -> Result<()> {
        if self
            .modules
            .insert(path.submodule.value.clone(), path.clone())
            .is_some()
        {
            return Err(Error::new(
                ErrorKind::DuplicateModule(path.submodule.value.clone()),
                path.submodule.span,
            ));
        }

        Ok(())
    }

    fn compute_type(
        &mut self,
        expr: &Expr,
        typed_fn_env: &mut TypedFnEnv,
        deps: &Dependencies,
    ) -> Result<Option<ExprTyInfo>> {
        let typ: Option<ExprTyInfo> = match &expr.kind {
            ExprKind::FieldAccess { lhs, rhs } => {
                // compute type of left-hand side
                let lhs_node = self
                    .compute_type(lhs, typed_fn_env, deps)?
                    .expect("type-checker bug: field access on an empty var");

                // obtain the type of the field
                let struct_name = match lhs_node.typ {
                    TyKind::Custom { module, name } => name,
                    _ => panic!("field access must be done on a custom struct"),
                };

                // get struct info
                let struct_info = self
                    .struct_info(&struct_name.value)
                    .expect("this struct is not defined, or you're trying to access a field of a struct defined in a third-party library (TODO: better error)");

                // find field type
                let res = struct_info
                    .fields
                    .iter()
                    .find(|(name, _)| name == &rhs.value)
                    .map(|(_, typ)| typ.clone())
                    .expect("could not find field");

                Some(ExprTyInfo::new(lhs_node.var_name, res))
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

                    let imported_module = self.modules.get(module_val).ok_or_else(|| {
                        Error::new(ErrorKind::UndefinedModule(module_val.clone()), module.span)
                    })?;

                    let fn_info = deps.get_fn(imported_module, fn_name)?;

                    fn_info.sig().clone()
                } else {
                    // functions present in the scope
                    let fn_info = self.functions.get(&fn_name.value).ok_or_else(|| {
                        Error::new(
                            ErrorKind::UndefinedFunction(fn_name.value.clone()),
                            fn_name.span,
                        )
                    })?;
                    fn_info.sig().clone()
                };

                // type check the function call
                let method_call = false;
                let res =
                    self.check_fn_call(typed_fn_env, deps, method_call, fn_sig, args, expr.span)?;

                res.map(ExprTyInfo::new_anon)
            }

            // `lhs.method_name(args)`
            ExprKind::MethodCall {
                lhs,
                method_name,
                args,
            } => {
                // retrieve struct name on the lhs
                let lhs_type = self.compute_type(lhs, typed_fn_env, deps)?;
                let (module, struct_name) = match lhs_type.map(|t| t.typ) {
                    Some(TyKind::Custom { module, name }) => (module, name),
                    _ => {
                        return Err(Error::new(
                            ErrorKind::MethodCallOnNonCustomStruct,
                            expr.span,
                        ))
                    }
                };

                // get struct info
                let struct_info = if let Some(module) = module {
                    let imported_module = self.modules.get(&module.value).ok_or_else(|| {
                        Error::new(
                            ErrorKind::UndefinedModule(module.value.clone()),
                            module.span,
                        )
                    })?;

                    deps.get_struct(imported_module, &struct_name)?
                } else {
                    self.struct_info(&struct_name.value)
                        .ok_or(Error::new(
                            ErrorKind::UndefinedStruct(struct_name.value.clone()),
                            struct_name.span,
                        ))?
                        .clone()
                };

                // get method info
                let method_type = struct_info
                    .methods
                    .get(&method_name.value)
                    .expect("method not found on custom struct (TODO: better error)");

                // type check the method call
                let method_call = true;
                let res = self.check_fn_call(
                    typed_fn_env,
                    deps,
                    method_call,
                    method_type.sig.clone(),
                    args,
                    expr.span,
                )?;

                res.map(|ty| ExprTyInfo::new(None, ty))
            }

            ExprKind::Assignment { lhs, rhs } => {
                // compute type of lhs
                let lhs_node = self
                    .compute_type(lhs, typed_fn_env, deps)?
                    .expect("type-checker bug: lhs access on an empty var");

                // lhs can be a local variable or a path to an array
                let lhs_name = match &lhs.kind {
                    // `name = <rhs>`
                    ExprKind::Variable { module, name } => {
                        if module.is_some() {
                            panic!("cannot assign to an external variable");
                        }

                        name.value.clone()
                    }

                    // `array[idx] = <rhs>`
                    ExprKind::ArrayAccess { array, idx } => {
                        // get variable behind array
                        let array_node = self
                            .compute_type(array, typed_fn_env, deps)?
                            .expect("type-checker bug: array access on an empty var");

                        array_node
                            .var_name
                            .expect("anonymous array access cannot be mutated")
                    }

                    // `struct.field = <rhs>`
                    ExprKind::FieldAccess { lhs, rhs } => {
                        // get variable behind lhs
                        let lhs_node = self
                            .compute_type(lhs, typed_fn_env, deps)?
                            .expect("type-checker bug: lhs access on an empty var");

                        lhs_node
                            .var_name
                            .expect("anonymous lhs access cannot be mutated")
                    }
                    _ => panic!("bad expression assignment (TODO: replace with error)"),
                };

                // check that the var exists locally
                let lhs_info = typed_fn_env
                    .get_type_info(&lhs_name)
                    .expect("variable not found (TODO: replace with error")
                    .clone();

                // and is mutable
                if !lhs_info.mutable {
                    return Err(Error::new(
                        ErrorKind::AssignmentToImmutableVariable,
                        expr.span,
                    ));
                }

                // and is of the same type as the rhs
                let rhs_typ = self.compute_type(rhs, typed_fn_env, deps)?.unwrap();

                if !rhs_typ.typ.match_expected(&lhs_node.typ) {
                    panic!("lhs type doesn't match rhs type (TODO: replace with error)");
                }

                None
            }

            ExprKind::BinaryOp { op, lhs, rhs, .. } => {
                let lhs_node = self
                    .compute_type(lhs, typed_fn_env, deps)?
                    .expect("type-checker bug");
                let rhs_node = self
                    .compute_type(rhs, typed_fn_env, deps)?
                    .expect("type-checker bug");

                if lhs_node.typ != rhs_node.typ {
                    // only allow bigint mixed with field
                    match (&lhs_node.typ, &rhs_node.typ) {
                        (TyKind::BigInt, TyKind::Field) | (TyKind::Field, TyKind::BigInt) => (),
                        _ => {
                            return Err(Error::new(
                                ErrorKind::MismatchType(lhs_node.typ.clone(), rhs_node.typ.clone()),
                                expr.span,
                            ))
                        }
                    }
                }

                let typ = match op {
                    Op2::Equality => TyKind::Bool,
                    Op2::Addition
                    | Op2::Subtraction
                    | Op2::Multiplication
                    | Op2::Division
                    | Op2::BoolAnd
                    | Op2::BoolOr => lhs_node.typ,
                };

                Some(ExprTyInfo::new_anon(typ))
            }

            ExprKind::Negated(inner) => {
                let inner_typ = self.compute_type(inner, typed_fn_env, deps)?.unwrap();
                if !matches!(inner_typ.typ, TyKind::Field | TyKind::BigInt) {
                    return Err(Error::new(
                        ErrorKind::MismatchType(TyKind::Field, inner_typ.typ),
                        expr.span,
                    ));
                }

                Some(ExprTyInfo::new_anon(TyKind::Field))
            }

            ExprKind::Not(inner) => {
                let inner_typ = self.compute_type(inner, typed_fn_env, deps)?.unwrap();
                if !matches!(inner_typ.typ, TyKind::Bool) {
                    return Err(Error::new(
                        ErrorKind::MismatchType(TyKind::Bool, inner_typ.typ),
                        expr.span,
                    ));
                }

                Some(ExprTyInfo::new_anon(TyKind::Bool))
            }

            ExprKind::BigInt(_) => Some(ExprTyInfo::new_anon(TyKind::BigInt)),

            ExprKind::Bool(_) => Some(ExprTyInfo::new_anon(TyKind::Bool)),

            // mod::path.of.var
            ExprKind::Variable { module, name } => {
                if is_type(&name.value) {
                    // if it's a type, make sure it exists
                    let _struct_info = self
                        .struct_info(&name.value)
                        .expect("custom type does not exist");

                    // and return its type
                    let res = ExprTyInfo::new_anon(TyKind::Custom {
                        module: module.clone(),
                        name: name.clone(),
                    });
                    Some(res)
                } else {
                    // if it's a variable,
                    // check if it's a constant first
                    let typ = if let Some(cst) = self.constants.get(&name.value) {
                        // if it's a field, we need to convert it to a bigint
                        if matches!(cst.typ.kind, TyKind::Field) {
                            TyKind::BigInt
                        } else {
                            cst.typ.kind.clone()
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

                    let res = ExprTyInfo::new_var(name.value.clone(), typ);
                    Some(res)
                }
            }

            ExprKind::ArrayAccess { array, idx } => {
                // get type of lhs
                let typ = self.compute_type(array, typed_fn_env, deps)?.unwrap();

                // check that it is an array
                if !matches!(typ.typ, TyKind::Array(..)) {
                    return Err(Error::new(ErrorKind::ArrayAccessOnNonArray, expr.span));
                }

                // check that expression is a bigint
                let idx_typ = self.compute_type(idx, typed_fn_env, deps)?;
                match idx_typ.map(|t| t.typ) {
                    Some(TyKind::BigInt) => (),
                    _ => return Err(Error::new(ErrorKind::ExpectedConstant, expr.span)),
                };

                // get type of element
                let el_typ = match typ.typ {
                    TyKind::Array(typkind, _) => *typkind,
                    _ => panic!("not an array"),
                };

                let res = ExprTyInfo::new(typ.var_name, el_typ);
                Some(res)
            }

            ExprKind::ArrayDeclaration(items) => {
                let len: u32 = items.len().try_into().expect("array too large");

                let mut tykind: Option<TyKind> = None;

                for item in items {
                    let item_typ = self
                        .compute_type(item, typed_fn_env, deps)?
                        .expect("expected a value");

                    if let Some(tykind) = &tykind {
                        if !tykind.same_as(&item_typ.typ) {
                            return Err(Error::new(
                                ErrorKind::MismatchType(tykind.clone(), item_typ.typ),
                                expr.span,
                            ));
                        }
                    } else {
                        tykind = Some(item_typ.typ);
                    }
                }

                let tykind = tykind.expect("empty array declaration?");

                let res = ExprTyInfo::new_anon(TyKind::Array(Box::new(tykind), len));
                Some(res)
            }

            ExprKind::IfElse { cond, then_, else_ } => {
                // cond can only be a boolean
                let cond_node = self
                    .compute_type(cond, typed_fn_env, deps)?
                    .expect("can't compute type of condition");
                if !matches!(cond_node.typ, TyKind::Bool) {
                    panic!("`if` must be followed by a boolean");
                }

                // then_ and else_ can only be variables, field accesses, or array accesses
                if !matches!(
                    &then_.kind,
                    ExprKind::Variable { .. }
                        | ExprKind::FieldAccess { .. }
                        | ExprKind::ArrayAccess { .. }
                ) {
                    panic!("`if` branch must be a variable, a field access, or an array access. It can't be logic that creates constraints.");
                }

                if !matches!(
                    &else_.kind,
                    ExprKind::Variable { .. }
                        | ExprKind::FieldAccess { .. }
                        | ExprKind::ArrayAccess { .. }
                ) {
                    panic!("`else` branch must be a variable, a field access, or an array access. It can't be logic that creates constraints.");
                }

                // compute type of if/else branches
                let then_node = self
                    .compute_type(then_, typed_fn_env, deps)?
                    .expect("can't compute type of first branch of `if/else`");
                let else_node = self
                    .compute_type(else_, typed_fn_env, deps)?
                    .expect("can't compute type of first branch of `if/else`");

                // make sure that the type of then_ and else_ match
                if then_node.typ != else_node.typ {
                    panic!("`if` branch and `else` branch must have matching types");
                }

                //
                Some(ExprTyInfo::new_anon(then_node.typ))
            }

            ExprKind::CustomTypeDeclaration {
                struct_name,
                fields,
            } => {
                let name = &struct_name.value;
                let struct_info = self.structs.get(name).ok_or_else(|| {
                    Error::new(ErrorKind::UndefinedStruct(name.clone()), expr.span)
                })?;

                let defined_fields = &struct_info.fields.clone();

                if defined_fields.len() != fields.len() {
                    return Err(Error::new(
                        ErrorKind::MismatchStructFields(name.clone()),
                        expr.span,
                    ));
                }

                for (defined, observed) in defined_fields.iter().zip(fields) {
                    if defined.0 != observed.0.value {
                        return Err(Error::new(
                            ErrorKind::InvalidStructField(
                                defined.0.clone(),
                                observed.0.value.clone(),
                            ),
                            expr.span,
                        ));
                    }

                    let observed_typ = self
                        .compute_type(&observed.1, typed_fn_env, deps)?
                        .expect("expected a value (TODO: better error)");

                    if !observed_typ.typ.match_expected(&defined.1) {
                        return Err(Error::new(
                            ErrorKind::InvalidStructFieldType(defined.1.clone(), observed_typ.typ),
                            expr.span,
                        ));
                    }
                }

                let res = ExprTyInfo::new_anon(TyKind::Custom {
                    module: None,
                    name: struct_name.clone(),
                });
                Some(res)
            }
        };

        // save the type of that expression in our typed global env
        if let Some(typ) = &typ {
            self.node_types.insert(expr.node_id, typ.typ.clone());
        }

        // return the type to the caller
        Ok(typ)
    }

    pub fn check_block(
        &mut self,
        typed_fn_env: &mut TypedFnEnv,
        deps: &Dependencies,
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

            return_typ = self.check_stmt(typed_fn_env, deps, stmt)?;
        }

        // check the return
        match (expected_return, return_typ) {
            (None, None) => (),
            (Some(expected), None) => {
                return Err(Error::new(ErrorKind::MissingReturn, expected.span))
            }
            (None, Some(_)) => {
                return Err(Error::new(
                    ErrorKind::NoReturnExpected,
                    stmts.last().unwrap().span,
                ))
            }
            (Some(expected), Some(observed)) => {
                if !observed.match_expected(&expected.kind) {
                    return Err(Error::new(
                        ErrorKind::ReturnTypeMismatch(expected.kind.clone(), observed.clone()),
                        expected.span,
                    ));
                }
            }
        };

        // exit the scope
        typed_fn_env.pop();

        Ok(())
    }

    pub fn check_stmt(
        &mut self,
        typed_fn_env: &mut TypedFnEnv,
        deps: &Dependencies,
        stmt: &Stmt,
    ) -> Result<Option<TyKind>> {
        match &stmt.kind {
            StmtKind::Assign { mutable, lhs, rhs } => {
                // inferance can be easy: we can do it the Golang way and just use the type that rhs has (in `let` assignments)

                // but first we need to compute the type of the rhs expression
                let node = self.compute_type(rhs, typed_fn_env, deps)?.unwrap();

                let type_info = if *mutable {
                    TypeInfo::new_mut(node.typ, lhs.span)
                } else {
                    TypeInfo::new(node.typ, lhs.span)
                };

                // store the type of lhs in the env
                typed_fn_env.store_type(lhs.value.clone(), type_info)?;
            }
            StmtKind::ForLoop { var, range, body } => {
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
                self.check_block(typed_fn_env, deps, body, None)?;

                // exit the scope
                typed_fn_env.pop();
            }
            StmtKind::Expr(expr) => {
                // make sure the expression does not return any type
                // (it's a statement expression, it should only work via side effect)

                let typ = self.compute_type(expr, typed_fn_env, deps)?;
                if typ.is_some() {
                    return Err(Error::new(ErrorKind::UnusedReturnValue, expr.span));
                }
            }
            StmtKind::Return(res) => {
                let node = self.compute_type(res, typed_fn_env, deps)?.unwrap();

                return Ok(Some(node.typ));
            }
            StmtKind::Comment(_) => (),
        }

        Ok(None)
    }

    /// type checks a function call.
    /// Note that this can also be a method call.
    pub fn check_fn_call(
        &mut self,
        typed_fn_env: &mut TypedFnEnv,
        deps: &Dependencies,
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
            if let Some(node) = self.compute_type(arg, typed_fn_env, deps)? {
                observed.push((node.typ.clone(), arg.span));
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
            if !typ.match_expected(&sig_arg.typ.kind) {
                return Err(Error::new(
                    ErrorKind::ArgumentTypeMismatch(sig_arg.typ.kind.clone(), typ),
                    span,
                ));
            }
        }

        // return the return type of the function
        Ok(fn_sig.return_type.as_ref().map(|ty| ty.kind.clone()))
    }
}
