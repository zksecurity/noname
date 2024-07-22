use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    backends::Backend,
    constants::Span,
    error::{ErrorKind, Result},
    imports::FnKind,
    parser::{
        types::{is_numeric, FnSig, FunctionDef, Ident, Stmt, StmtKind, Symbolic, Ty, TyKind},
        CustomType, Expr, ExprKind, Op2,
    },
    syntax::{is_generic_parameter, is_type},
};

use super::{FullyQualified, TypeChecker, TypeInfo, TypedFnEnv};

/// Keeps track of the signature of a user-defined function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FnInfo<B>
where
    B: Backend,
{
    pub kind: FnKind<B>,
    pub span: Span,
    // todo: generics for inferred values
    // - will be useful for builtins that will need to know the generic values
}

impl<B: Backend> FnInfo<B> {
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
    pub methods: HashMap<String, FunctionDef>,
}

/// Information that we need to pass around between expression nodes when type checking.
#[derive(Debug)]
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

impl<B: Backend> TypeChecker<B> {
    fn compute_type(
        &mut self,
        expr: &Expr,
        typed_fn_env: &mut TypedFnEnv,
    ) -> Result<Option<ExprTyInfo>> {
        let typ: Option<ExprTyInfo> = match &expr.kind {
            ExprKind::FieldAccess { lhs, rhs } => {
                // compute type of left-hand side
                let lhs_node = self
                    .compute_type(lhs, typed_fn_env)?
                    .expect("type-checker bug: field access on an empty var");

                // obtain the type of the field
                let (module, struct_name) = match lhs_node.typ {
                    TyKind::Custom { module, name } => (module, name),
                    _ => panic!("field access must be done on a custom struct"),
                };

                // get struct info
                let qualified = FullyQualified::new(&module, &struct_name);
                let struct_info = self
                    .struct_info(&qualified)
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
                let qualified = FullyQualified::new(&module, &fn_name.value);
                let fn_info = self.fn_info(&qualified).ok_or_else(|| {
                    self.error(
                        ErrorKind::UndefinedFunction(fn_name.value.clone()),
                        fn_name.span,
                    )
                })?;
                let fn_sig = fn_info.sig().clone();

                // type check the function call
                let method_call = false;
                let res = self.check_fn_call(typed_fn_env, method_call, fn_sig, args, expr.span)?;

                res.map(ExprTyInfo::new_anon)
            }

            // `lhs.method_name(args)`
            ExprKind::MethodCall {
                lhs,
                method_name,
                args,
            } => {
                // retrieve struct name on the lhs
                let lhs_type = self.compute_type(lhs, typed_fn_env)?;
                let (module, struct_name) = match lhs_type.map(|t| t.typ) {
                    Some(TyKind::Custom { module, name }) => (module, name),
                    _ => return Err(self.error(ErrorKind::MethodCallOnNonCustomStruct, expr.span)),
                };

                // get struct info
                let qualified = FullyQualified::new(&module, &struct_name);
                let struct_info = self
                    .struct_info(&qualified)
                    .ok_or(self.error(ErrorKind::UndefinedStruct(struct_name.clone()), lhs.span))?
                    .clone();

                // get method info
                let method_type = struct_info
                    .methods
                    .get(&method_name.value)
                    .expect("method not found on custom struct (TODO: better error)");

                // type check the method call
                let method_call = true;
                let res = self.check_fn_call(
                    typed_fn_env,
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
                    .compute_type(lhs, typed_fn_env)?
                    .expect("type-checker bug: lhs access on an empty var");

                // lhs can be a local variable or a path to an array
                let lhs_name = match &lhs.kind {
                    // `name = <rhs>`
                    ExprKind::Variable { module, name } => {
                        // we first check if it's a constant
                        // note: the only way to check that atm is to check in the constants hashmap
                        // this is because we don't differentiate const vars from normal variables
                        // (perhaps we should)
                        let qualified = FullyQualified::new(&module, &name.value);
                        if let Some(_cst_info) = self.const_info(&qualified) {
                            return Err(self.error(
                                ErrorKind::UnexpectedError("cannot assign to an external variable"),
                                lhs.span,
                            ));
                        }

                        name.value.clone()
                    }

                    // `array[idx] = <rhs>`
                    ExprKind::ArrayAccess { array, idx } => {
                        // get variable behind array
                        let array_node = self
                            .compute_type(array, typed_fn_env)?
                            .expect("type-checker bug: array access on an empty var");

                        array_node
                            .var_name
                            .expect("anonymous array access cannot be mutated")
                    }

                    // `struct.field = <rhs>`
                    ExprKind::FieldAccess { lhs, rhs } => {
                        // get variable behind lhs
                        let lhs_node = self
                            .compute_type(lhs, typed_fn_env)?
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
                    return Err(self.error(ErrorKind::AssignmentToImmutableVariable, expr.span));
                }

                // and is of the same type as the rhs
                let rhs_typ = self.compute_type(rhs, typed_fn_env)?.unwrap();

                if !rhs_typ.typ.match_expected(&lhs_node.typ) {
                    panic!("lhs type doesn't match rhs type (TODO: replace with error)");
                }

                None
            }

            ExprKind::BinaryOp { op, lhs, rhs, .. } => {
                let lhs_node = self
                    .compute_type(lhs, typed_fn_env)?
                    .expect("type-checker bug");
                let rhs_node = self
                    .compute_type(rhs, typed_fn_env)?
                    .expect("type-checker bug");

                // generic parameter is assumed to be of numeric type
                if lhs_node.typ != rhs_node.typ
                    && (!crate::parser::types::is_numeric(&lhs_node.typ)
                        || !crate::parser::types::is_numeric(&rhs_node.typ))
                {
                    return Err(self.error(
                        ErrorKind::MismatchType(lhs_node.typ.clone(), rhs_node.typ.clone()),
                        expr.span,
                    ));
                }

                let typ = match op {
                    Op2::Equality => TyKind::Bool,
                    Op2::Inequality => TyKind::Bool,
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
                let inner_typ = self.compute_type(inner, typed_fn_env)?.unwrap();
                if !matches!(inner_typ.typ, TyKind::Field | TyKind::BigInt) {
                    return Err(self.error(
                        ErrorKind::MismatchType(TyKind::Field, inner_typ.typ),
                        expr.span,
                    ));
                }

                Some(ExprTyInfo::new_anon(TyKind::Field))
            }

            ExprKind::Not(inner) => {
                let inner_typ = self.compute_type(inner, typed_fn_env)?.unwrap();
                if !matches!(inner_typ.typ, TyKind::Bool) {
                    return Err(self.error(
                        ErrorKind::MismatchType(TyKind::Bool, inner_typ.typ),
                        expr.span,
                    ));
                }

                Some(ExprTyInfo::new_anon(TyKind::Bool))
            }

            ExprKind::BigUInt(_) => Some(ExprTyInfo::new_anon(TyKind::BigInt)),

            ExprKind::Bool(_) => Some(ExprTyInfo::new_anon(TyKind::Bool)),

            // mod::path.of.var
            ExprKind::Variable { module, name } => {
                let qualified = FullyQualified::new(module, &name.value);

                // generic parameter should be checked as a local variable
                if is_type(&name.value) && !is_generic_parameter(&name.value) {
                    // if it's a type, make sure it exists
                    let _struct_info = self
                        .struct_info(&qualified)
                        .expect("custom type does not exist (TODO: better error)");

                    // and return its type
                    let res = ExprTyInfo::new_anon(TyKind::Custom {
                        module: module.clone(),
                        name: name.value.clone(),
                    });
                    Some(res)
                } else {
                    // if it's a variable,
                    // check if it's a constant first
                    let typ = if let Some(cst) = self.constants.get(&qualified) {
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
                            .ok_or_else(|| self.error(ErrorKind::UndefinedVariable, name.span))?
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
                let typ = self.compute_type(array, typed_fn_env)?.unwrap();

                // check that it is an array
                if !matches!(typ.typ, TyKind::Array(..) | TyKind::GenericArray(..)) {
                    return Err(self.error(ErrorKind::ArrayAccessOnNonArray, expr.span));
                }

                // check that expression is a bigint
                let idx_typ = self.compute_type(idx, typed_fn_env)?;
                match idx_typ.map(|t| t.typ) {
                    Some(TyKind::BigInt) => (),
                    Some(TyKind::Generic(_)) => (),
                    _ => return Err(self.error(ErrorKind::ExpectedConstant, expr.span)),
                };

                // get type of element
                let el_typ = match typ.typ {
                    TyKind::Array(typkind, _) => *typkind,
                    TyKind::GenericArray(typkind, _) => *typkind,
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
                        .compute_type(item, typed_fn_env)?
                        .expect("expected a value");

                    if let Some(tykind) = &tykind {
                        if !tykind.same_as(&item_typ.typ) {
                            return Err(self.error(
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
                    .compute_type(cond, typed_fn_env)?
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
                    .compute_type(then_, typed_fn_env)?
                    .expect("can't compute type of first branch of `if/else`");
                let else_node = self
                    .compute_type(else_, typed_fn_env)?
                    .expect("can't compute type of first branch of `if/else`");

                // make sure that the type of then_ and else_ match
                if then_node.typ != else_node.typ {
                    panic!("`if` branch and `else` branch must have matching types");
                }

                //
                Some(ExprTyInfo::new_anon(then_node.typ))
            }

            ExprKind::CustomTypeDeclaration { custom, fields } => {
                let CustomType {
                    module,
                    name,
                    span: _,
                } = custom;
                let qualified = FullyQualified::new(module, name);
                let struct_info = self.struct_info(&qualified).ok_or_else(|| {
                    self.error(ErrorKind::UndefinedStruct(name.clone()), expr.span)
                })?;

                let defined_fields = &struct_info.fields.clone();

                if defined_fields.len() != fields.len() {
                    return Err(
                        self.error(ErrorKind::MismatchStructFields(name.clone()), expr.span)
                    );
                }

                for (defined, observed) in defined_fields.iter().zip(fields) {
                    if defined.0 != observed.0.value {
                        return Err(self.error(
                            ErrorKind::InvalidStructField(
                                defined.0.clone(),
                                observed.0.value.clone(),
                            ),
                            expr.span,
                        ));
                    }

                    let observed_typ = self
                        .compute_type(&observed.1, typed_fn_env)?
                        .expect("expected a value (TODO: better error)");

                    if !observed_typ.typ.match_expected(&defined.1) {
                        return Err(self.error(
                            ErrorKind::InvalidStructFieldType(defined.1.clone(), observed_typ.typ),
                            expr.span,
                        ));
                    }
                }

                let res = ExprTyInfo::new_anon(TyKind::Custom {
                    module: module.clone(),
                    name: name.clone(),
                });
                Some(res)
            }
            ExprKind::RepeatedArrayDeclaration { item, size } => {
                let item_node = self
                    .compute_type(item, typed_fn_env)?
                    .expect("expected a typed item");

                let size_node = self
                    .compute_type(size, typed_fn_env)?
                    .expect("expected a typed size");

                if crate::parser::types::is_numeric(&size_node.typ) {
                    // todo: see if we can determine whether the size node is generic or not
                    // use generic array to bypass the array check, as the size node might include generic parameters.
                    // the mast phase will resolve the size node to a constant, and check the types.
                    let res = ExprTyInfo::new_anon(TyKind::GenericArray(
                        Box::new(item_node.typ),
                        // mock up a symbolic size
                        Symbolic::Generic(Ident::new("x".to_string(), size.span)),
                    ));
                    Some(res)
                } else {
                    return Err(self.error(ErrorKind::InvalidArraySize, expr.span));
                }
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

            return_typ = self.check_stmt(typed_fn_env, stmt)?;
        }

        // check the return
        match (expected_return, return_typ) {
            (None, None) => (),
            (Some(expected), None) => {
                return Err(self.error(ErrorKind::MissingReturn, expected.span))
            }
            (None, Some(_)) => {
                return Err(self.error(ErrorKind::NoReturnExpected, stmts.last().unwrap().span))
            }
            (Some(expected), Some(observed)) => {
                if !observed.match_expected(&expected.kind) {
                    return Err(self.error(
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
        stmt: &Stmt,
    ) -> Result<Option<TyKind>> {
        match &stmt.kind {
            StmtKind::Assign { mutable, lhs, rhs } => {
                // inferance can be easy: we can do it the Golang way and just use the type that rhs has (in `let` assignments)

                // but first we need to compute the type of the rhs expression
                let node = self.compute_type(rhs, typed_fn_env)?.unwrap();

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

                let start_type = self.compute_type(&range.start, typed_fn_env)?.unwrap();
                let end_type = self.compute_type(&range.end, typed_fn_env)?.unwrap();
                if !is_numeric(&start_type.typ) || !is_numeric(&end_type.typ) {
                    return Err(self.error(ErrorKind::InvalidRangeSize, range.span));
                }
                
                // check block
                self.check_block(typed_fn_env, body, None)?;

                // exit the scope
                typed_fn_env.pop();
            }
            StmtKind::Expr(expr) => {
                // make sure the expression does not return any type
                // (it's a statement expression, it should only work via side effect)

                let typ = self.compute_type(expr, typed_fn_env)?;
                if typ.is_some() {
                    return Err(self.error(ErrorKind::UnusedReturnValue, expr.span));
                }
            }
            StmtKind::Return(res) => {
                let node = self.compute_type(res, typed_fn_env)?.unwrap();

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
        method_call: bool, // indicates if it's a fn call or a method call
        fn_sig: FnSig,
        args: &[Expr],
        span: Span,
    ) -> Result<Option<TyKind>> {
        // check if a function names is in use already by another variable
        match typed_fn_env.get_type_info(&fn_sig.name.value) {
            Some(_) => {
                return Err(self.error(
                    ErrorKind::FunctionNameInUsebyVariable(fn_sig.name.value),
                    fn_sig.name.span,
                ))
            }
            None => (),
        };

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
            if let Some(node) = self.compute_type(arg, typed_fn_env)? {
                observed.push((node.typ.clone(), arg.span));
            } else {
                return Err(self.error(ErrorKind::CannotComputeExpression, arg.span));
            }
        }

        // check argument length
        if expected.len() != observed.len() {
            return Err(self.error(
                ErrorKind::MismatchFunctionArguments(observed.len(), expected.len()),
                span,
            ));
        }

        // compare argument types with the function signature
        for (sig_arg, (typ, span)) in expected.iter().zip(observed) {
            if !typ.match_expected(&sig_arg.typ.kind) {
                return Err(self.error(
                    ErrorKind::ArgumentTypeMismatch(sig_arg.typ.kind.clone(), typ),
                    span,
                ));
            }
        }

        // return the return type of the function
        Ok(fn_sig.return_type.as_ref().map(|ty| ty.kind.clone()))
    }
}
