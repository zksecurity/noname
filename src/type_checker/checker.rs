use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    backends::Backend,
    constants::Span,
    error::{ErrorKind, Result},
    imports::FnKind,
    parser::{
        types::{ArraySize, FnSig, FunctionDef, ModulePath, NumOrVar, Stmt, StmtKind, Ty, TyKind},
        CustomType, Expr, ExprKind, Op2,
    },
    syntax::is_type,
    utils::to_u32,
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

    /// This is needed to pass around the const value (values not represented by variables)
    /// that is not stored in the global constants hashmap.
    /// For example:
    /// let xx = [1; 10]; // where 10 is a const value that should be passed around in the type checker
    value: Option<String>,
}

impl ExprTyInfo {
    fn new(var_name: Option<String>, typ: TyKind) -> Self {
        Self {
            var_name,
            typ,
            value: None,
        }
    }

    fn new_var(var_name: String, typ: TyKind) -> Self {
        Self {
            var_name: Some(var_name),
            typ,
            value: None,
        }
    }

    fn new_anon(typ: TyKind) -> Self {
        Self {
            var_name: None,
            typ,
            value: None,
        }
    }

    fn new_anon_with_value(typ: TyKind, value: String) -> Self {
        Self {
            var_name: None,
            typ,
            value: Some(value),
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

                if lhs_node.typ != rhs_node.typ {
                    // only allow bigint mixed with field
                    match (&lhs_node.typ, &rhs_node.typ) {
                        (TyKind::BigInt, TyKind::Field) | (TyKind::Field, TyKind::BigInt) => (),
                        _ => {
                            return Err(self.error(
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

            ExprKind::BigInt(v) => Some(ExprTyInfo::new_anon_with_value(
                TyKind::BigInt,
                v.to_string(),
            )),

            ExprKind::Bool(_) => Some(ExprTyInfo::new_anon(TyKind::Bool)),

            // mod::path.of.var
            ExprKind::Variable { module, name } => {
                let qualified = FullyQualified::new(module, &name.value);

                if is_type(&name.value) {
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
                if !matches!(typ.typ, TyKind::Array(..)) {
                    return Err(self.error(ErrorKind::ArrayAccessOnNonArray, expr.span));
                }

                // check that expression is a bigint
                let idx_typ = self.compute_type(idx, typed_fn_env)?;
                match idx_typ.map(|t| t.typ) {
                    Some(TyKind::BigInt) => (),
                    _ => return Err(self.error(ErrorKind::ExpectedConstant, expr.span)),
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

                let res =
                    ExprTyInfo::new_anon(TyKind::Array(Box::new(tykind), ArraySize::Number(len)));
                Some(res)
            }

            ExprKind::DefaultArrayDeclaration { item, size } => {
                let item_typ = self
                    .compute_type(item, typed_fn_env)?
                    .expect("expected a value");

                let size_typ = self.compute_type(size, typed_fn_env)?;
                let size_typ = size_typ.unwrap();
                let res = match size_typ.var_name {
                    // anonymous constant
                    None => match size_typ.typ {
                        TyKind::BigInt => {
                            let value = size_typ.value;
                            if value.is_none() {
                                return Err(self.error(ErrorKind::ExpectedConstant, expr.span));
                            }

                            let size = value.unwrap().parse::<u32>().expect("expected a number");

                            ExprTyInfo::new_anon(TyKind::Array(
                                Box::new(item_typ.typ),
                                ArraySize::Number(size),
                            ))
                        }
                        _ => panic!("expected a size num"),
                    },
                    // const variable
                    Some(name) => {
                        let type_info = typed_fn_env.get_type_info(&name);

                        let qualified = FullyQualified::new(&ModulePath::Local, &name);
                        let cst = self.constants.get(&qualified);

                        // local var type is not constant
                        if let Some(type_info) = type_info {
                            if !type_info.constant {
                                return Err(
                                    self.error(ErrorKind::InvalidDefaultArraySize, expr.span)
                                );
                            }
                        } else {
                            // no local or global constant found for the variable name
                            if cst.is_none() {
                                return Err(
                                    self.error(ErrorKind::InvalidDefaultArraySize, expr.span)
                                );
                            }
                        }

                        // determine the size of array
                        match cst {
                            Some(cst) => {
                                let cst = to_u32(cst.value[0]);
                                ExprTyInfo::new_anon(TyKind::Array(
                                    Box::new(item_typ.typ),
                                    ArraySize::Number(cst),
                                ))
                            }
                            // todo: if not constant found, should it throw? but what if it is const arg?
                            // can't determine from local scope
                            _ => ExprTyInfo::new_anon(TyKind::Array(
                                Box::new(item_typ.typ),
                                ArraySize::ConstVar(name),
                            )),
                        }
                    }
                };

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

                // ensure start..end makes sense
                match (&range.start, &range.end) {
                    (NumOrVar::Number(start), NumOrVar::Number(end)) => {
                        if end < start {
                            panic!("end can't be smaller than start (TODO: better error)");
                        }
                    }
                    (NumOrVar::Variable(var), NumOrVar::Number(_))
                    | (NumOrVar::Number(_), NumOrVar::Variable(var)) => {
                        if !self.is_constant(var, typed_fn_env) {
                            return Err(self.error(ErrorKind::InvalidRangeType, range.span));
                        }
                    }
                    (NumOrVar::Variable(start), NumOrVar::Variable(end)) => {
                        if !self.is_constant(start, typed_fn_env)
                            || !self.is_constant(end, typed_fn_env)
                        {
                            return Err(self.error(ErrorKind::InvalidRangeType, range.span));
                        }
                    }
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

    fn is_constant(&mut self, var: &String, typed_fn_env: &mut TypedFnEnv) -> bool {
        // check if it is a constant from local scope
        let qualified = FullyQualified::new(&ModulePath::Local, var);
        if self.constants.get(&qualified).is_some() {
            return true;
        }

        // check if it is constant from type info
        let typ = typed_fn_env.get_type_info(var);
        if let Some(typ) = typ {
            return typ.constant;
        }

        false
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
            // todo: should check for attribute const: so if the expected arg is const, the observed should be const also?
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

mod test {
    use crate::{
        backends::kimchi::KimchiVesta,
        compiler::{typecheck_next_file_inner, Sources},
        error::{Error, ErrorKind, Result},
        type_checker::TypeChecker,
    };

    fn typecheck_code(code: &str) -> Result<usize> {
        let mut sources = Sources::new();
        let mut tast = TypeChecker::<KimchiVesta>::new();
        let this_module = None;
        typecheck_next_file_inner(
            &mut tast,
            this_module,
            &mut sources,
            "".to_string(),
            code.to_string(),
            0,
        )
    }

    #[test]
    fn test_invalid_size_var() {
        const CODE: &str = r#"
        fn const_generic() -> Field {
            let num = 3;
            let xx = [0; num];
            return xx[num - 1];
        }
        "#;

        let result = typecheck_code(CODE);

        assert!(result.is_err(), "expected error");
        match result.unwrap_err() {
            Error {
                kind: ErrorKind::InvalidDefaultArraySize,
                ..
            } => (),
            err => panic!("unexpected error: {:?}", err),
        }
    }

    #[test]
    fn test_miss_const_attr() {
        const CODE: &str = r#"
        fn const_generic(cst: Field, yy: Field) -> [Field; cst] {
            let xx = [1; cst];
            return xx;
        }
        "#;

        let result = typecheck_code(CODE);

        assert!(result.is_err(), "expected error");
        match result.unwrap_err() {
            Error {
                kind: ErrorKind::InvalidDefaultArraySize,
                ..
            } => (),
            err => panic!("unexpected error: {:?}", err),
        }
    }

    #[test]
    fn test_invalid_start_range_var() {
        const CODE: &str = r#"
        fn const_generic(const cst: Field, yy: Field) -> [Field; cst] {
            let mut arr = [0; cst];
            let start = 1;
            for ii in start..cst {
                arr[ii] = 2;
            }
            return arr;
        }
        "#;

        let result = typecheck_code(CODE);

        assert!(result.is_err(), "expected error");
        match result.unwrap_err() {
            Error {
                kind: ErrorKind::InvalidRangeType,
                ..
            } => (),
            err => panic!("unexpected error: {:?}", err),
        }
    }

    #[test]
    fn test_invalid_end_range_var() {
        const CODE: &str = r#"
        const start = 1;
        fn const_generic(const cst: Field, yy: Field) -> [Field; cst] {
            let mut arr = [0; cst];
            let end = 3;
            for ii in start..end {
                arr[ii] = 2;
            }
            return arr;
        }
        "#;

        let result = typecheck_code(CODE);

        assert!(result.is_err(), "expected error");
        match result.unwrap_err() {
            Error {
                kind: ErrorKind::InvalidRangeType,
                ..
            } => (),
            err => panic!("unexpected error: {:?}", err),
        }
    }

    #[test]
    fn test_const_attr() {
        const CODE: &str = r#"
        fn const_generic(const cst: Field, yy: Field) -> [Field; cst] {
            let xx = [1; cst];
            return xx;
        }
        "#;

        let result = typecheck_code(CODE);
        assert!(
            result.is_ok(),
            "unexpected error: {:?}",
            result.unwrap_err()
        );
    }

    #[test]
    fn test_global_const() {
        const CODE: &str = r#"
        const cst = 3;
        fn const_generic(yy: Field) -> [Field; cst] {
            let xx = [1; cst];
            return xx;
        }
        "#;

        let result = typecheck_code(CODE);
        assert!(
            result.is_ok(),
            "unexpected error: {:?}",
            result.unwrap_err()
        );
    }

    #[test]
    fn test_const_forloop() {
        const CODE: &str = r#"
        fn const_generic(const cst: Field) -> [Field; cst] {
            let mut xx = [1; cst];
            for ii in 1..cst {
                xx[ii] = 2;
            }
            return xx;
        }
        "#;

        let result = typecheck_code(CODE);
        assert!(
            result.is_ok(),
            "unexpected error: {:?}",
            result.unwrap_err()
        );
    }

    #[test]
    fn test_struct_array() {
        const CODE: &str = r#"
        const rooms = 3;
        const room_size = 20;

        struct Room {
            size: Field,
        }

        struct House {
            rooms: [Room; rooms],
        }

        fn build_house(const cst: Field) -> [House; cst] {
            let houses = [
                House {
                    rooms: [Room {size: room_size}; rooms]
                }; 
                cst
            ];
            return houses;
        }
        "#;

        let result = typecheck_code(CODE);
        assert!(
            result.is_ok(),
            "unexpected error: {:?}",
            result.unwrap_err()
        );
    }

    #[test]
    fn test_tracing_array_declaration() {
        const CODE: &str = r#"
        const room_num = 3;
        const house_num = 2;
        const room_size = 20;
        const slots = 10;

        fn build_houses() -> [[Field; room_num]; house_num] {
            let houses = [[1; room_num]; house_num];
            return houses;
        }

        fn test() -> Field {
            let hs = build_houses();
            let hh = hs[1];
            return hh[1];
        }
        "#;

        let result = typecheck_code(CODE);
        assert!(
            result.is_ok(),
            "unexpected error: {:?}",
            result.unwrap_err()
        );
    }
}
