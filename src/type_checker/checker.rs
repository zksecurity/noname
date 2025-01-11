use std::collections::HashMap;

use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};

use crate::{
    backends::Backend,
    cli::packages::UserRepo,
    constants::Span,
    error::{ErrorKind, Result},
    imports::FnKind,
    parser::{
        types::{
            is_numeric, FnSig, ForLoopArgument, FunctionDef, ModulePath, Stmt, StmtKind, Symbolic,
            Ty, TyKind,
        },
        CustomType, Expr, ExprKind, Op2,
    },
    stdlib::builtins::QUALIFIED_BUILTINS,
    syntax::is_type,
};

use super::{FullyQualified, TypeChecker, TypeInfo, TypedFnEnv};

/// Keeps track of the signature of a user-defined function.
#[derive(Debug, Clone, Serialize)]
pub struct FnInfo<B>
where
    B: Backend,
{
    #[serde(bound = "FnKind<B>: Serialize")]
    pub kind: FnKind<B>,
    // TODO: We will remove this once the native hint is supported
    // This field is to indicate if a builtin function should be treated as a hint.
    // instead of adding this flag to the FnKind::Builtin, we add this field to the FnInfo.
    // Then this flag will only present in the FunctionDef.
    pub is_hint: bool,
    pub span: Span,
}

impl<B: Backend> FnInfo<B> {
    pub fn sig(&self) -> &FnSig {
        match &self.kind {
            FnKind::BuiltIn(sig, _, _) => sig,
            FnKind::Native(func) => &func.sig,
        }
    }

    pub fn native(&self) -> Option<&FunctionDef> {
        match &self.kind {
            FnKind::Native(func) => Some(func),
            _ => None,
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
                    _ => return Err(self.error(ErrorKind::FieldAccessOnNonCustomStruct, expr.span)),
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
                    .map(|(_, typ)| typ.clone());

                if let Some(res) = res {
                    Some(ExprTyInfo::new(lhs_node.var_name, res))
                } else {
                    return Err(self.error(
                        ErrorKind::UndefinedField(struct_info.name.clone(), rhs.value.clone()),
                        expr.span,
                    ));
                }
            }

            // `module::fn_name(args)`
            ExprKind::FnCall {
                module,
                fn_name,
                args,
                unsafe_attr,
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

                // check if the function is a hint
                if fn_info.is_hint && !unsafe_attr {
                    return Err(self.error(ErrorKind::ExpectedUnsafeAttribute, expr.span));
                }

                // unsafe attribute should only be used on hints
                if !fn_info.is_hint && *unsafe_attr {
                    return Err(self.error(ErrorKind::UnexpectedUnsafeAttribute, expr.span));
                }

                // check if generic is allowed
                if fn_sig.require_monomorphization() && typed_fn_env.is_in_forloop() {
                    for (observed_arg, expected_arg) in args.iter().zip(fn_sig.arguments.iter()) {
                        // check if the arg involves generic vars
                        if !expected_arg.extract_generic_names().is_empty() {
                            let mut forbidden_env = typed_fn_env.clone();
                            forbidden_env.forbid_forloop_scope();

                            // rewalk the observed arg expression
                            // it should throw an error if the arg contains generic vars relating to the variables in the forloop scope
                            self.compute_type(observed_arg, &mut forbidden_env)?;

                            // release the forbidden flag
                            forbidden_env.allow_forloop_scope();
                        }
                    }
                }

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
                let method_type = struct_info.methods.get(&method_name.value);

                if method_type.is_none() {
                    return Err(self.error(
                        ErrorKind::UndefinedMethod(struct_name.clone(), method_name.value.clone()),
                        method_name.span,
                    ));
                }
                let method_type = method_type.unwrap();

                // check if generic is allowed
                if method_type.sig.require_monomorphization() && typed_fn_env.is_in_forloop() {
                    for (observed_arg, expected_arg) in args.iter().zip(
                        method_type
                            .sig
                            .arguments
                            .iter()
                            .filter(|arg| arg.name.value != "self"),
                    ) {
                        // check if the arg involves generic vars
                        if !expected_arg.extract_generic_names().is_empty() {
                            let mut forbidden_env = typed_fn_env.clone();
                            forbidden_env.forbid_forloop_scope();

                            // rewalk the observed arg expression
                            // it should throw an error if the arg contains generic vars relating to the variables in the forloop scope
                            self.compute_type(observed_arg, &mut forbidden_env)?;

                            // release the forbidden flag
                            forbidden_env.allow_forloop_scope();
                        }
                    }
                }

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

                // todo: check and update the const field type for other cases
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

                    // `array[idx] = <rhs>` or `tuple[idx] = rhs`
                    ExprKind::ArrayOrTupleAccess { container, idx } => {
                        // get variable behind container
                        let cotainer_node = self
                            .compute_type(container, typed_fn_env)?
                            .expect("type-checker bug: array or tuple access on an empty var");

                        cotainer_node
                            .var_name
                            .expect("anonymous array or tuple access cannot be mutated")
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
                    _ => Err(self.error(
                        ErrorKind::UnexpectedError("bad expression assignment"),
                        expr.span,
                    ))?,
                };

                // check that the var exists locally
                let lhs_info = typed_fn_env
                    .get_type_info(&lhs_name)?
                    .ok_or_else(|| {
                        self.error(ErrorKind::UnexpectedError("variable not found"), expr.span)
                    })?
                    .clone();

                // and is mutable
                if !lhs_info.mutable {
                    Err(self.error(ErrorKind::AssignmentToImmutableVariable, expr.span))?;
                }

                // and is of the same type as the rhs
                let rhs_typ = self.compute_type(rhs, typed_fn_env)?.unwrap();

                if !rhs_typ.typ.match_expected(&lhs_node.typ, false) {
                    return Err(self.error(
                        ErrorKind::MismatchType(lhs_node.typ.clone(), rhs_typ.typ.clone()),
                        expr.span,
                    ));
                }

                // update struct field type
                if let ExprKind::FieldAccess {
                    lhs,
                    rhs: field_name,
                } = &lhs.kind
                {
                    // get variable behind lhs
                    let lhs_node = self
                        .compute_type(lhs, typed_fn_env)?
                        .expect("type-checker bug: lhs access on an empty var");

                    // obtain the qualified name of the struct
                    let (module, struct_name) = match lhs_node.typ {
                        TyKind::Custom { module, name } => (module, name),
                        _ => {
                            return Err(
                                self.error(ErrorKind::FieldAccessOnNonCustomStruct, lhs.span)
                            )
                        }
                    };

                    let qualified = FullyQualified::new(&module, &struct_name);
                    self.update_struct_field(&qualified, &field_name.value, rhs_typ.typ);
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
                    && (!is_numeric(&lhs_node.typ) || !is_numeric(&rhs_node.typ))
                {
                    // only allow fields
                    match (&lhs_node.typ, &rhs_node.typ) {
                        (TyKind::Field { .. }, TyKind::Field { .. }) => (),
                        _ => Err(self.error(
                            ErrorKind::MismatchType(lhs_node.typ.clone(), rhs_node.typ.clone()),
                            expr.span,
                        ))?,
                    }
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
                if !matches!(inner_typ.typ, TyKind::Field { .. }) {
                    Err(self.error(
                        // it can be either constant or not.
                        // here we just default it to constant for error message.
                        ErrorKind::MismatchType(TyKind::Field { constant: true }, inner_typ.typ),
                        expr.span,
                    ))?
                } else {
                    Some(ExprTyInfo::new_anon(inner_typ.typ))
                }
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

            ExprKind::BigUInt(_) => Some(ExprTyInfo::new_anon(TyKind::Field { constant: true })),

            ExprKind::Bool(_) => Some(ExprTyInfo::new_anon(TyKind::Bool)),

            ExprKind::StringLiteral(s) => Some(ExprTyInfo::new_anon(TyKind::String(s.clone()))),

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
                        cst.typ.kind.clone()
                    } else {
                        // otherwise it's a local variable
                        // generic parameter is also checked as a local variable
                        typed_fn_env
                            .get_type(&name.value)?
                            .ok_or_else(|| self.error(ErrorKind::UndefinedVariable, name.span))?
                            .clone()
                    };

                    let res = ExprTyInfo::new_var(name.value.clone(), typ);
                    Some(res)
                }
            }

            ExprKind::ArrayOrTupleAccess { container, idx } => {
                // get type of lhs
                let typ = self.compute_type(container, typed_fn_env)?.unwrap();

                // check that it is an array or tuple
                if !matches!(
                    typ.typ,
                    TyKind::Array(..) | TyKind::GenericSizedArray(..) | TyKind::Tuple(..)
                ) {
                    Err(self.error(ErrorKind::AccessOnNonCollection, expr.span))?
                }

                // check that expression is a bigint
                let idx_typ = self.compute_type(idx, typed_fn_env)?;
                match idx_typ.map(|t| t.typ) {
                    Some(TyKind::Field { constant: true }) => (),
                    _ => return Err(self.error(ErrorKind::ExpectedConstant, expr.span)),
                };

                // get type of element
                let el_typ = match typ.typ {
                    TyKind::Array(typkind, _) => *typkind,
                    TyKind::GenericSizedArray(typkind, _) => *typkind,
                    TyKind::Tuple(typs) => match &idx.kind {
                        ExprKind::BigUInt(index) => typs[index.to_usize().unwrap()].clone(),
                        _ => return Err(self.error(ErrorKind::ExpectedConstant, expr.span)),
                    },
                    _ => Err(self.error(ErrorKind::UnexpectedError("not an array"), expr.span))?,
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
                        if !tykind.match_expected(&item_typ.typ, false) {
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
            ExprKind::TupleDeclaration(items) => {
                // restricting tupple len as array len
                let _: u32 = items.len().try_into().expect("tupple too large");
                let typs: Vec<TyKind> = items
                    .iter()
                    .map(|item| {
                        self.compute_type(item, typed_fn_env)
                            .unwrap()
                            .expect("expected some val")
                            .typ
                    })
                    .collect();
                Some(ExprTyInfo::new_anon(TyKind::Tuple(typs)))
            }

            ExprKind::IfElse { cond, then_, else_ } => {
                // cond can only be a boolean
                let cond_node = self
                    .compute_type(cond, typed_fn_env)?
                    .expect("can't compute type of condition");
                if !matches!(cond_node.typ, TyKind::Bool) {
                    Err(self.error(
                        ErrorKind::IfElseInvalidConditionType(cond_node.typ.clone()),
                        cond.span,
                    ))?
                }

                // then_ and else_ can only be variables, field accesses, or array accesses
                if !matches!(
                    &then_.kind,
                    ExprKind::Variable { .. }
                        | ExprKind::FieldAccess { .. }
                        | ExprKind::ArrayOrTupleAccess { .. }
                ) {
                    return Err(self.error(ErrorKind::IfElseInvalidIfBranch(), then_.span));
                }

                if !matches!(
                    &else_.kind,
                    ExprKind::Variable { .. }
                        | ExprKind::FieldAccess { .. }
                        | ExprKind::ArrayOrTupleAccess { .. }
                ) {
                    return Err(self.error(ErrorKind::IfElseInvalidElseBranch(), else_.span));
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
                    return Err(self.error(ErrorKind::IfElseMismatchingBranchesTypes(), expr.span));
                }

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

                    if !observed_typ.typ.match_expected(&defined.1, false) {
                        return Err(self.error(
                            ErrorKind::InvalidStructFieldType(defined.1.clone(), observed_typ.typ),
                            expr.span,
                        ));
                    }

                    // If the observed type is a Field type, then init that struct field as the observed type.
                    // This is because the field type can be a constant or not, which needs to be propagated.
                    if matches!(observed_typ.typ, TyKind::Field { .. }) {
                        self.update_struct_field(&qualified, &defined.0, observed_typ.typ.clone());
                    }
                }

                let res = ExprTyInfo::new_anon(TyKind::Custom {
                    module: module.clone(),
                    name: name.clone(),
                });
                Some(res)
            }
            ExprKind::RepeatedArrayInit { item, size } => {
                let item_node = self
                    .compute_type(item, typed_fn_env)?
                    .expect("expected a typed item");

                let size_node = self
                    .compute_type(size, typed_fn_env)?
                    .expect("expected a typed size");

                if is_numeric(&size_node.typ) {
                    let sym = Symbolic::parse(size)?;
                    let res = if let Symbolic::Concrete(size) = sym {
                        // if sym is a concrete variant, then just return concrete array type
                        ExprTyInfo::new_anon(TyKind::Array(
                            Box::new(item_node.typ),
                            size.to_u32().expect("array size too large"),
                        ))
                    } else {
                        // use generic array as the size node might include generic parameters or constant vars
                        ExprTyInfo::new_anon(TyKind::GenericSizedArray(
                            Box::new(item_node.typ),
                            sym,
                        ))
                    };

                    Some(res)
                } else {
                    Err(self.error(ErrorKind::InvalidArraySize, expr.span))?
                }
            }
        };

        // save the type of that expression in our typed global env
        if let Some(typ) = &typ {
            self.node_types.insert(expr.node_id, typ.typ.clone());
        }

        // update last node id
        // todo: probably better to pass the node id from nast
        if self.node_id < expr.node_id {
            self.node_id = expr.node_id;
        }

        // return the type to the caller
        Ok(typ)
    }

    pub fn check_block(
        &mut self,
        typed_fn_env: &mut TypedFnEnv,
        stmts: &[Stmt],
        expected_return: Option<&Ty>,
        new_scope: bool,
    ) -> Result<()> {
        // enter the scope
        if new_scope {
            typed_fn_env.nest();
        }

        let mut return_typ = None;

        for stmt in stmts {
            if return_typ.is_some() {
                Err(self.error(
                    ErrorKind::UnexpectedError(
                        "early return detected: we don't allow that for now",
                    ),
                    stmt.span,
                ))?
            }

            return_typ = self.check_stmt(typed_fn_env, stmt)?;
        }

        // check the return
        match (expected_return, return_typ) {
            (None, None) => (),
            (Some(expected), None) => Err(self.error(ErrorKind::MissingReturn, expected.span))?,
            (None, Some(_)) => {
                Err(self.error(ErrorKind::NoReturnExpected, stmts.last().unwrap().span))?
            }
            (Some(expected), Some(observed)) => {
                if !observed.match_expected(&expected.kind, false) {
                    Err(self.error(
                        ErrorKind::ReturnTypeMismatch(expected.kind.clone(), observed.clone()),
                        expected.span,
                    ))?
                }
            }
        };

        // exit the scope
        if new_scope {
            typed_fn_env.pop();
        }

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
            StmtKind::ForLoop {
                var,
                argument,
                body,
            } => {
                // enter a new scope
                typed_fn_env.nest();

                match argument {
                    ForLoopArgument::Range(range) => {
                        // create var (for now it's always a constant)
                        typed_fn_env.store_type(
                            var.value.clone(),
                            TypeInfo::new(TyKind::Field { constant: true }, var.span),
                        )?;

                        let start_type = self.compute_type(&range.start, typed_fn_env)?.unwrap();
                        let end_type = self.compute_type(&range.end, typed_fn_env)?.unwrap();
                        if !is_numeric(&start_type.typ) || !is_numeric(&end_type.typ) {
                            Err(self.error(ErrorKind::InvalidRangeSize, range.span))?;
                        }
                    }
                    ForLoopArgument::Iterator(iterator) => {
                        // make sure that the iterator expression is an iterator,
                        // for now this means that the expression should have type `Array`
                        let iterator_typ = self.compute_type(iterator, typed_fn_env)?;
                        let iterator_typ = iterator_typ.ok_or_else(|| {
                            self.error(
                                ErrorKind::UnexpectedError("Could not compute type of iterator"),
                                iterator.span,
                            )
                        })?;

                        // the type of the variable is the type of the items of the iterator
                        let element_type = match iterator_typ.typ {
                            TyKind::Array(element_type, _len) => *element_type,
                            TyKind::GenericSizedArray(element_type, _size) => *element_type,
                            _ => Err(self.error(
                                ErrorKind::InvalidIteratorType(iterator_typ.typ.clone()),
                                iterator.span,
                            ))?,
                        };

                        typed_fn_env
                            .store_type(var.value.clone(), TypeInfo::new(element_type, var.span))?;
                    }
                }

                typed_fn_env.start_forloop();

                // check block
                self.check_block(typed_fn_env, body, None, false)?;

                // exit the scope
                typed_fn_env.pop();
                typed_fn_env.end_forloop();
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
        match typed_fn_env.get_type_info(&fn_sig.name.value)? {
            Some(_) => {
                return Err(self.error(
                    ErrorKind::FunctionNameInUsebyVariable(fn_sig.name.value),
                    fn_sig.name.span,
                ))
            }
            None => (),
        };

        // get the ignore_arg_types flag from the function info if it's a builtin
        let ignore_arg_types = match self
            .fn_info(&FullyQualified::new(
                &ModulePath::Absolute(UserRepo::new(QUALIFIED_BUILTINS)),
                &fn_sig.name.value,
            ))
            .map(|info| &info.kind)
        {
            // check builtin
            Some(FnKind::BuiltIn(_, _, ignore)) => *ignore,
            _ => false,
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
            if !ignore_arg_types {
                return Err(self.error(
                    ErrorKind::MismatchFunctionArguments(observed.len(), expected.len()),
                    span,
                ));
            }
        }

        // skip argument type checking if ignore_arg_types is true
        if !ignore_arg_types {
            // compare argument types with the function signature
            for (sig_arg, (typ, span)) in expected.iter().zip(observed) {
                // when const attribute presented, the argument must be a constant
                if sig_arg.is_constant() && !matches!(typ, TyKind::Field { constant: true }) {
                    return Err(self.error(
                        ErrorKind::ArgumentTypeMismatch(sig_arg.typ.kind.clone(), typ),
                        span,
                    ));
                }

                if !typ.match_expected(&sig_arg.typ.kind, false) {
                    return Err(self.error(
                        ErrorKind::ArgumentTypeMismatch(sig_arg.typ.kind.clone(), typ),
                        span,
                    ));
                }
            }
        }

        // return the return type of the function
        Ok(fn_sig.return_type.as_ref().map(|ty| ty.kind.clone()))
    }
}
