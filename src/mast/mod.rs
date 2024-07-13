use std::collections::HashMap;

use crate::{
    backends::Backend, constants::Span, error::{Error, ErrorKind, Result}, imports::FnKind, name_resolution::NAST, parser::{
        types::{FnSig, FuncOrMethod, Stmt, StmtKind, Ty, TyKind}, CustomType, Expr, ExprKind, Op2, RootKind
    }, syntax::is_type, type_checker::{checker::ExprTyInfo, FnInfo, FullyQualified, TypeChecker, TypeInfo, TypedFnEnv}
};

/// Monomorphized AST
pub struct Mast<B>
where
    B: Backend,
{
    tast: TypeChecker<B>,

    /// Mapping from node id to monomorphized type
    node_types: HashMap<usize, TyKind>,
}

// TypedFnEnv
// records generic parameters
// records types including inferred types

impl<B: Backend> Mast<B> {
    pub fn monomorphize(&mut self, nast: NAST<B>) -> Result<()> {
        // process the main function
        // process fn calls
        // infer generic values from function args
        // apply inferred values to function body and signature
        // type check the observed return and the inferred expected return

        // store mtype in mast

        for root in &nast.ast.0 {
            match &root.kind {
                // `fn main() { ... }`
                RootKind::FunctionDef(function) => {
                    // create a new typed fn environment to type check the function
                    let mut typed_fn_env = TypedFnEnv::default();

                    // if we're expecting a library, this should not be the main function
                    let is_main = function.is_main();
                    if !is_main {
                        continue;
                    }

                    // store variables and their types in the fn_env
                    for arg in &function.sig.arguments {
                        // store the args' type in the fn environment
                        let arg_typ = arg.typ.kind.clone();

                        if arg.is_constant() {
                            typed_fn_env.store_type(
                                arg.name.value.clone(),
                                TypeInfo::new_cst(arg_typ, arg.span),
                            )?;
                        } else if function.sig.is_generic(arg.clone()) {
                            // then assume it is a const generic
                            typed_fn_env.store_type(
                                arg.name.value.clone(),
                                TypeInfo::new(TyKind::BigInt, arg.span),
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
                        match typ.kind {
                            TyKind::Field => {
                                typed_fn_env.store_type(
                                    "public_output".to_string(),
                                    TypeInfo::new_mut(typ.kind.clone(), typ.span),
                                )?;
                            }
                            TyKind::Array(_, _) => {
                                typed_fn_env.store_type(
                                    "public_output".to_string(),
                                    TypeInfo::new_mut(typ.kind.clone(), typ.span),
                                )?;
                            }
                            _ => unimplemented!(),
                        }
                    }

                    // type system pass on the function body
                    self.check_block(
                        &mut typed_fn_env,
                        &function.body,
                        function.sig.return_type.as_ref(),
                    )?;
                }

                _ => (),
            };
        }

        Ok(())
    }

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
                    .tast
                    .struct_info(&qualified)
                    .expect("this struct is not defined, or you're trying to access a field of a struct defined in a third-party library");

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
                let fn_info = self.tast.fn_info(&qualified).unwrap();
                let fn_sig = fn_info.sig().clone();

                // type check the function call
                let method_call = false;
                // todo: may need to call fn_env.nest()
                let res = self.check_fn_call(typed_fn_env, method_call, fn_sig, args, expr.span)?;
                // todo: may need to call fn_env.pop()

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
                let struct_info = self.tast
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
                        // todo: remove this check, as it is already done in the TAST
                        // we first check if it's a constant
                        // note: the only way to check that atm is to check in the constants hashmap
                        // this is because we don't differentiate const vars from normal variables
                        // (perhaps we should)
                        let qualified = FullyQualified::new(&module, &name.value);
                        if let Some(_cst_info) = self.tast.const_info(&qualified) {
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

                // todo: remove this check
                // and is mutable
                if !lhs_info.mutable {
                    return Err(self.error(ErrorKind::AssignmentToImmutableVariable, expr.span));
                }

                // and is of the same type as the rhs
                let rhs_typ = self.compute_type(rhs, typed_fn_env)?.unwrap();

                // todo: use exact match
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

                // todo: remove this check
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
                // todo: remove this check
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
                // todo: remove this check
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

                // todo: what is the case a Type can ba variable?
                // todo: if it is not a local variable, can we just load the type from the tast.node_types?
                if is_type(&name.value) {
                    // if it's a type, make sure it exists
                    let _struct_info = self
                        .tast
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
                    let typ = if let Some(cst) = self.tast.constants.get(&qualified) {
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
                // todo: remove this check
                if !matches!(typ.typ, TyKind::Array(..)) {
                    return Err(self.error(ErrorKind::ArrayAccessOnNonArray, expr.span));
                }

                // check that expression is a bigint
                let idx_typ = self.compute_type(idx, typed_fn_env)?;
                // todo: remove this check
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
                let struct_info = self.tast.struct_info(&qualified).ok_or_else(|| {
                    self.error(ErrorKind::UndefinedStruct(name.clone()), expr.span)
                })?;

                let defined_fields = &struct_info.fields.clone();

                // todo: remove this check
                if defined_fields.len() != fields.len() {
                    return Err(
                        self.error(ErrorKind::MismatchStructFields(name.clone()), expr.span)
                    );
                }

                // todo: infer for generic type
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
                    .expect("expected a value (TODO: better error)");

                // expect the size node to be a u32
                let size_node = self
                    .compute_type(size, typed_fn_env)?
                    .expect("expected a value (TODO: better error)");

                // if !matches!(size_node.typ, TyKind::BigInt) {
                //     return Err(self.error(ErrorKind::InvalidArraySize, expr.span));
                // }
                println!("size_node.typ: {:#?}", size_node.typ);
                // todo: infer for generic type
                let res = ExprTyInfo::new_anon(TyKind::GenericArray(
                    Box::new(item_node.typ),
                    Symbolic::Generic(Ident::new("x".to_string(), size.span)),
                ));
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
                if range.end < range.start {
                    panic!("end can't be smaller than start (TODO: better error)");
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
        // todo: remove this check
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
            // todo: infer generic values from the observed arg
            // generic array
            // generic const
            // match sig_arg.typ.kind {
            //     TyKind::GenericArray(, )
            //     TyKind::GenericConst(())
            // }

            // store the inferred value in fn_env

            // should it just type check inferred expected type?
            println!("sig_arg: {:#?}", sig_arg);
            println!("observed typ: {:#?}", typ);
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

    pub fn error(&self, kind: ErrorKind, span: Span) -> Error {
        Error::new("mast", kind, span)
    }
}
