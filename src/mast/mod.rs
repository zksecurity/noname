use std::collections::HashMap;

use ark_ff::PrimeField;
use num_bigint::BigUint;

use crate::{
    backends::Backend,
    circuit_writer::fn_env,
    constants::Span,
    error::{Error, ErrorKind, Result},
    imports::FnKind,
    name_resolution::NAST,
    parser::{
        types::{FnSig, FuncOrMethod, Ident, Stmt, StmtKind, Symbolic, Ty, TyKind},
        CustomType, Expr, ExprKind, Op2, RootKind,
    },
    syntax::is_type,
    type_checker::{FnInfo, FullyQualified, TypeChecker, TypedFnEnv},
};

#[derive(Debug)]
pub struct ExprTyMInfo {
    /// This is needed to obtain information on variables.
    /// For example, parsing of an assignment expression needs to know if the lhs variable is mutable.
    pub var_name: Option<String>,

    /// The type of the expression node.
    pub typ: TyKind,

    /// numeric value of the expression
    /// applicable to BigInt type
    pub constant: Option<u32>,
}

impl ExprTyMInfo {
    pub fn new(var_name: Option<String>, typ: TyKind, value: Option<u32>) -> Self {
        if value.is_some() && !matches!(typ, TyKind::BigInt) {
            panic!("value can only be set for BigInt type");
        }

        Self {
            var_name,
            typ,
            constant: value,
        }
    }

    pub fn new_var(var_name: String, typ: TyKind, value: Option<u32>) -> Self {
        Self::new(Some(var_name), typ, value)
    }

    pub fn new_anon(typ: TyKind, value: Option<u32>) -> Self {
        Self::new(None, typ, value)
    }
}

#[derive(Debug, Clone)]
pub struct MTypeInfo {
    /// Some type information.
    pub typ: TyKind,

    /// Store constant value
    pub value: Option<u32>,

    /// The span of the variable declaration.
    pub span: Span,
}

impl MTypeInfo {
    pub fn new(typ: TyKind, span: Span, value: Option<u32>) -> Self {
        Self {
            typ,
            span,
            value,
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct MonomorphizedFnEnv {
    current_scope: usize,

    vars: HashMap<String, (usize, MTypeInfo)>,
}

impl MonomorphizedFnEnv {
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
        self.current_scope = self.current_scope.checked_sub(1).expect("scope bug");

        // remove variables as we exit the scope
        // (we don't need to keep them around to detect shadowing,
        // as we already did that in type checker)
        let mut to_delete = vec![];
        for (name, (scope, _)) in self.vars.iter() {
            if *scope > self.current_scope {
                to_delete.push(name.clone());
            }
        }
        for d in to_delete {
            self.vars.remove(&d);
        }
    }

    /// Returns true if a scope is a prefix of our scope.
    pub fn is_in_scope(&self, prefix_scope: usize) -> bool {
        self.current_scope >= prefix_scope
    }

    /// Stores type information about a local variable.
    /// Note that we forbid shadowing at all scopes.
    pub fn store_type(&mut self, ident: String, type_info: MTypeInfo) -> Result<()> {
        match self
            .vars
            .insert(ident.clone(), (self.current_scope, type_info.clone()))
        {
            Some(_) => Err(Error::new(
                "type-checker",
                ErrorKind::DuplicateDefinition(ident),
                type_info.span,
            )),
            None => Ok(()),
        }
    }

    pub fn reassign_type(&mut self, ident: String, type_info: MTypeInfo) -> Result<()> {
        self
            .vars
            .insert(ident.clone(), (self.current_scope, type_info.clone()));

        Ok(())
    }

    pub fn get_type(&self, ident: &str) -> Option<&TyKind> {
        self.get_type_info(ident).map(|type_info| &type_info.typ)
    }

    /// Retrieves type information on a variable, given a name.
    /// If the variable is not in scope, return false.
    pub fn get_type_info(&self, ident: &str) -> Option<&MTypeInfo> {
        if let Some((scope, type_info)) = self.vars.get(ident) {
            if self.is_in_scope(*scope) {
                Some(type_info)
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[derive(Debug)]
/// Monomorphized AST
pub struct Mast<B>
where
    B: Backend,
{
    pub tast: TypeChecker<B>,

    /// Mapping from node id to monomorphized type
    node_types: HashMap<usize, TyKind>,
}

// TypedFnEnv
// records generic parameters
// records types including inferred types

impl<B: Backend> Mast<B> {
    pub fn new(tast: TypeChecker<B>) -> Self {
        Self {
            tast,
            node_types: HashMap::new(),
        }
    }

    pub(crate) fn expr_type(&self, expr: &Expr) -> Option<&TyKind> {
        self.node_types.get(&expr.node_id)
    }

    pub fn monomorphize(&mut self) -> Result<()> {
        // process the main function
        // process fn calls
        // infer generic values from function args
        // apply inferred values to function body and signature
        // type check the observed return and the inferred expected return

        // store mtype in mast
        let qualified = FullyQualified::local("main".to_string());
        let main_fn = self.tast
            .fn_info(&qualified)
            .ok_or(self.error(ErrorKind::NoMainFunction, Span::default()))?;

        let func_def =match &main_fn.kind {
            // `fn main() { ... }`
            FnKind::Native(function) => {
                function.clone()
            }

            _ => panic!("main function must be native"),
        };

        // create a new typed fn environment to type check the function
        let mut typed_fn_env = MonomorphizedFnEnv::default();

        // store variables and their types in the fn_env
        for arg in &func_def.sig.arguments {
            // store the args' type in the fn environment
            let arg_typ = arg.typ.kind.clone();

            typed_fn_env.store_type(
                arg.name.value.clone(),
                MTypeInfo::new(arg_typ, arg.span, None),
            )?;
        }

        // the output value returned by the main function is also a main_args with a special name (public_output)
        if let Some(typ) = &func_def.sig.return_type {
            match typ.kind {
                TyKind::Field => {
                    typed_fn_env.store_type(
                        "public_output".to_string(),
                        MTypeInfo::new(typ.kind.clone(), typ.span, None),
                    )?;
                }
                TyKind::Array(_, _) => {
                    typed_fn_env.store_type(
                        "public_output".to_string(),
                        MTypeInfo::new(typ.kind.clone(), typ.span, None),
                    )?;
                }
                _ => unimplemented!(),
            }
        }

        typed_fn_env.nest();
        // type system pass on the function body
        self.check_block(
            &mut typed_fn_env,
            &func_def.body,
            func_def.sig.return_type.as_ref(),
        )?;
        typed_fn_env.pop();

        Ok(())
    }

    fn compute_type(
        &mut self,
        expr: &Expr,
        typed_fn_env: &mut MonomorphizedFnEnv,
    ) -> Result<Option<ExprTyMInfo>> {
        let typ: Option<ExprTyMInfo> = match &expr.kind {
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

                // struct field doesn't support constant
                let cst = None;
                Some(ExprTyMInfo::new(lhs_node.var_name, res, cst))
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

                // type check the function call
                let method_call = false;
                let res = self.check_fn_call(typed_fn_env, method_call, fn_info.clone(), args, expr.span)?;

                // assume the function call won't return constant value
                res.map(|ty| ExprTyMInfo::new_anon(ty, None))
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
                    .tast
                    .struct_info(&qualified)
                    .ok_or(self.error(ErrorKind::UndefinedStruct(struct_name.clone()), lhs.span))?
                    .clone();

                // get method info
                let method_type = struct_info
                    .methods
                    .get(&method_name.value)
                    .expect("method not found on custom struct (TODO: better error)");

                let fn_kind = FnKind::Native(method_type.clone());
                let fn_info = FnInfo {
                    kind: fn_kind,
                    span: method_type.span,
                };

                // type check the method call
                let method_call = true;

                // store lhs type as the self arg
                typed_fn_env.store_type("self".to_string(), lhs_type.unwrap());

                // typed_fn_env.nest();
                let res = self.check_fn_call(
                    typed_fn_env,
                    method_call,
                    fn_info,
                    args,
                    expr.span,
                )?;
                // typed_fn_env.pop();

                // assume the function call won't return constant value
                res.map(|ty| ExprTyMInfo::new_anon(ty, None))
            }

            ExprKind::Assignment { lhs, rhs } => {
                // compute type of lhs
                let lhs_node = self
                    .compute_type(lhs, typed_fn_env)?
                    .expect("type-checker bug: lhs access on an empty var");

                // and is of the same type as the rhs
                let rhs_typ = self.compute_type(rhs, typed_fn_env)?.unwrap();

                // todo: use exact match
                if !rhs_typ.typ.same_as(&lhs_node.typ) {
                    panic!("lhs type doesn't match rhs type");
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

                let cst = match (lhs_node.constant, rhs_node.constant) {
                    (Some(lhs), Some(rhs)) => {
                        let res = match op {
                            Op2::Addition => Some(lhs + rhs),
                            Op2::Subtraction => Some(lhs - rhs),
                            Op2::Multiplication => Some(lhs * rhs),
                            Op2::Division => Some(lhs / rhs),
                            _ => None,
                        };
                        res
                    }
                    _ => None,
                };

                Some(ExprTyMInfo::new_anon(typ, cst))
            }

            ExprKind::Negated(inner) => {
                // todo: can constant be negative?
                // let inner_typ = self.compute_type(inner, typed_fn_env)?.unwrap();
                // let cst = inner_typ.constant.map(|c| -c);

                Some(ExprTyMInfo::new_anon(TyKind::Field, None))
            }

            ExprKind::Not(_) => Some(ExprTyMInfo::new_anon(TyKind::Bool, None)),

            ExprKind::BigUInt(inner) => {
                let cst: u32 = inner.try_into().expect("biguint too large");
                Some(ExprTyMInfo::new_anon(TyKind::BigInt, Some(cst)))
            }

            ExprKind::Bool(_) => Some(ExprTyMInfo::new_anon(TyKind::Bool, None)),

            // mod::path.of.var
            ExprKind::Variable { module, name } => {
                let qualified = FullyQualified::new(module, &name.value);

                // todo: what is the case a Type can ba variable?
                // todo: if it is not a local variable, can we just load the type from the tast.node_types?
                if is_type(&name.value) {
                    // and return its type
                    let res = ExprTyMInfo::new_anon(
                        TyKind::Custom {
                            module: module.clone(),
                            name: name.value.clone(),
                        },
                        None,
                    );
                    Some(res)
                } else {
                    // if it's a variable,
                    // check if it's a constant first
                    let res = if let Some(cst) = self.tast.constants.get(&qualified) {
                        let bigint: BigUint = cst.value[0].into();
                        let cst: u32 = bigint.try_into().expect("biguint too large");
                        ExprTyMInfo::new_var(name.value.clone(), TyKind::BigInt, Some(cst))
                    } else {
                        // otherwise it's a local variable
                        let mtype = typed_fn_env.get_type_info(&name.value).unwrap().clone();
                        ExprTyMInfo::new_var(name.value.clone(), mtype.typ, mtype.value)
                    };

                    Some(res)
                }
            }

            ExprKind::ArrayAccess { array, idx } => {
                // get type of lhs
                let typ = self.compute_type(array, typed_fn_env)?.unwrap();

                // get type of element
                let el_typ = match typ.typ {
                    TyKind::Array(typkind, _) => *typkind,
                    _ => panic!("not an array"),
                };

                // todo: check the bounds of the array

                let res = ExprTyMInfo::new(typ.var_name, el_typ, None);
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

                let res = self
                    .compute_type(&items[0], typed_fn_env)?
                    .expect("expected a value");

                let mty = ExprTyMInfo::new_anon(TyKind::Array(Box::new(res.typ), len), None);

                Some(mty)
            }

            ExprKind::IfElse { cond, then_, else_ } => {
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

                Some(then_node)
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
                        .expect("expected a value");

                    // todo: use exact match
                    if !observed_typ.typ.same_as(&defined.1) {
                        return Err(self.error(
                            ErrorKind::InvalidStructFieldType(defined.1.clone(), observed_typ.typ),
                            expr.span,
                        ));
                    }
                }

                let res = ExprTyMInfo::new_anon(
                    TyKind::Custom {
                        module: module.clone(),
                        name: name.clone(),
                    },
                    None,
                );
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

                if let Some(cst) = size_node.constant {
                    let res =
                        ExprTyMInfo::new_anon(TyKind::Array(Box::new(item_node.typ), cst), None);
                    Some(res)
                } else {
                    // todo: better error indicating that the size must be resolved
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
        typed_fn_env: &mut MonomorphizedFnEnv,
        stmts: &[Stmt],
        expected_return: Option<&Ty>,
    ) -> Result<Option<TyKind>> {
        let mut return_typ = None;

        for stmt in stmts {
            return_typ = self.check_stmt(typed_fn_env, stmt)?;
        }

        // check the return
        match (expected_return, return_typ.clone()) {
            (None, None) => (),
            (Some(expected), None) => {
                return Err(self.error(ErrorKind::MissingReturn, expected.span))
            }
            (None, Some(_)) => {
                return Err(self.error(ErrorKind::NoReturnExpected, stmts.last().unwrap().span))
            }
            (Some(expected), Some(observed)) => {
                // todo: use exact match
                if !observed.same_as(&expected.kind) {
                    return Err(self.error(
                        ErrorKind::ReturnTypeMismatch(observed.clone(), expected.kind.clone()),
                        expected.span,
                    ));
                }
            }
        };

        Ok(return_typ)
    }

    pub fn check_stmt(
        &mut self,
        typed_fn_env: &mut MonomorphizedFnEnv,
        stmt: &Stmt,
    ) -> Result<Option<TyKind>> {
        match &stmt.kind {
            StmtKind::Assign { mutable, lhs, rhs } => {
                // but first we need to compute the type of the rhs expression
                let node = self.compute_type(rhs, typed_fn_env)?.unwrap();
                let type_info = MTypeInfo::new(node.typ, lhs.span, None);

                // store the type of lhs in the env
                typed_fn_env.store_type(lhs.value.clone(), type_info)?;
            }
            StmtKind::ForLoop { var, range, body } => {
                // todo: should we loop through each iteration of the block?
                for i in range.start..=range.end {
                    typed_fn_env.reassign_type(
                        var.value.clone(),
                        MTypeInfo::new(TyKind::BigInt, var.span, Some(i)),
                    )?;

                    // check block
                    self.check_block(typed_fn_env, body, None)?;
                }
            }
            StmtKind::Expr(expr) => {
                self.compute_type(expr, typed_fn_env)?;
            },
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
        typed_fn_env: &mut MonomorphizedFnEnv,
        method_call: bool, // indicates if it's a fn call or a method call
        fn_info: FnInfo<B>,
        args: &[Expr],
        span: Span,
    ) -> Result<Option<TyKind>> {
        let (fn_sig, stmts) = match &fn_info.kind {
            FnKind::BuiltIn(sig, _) => {
                (sig, Vec::<Stmt>::new())
            },
            FnKind::Native(func) => (&func.sig, func.body.clone()),
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
                observed.push((node, arg.span));
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

        // create a context for the function call
        let typed_fn_env = &mut MonomorphizedFnEnv::new();

        // typed_fn_env.nest();
        // compare argument types with the function signature
        for (sig_arg, (type_info, span)) in expected.iter().zip(observed) {
            // println!("{:?} {:?}", sig_arg, type_info);
            match &sig_arg.typ.kind {
                TyKind::Generic(gen_name) => {
                    // infer the generic value from the observed type
                    let val = type_info.constant;
                    let mty = MTypeInfo::new(type_info.typ, span, val);

                    // store the inferred value for generic parameter
                    typed_fn_env.store_type(
                        gen_name.clone(),
                        mty.clone(),
                    )?;

                    // store local var value
                    typed_fn_env.store_type(
                        sig_arg.name.value.clone(),
                        mty,
                    )?;
                }
                TyKind::GenericArray(typ, _) => todo!("generic array"),
                _ => {
                    // store the type of the argument in the env
                    typed_fn_env.store_type(
                        sig_arg.name.value.clone(),
                        MTypeInfo::new(type_info.typ.clone(), span, type_info.constant),
                    )?;
                },
            }
        }

        // evaluate generic return types using inferred values
        let ret_ty = match &fn_sig.return_type {
            Some(ret_ty) => {
                match &ret_ty.kind {
                    TyKind::Generic(gen_name) => {
                        // let val = eval_generic_array_size(&ret_ty.size, typed_fn_env);
                        // let mty = MTypeInfo::new(ret_ty.kind.clone(), ret_ty.span, Some(val));
                        // typed_fn_env.store_type(gen_name.clone(), mty)?;
                        todo!()
                    }
                    TyKind::GenericArray(typ, size) => {
                        let val = eval_generic_array_size(size, typed_fn_env);
                        let tykind = TyKind::Array(typ.clone(), val);
                        Some(Ty {
                            kind: tykind,
                            span: ret_ty.span,
                        })
                    },
                    _ => Some(ret_ty.clone()),
                }
            }
            None => None,
        };

        let res = self.check_block(typed_fn_env, &stmts, ret_ty.as_ref());
        // typed_fn_env.pop();

        res
    }

    pub fn error(&self, kind: ErrorKind, span: Span) -> Error {
        Error::new("mast", kind, span)
    }
}

pub fn eval_generic_array_size(
    sym: &Symbolic,
    typed_fn_env: &MonomorphizedFnEnv,
) -> u32 {
    match sym {
        Symbolic::Concrete(v) => *v,
        Symbolic::Generic(g) => typed_fn_env.get_type_info(&g.value).unwrap().value.unwrap(),
        Symbolic::Add(a, b) => {
            let a = eval_generic_array_size(a, typed_fn_env);
            let b = eval_generic_array_size(b, typed_fn_env);
            a + b
        }
        Symbolic::Mul(a, b) => {
            let a = eval_generic_array_size(a, typed_fn_env);
            let b = eval_generic_array_size(b, typed_fn_env);
            a * b
        }
    }
}