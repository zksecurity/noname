use num_bigint::BigUint;
use num_traits::ToPrimitive;
use serde::Serialize;
use std::collections::HashMap;

use crate::{
    backends::Backend,
    constants::Span,
    error::{Error, ErrorKind, Result},
    imports::FnKind,
    parser::{
        types::{
            FnSig, ForLoopArgument, GenericParameters, Ident, Range, Stmt, StmtKind, Symbolic, Ty,
            TyKind,
        },
        CustomType, Expr, ExprKind, FunctionDef, Op2,
    },
    syntax::{is_generic_parameter, is_type},
    type_checker::{ConstInfo, FnInfo, FullyQualified, StructInfo, TypeChecker},
};

pub mod ast;

/// ExprMonoInfo holds the monomorphized expression node and its resolved type.
#[derive(Debug, Clone)]
pub struct ExprMonoInfo {
    /// The monomorphized expression node.
    pub expr: Expr,

    /// The resolved type of the expression node.
    /// The generic types shouldn't be presented in this field.
    pub typ: Option<TyKind>,

    /// Propagated constant value
    pub constant: Option<PropagatedConstant>,
}

#[derive(Debug, Clone)]
pub enum PropagatedConstant {
    Single(BigUint),
    Array(Vec<PropagatedConstant>),
    Custom(HashMap<Ident, PropagatedConstant>),
}

impl PropagatedConstant {
    pub fn as_single(&self) -> BigUint {
        match self {
            PropagatedConstant::Single(v) => v.clone(),
            _ => panic!("expected single value"),
        }
    }

    pub fn as_array(&self) -> Vec<BigUint> {
        match self {
            PropagatedConstant::Array(v) => v.iter().map(|c| c.as_single()).collect(),
            _ => panic!("expected array value"),
        }
    }

    pub fn as_custom(&self) -> HashMap<Ident, BigUint> {
        match self {
            PropagatedConstant::Custom(v) => {
                v.iter().map(|(k, c)| (k.clone(), c.as_single())).collect()
            }
            _ => panic!("expected custom value"),
        }
    }
}

/// impl From trait for single value
impl From<BigUint> for PropagatedConstant {
    fn from(v: BigUint) -> Self {
        PropagatedConstant::Single(v)
    }
}

impl ExprMonoInfo {
    pub fn new(expr: Expr, typ: Option<TyKind>, value: Option<PropagatedConstant>) -> Self {
        Self {
            expr,
            typ,
            constant: value,
        }
    }

    /// There can be case expression node doesn't have a type.
    /// For example, the ExprKind::Assignment won't return a type.
    pub fn new_notype(expr: Expr) -> Self {
        Self {
            expr,
            typ: None,
            constant: None,
        }
    }
}

/// MTypeInfo holds the resolved type info to pass within a function scope.
/// It is stored in the scope context environment [MonomorphizedFnEnv].
#[derive(Debug, Clone)]
pub struct MTypeInfo {
    /// Some type information.
    pub typ: TyKind,

    /// Store constant value
    pub constant: Option<PropagatedConstant>,

    /// The span of the variable declaration.
    pub span: Span,
}

impl MTypeInfo {
    pub fn new(typ: &TyKind, span: Span, value: Option<PropagatedConstant>) -> Self {
        Self {
            typ: typ.clone(),
            span,
            constant: value,
        }
    }
}

/// A storage to manage the variables in function scopes.
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

        self.vars
            .retain(|_, (scope, _)| *scope <= self.current_scope);
    }

    /// Returns true if a scope is a prefix of our scope.
    pub fn is_in_scope(&self, prefix_scope: usize) -> bool {
        self.current_scope >= prefix_scope
    }

    /// Stores type information about a local variable.
    pub fn store_type(&mut self, ident: &str, type_info: &MTypeInfo) -> Result<()> {
        match self
            .vars
            .insert(ident.to_string(), (self.current_scope, type_info.clone()))
        {
            Some(_) => Err(Error::new(
                "mast",
                ErrorKind::DuplicateDefinition(ident.to_string()),
                type_info.span,
            )),
            None => Ok(()),
        }
    }

    /// Retrieves type information on a variable, given a name.
    /// If the variable is not in scope, return None.
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

impl<B: Backend> FnInfo<B> {
    /// Resolves the generic values based on observed arguments.
    pub fn resolve_generic_signature(
        &mut self,
        observed_args: &[ExprMonoInfo],
        ctx: &mut MastCtx<B>,
    ) -> Result<FnSig> {
        match self.kind {
            FnKind::BuiltIn(ref mut sig, _, _) => {
                sig.resolve_generic_values(observed_args, ctx)?;
            }
            FnKind::Native(ref mut func) => {
                func.sig.resolve_generic_values(observed_args, ctx)?;
            }
        };

        Ok(self.resolved_sig())
    }

    /// Returns the resolved signature of the function.
    pub fn resolved_sig(&self) -> FnSig {
        let fn_sig = self.sig();

        let (ret_typed, fn_args_typed) = if let Some(resolved) = &fn_sig.generics.resolved_sig {
            (resolved.return_type.clone(), resolved.arguments.clone())
        } else {
            (fn_sig.return_type.clone(), fn_sig.arguments.clone())
        };

        FnSig {
            name: fn_sig.monomorphized_name(),
            arguments: fn_args_typed,
            return_type: ret_typed,
            ..fn_sig.clone()
        }
    }
}

impl FnSig {
    /// Recursively resolve a type based on generic values
    pub fn resolve_type<B: Backend>(&self, typ: &TyKind, ctx: &mut MastCtx<B>) -> TyKind {
        match typ {
            TyKind::Array(ty, size) => TyKind::Array(Box::new(self.resolve_type(ty, ctx)), *size),
            TyKind::GenericSizedArray(ty, sym) => {
                let val = sym.eval(&self.generics, &ctx.tast);
                TyKind::Array(
                    Box::new(self.resolve_type(ty, ctx)),
                    val.to_u32().expect("array size exceeded u32"),
                )
            }
            _ => typ.clone(),
        }
    }

    /// Resolve generic values for each generic parameter
    pub fn resolve_generic_values<B: Backend>(
        &mut self,
        observed: &[ExprMonoInfo],
        ctx: &mut MastCtx<B>,
    ) -> Result<()> {
        for (sig_arg, observed_arg) in self.arguments.clone().iter().zip(observed) {
            let observed_ty = observed_arg.typ.clone().expect("expected type");
            match (&sig_arg.typ.kind, &observed_ty) {
                (TyKind::GenericSizedArray(_, _), TyKind::Array(_, _))
                | (TyKind::Array(_, _), TyKind::Array(_, _)) => {
                    self.resolve_generic_array(
                        &sig_arg.typ.kind,
                        &observed_ty,
                        observed_arg.expr.span,
                    )?;
                }
                // const NN: Field
                _ => {
                    let cst = observed_arg.constant.clone();
                    if is_generic_parameter(sig_arg.name.value.as_str()) && cst.is_some() {
                        self.generics.assign(
                            &sig_arg.name.value,
                            cst.unwrap().as_single(),
                            observed_arg.expr.span,
                        )?;
                    }
                }
            }
        }

        // resolve the argument types
        let mut resolved_args = vec![];
        for arg in &self.arguments {
            let resolved_arg_typ = self.resolve_type(&arg.typ.kind, ctx);
            let mut resolved_arg = arg.clone();
            resolved_arg.typ = Ty {
                kind: resolved_arg_typ,
                span: arg.typ.span,
            };
            resolved_args.push(resolved_arg);
        }

        // resolve the return type
        let mut return_type: Option<Ty> = None;
        if let Some(ty) = &self.return_type {
            let ret_typed = self.resolve_type(&ty.kind, ctx);
            return_type = Some(Ty {
                kind: ret_typed,
                span: ty.span,
            });
        }

        // store the resolved types in arguments and return
        self.generics.resolve_sig(resolved_args, return_type);

        Ok(())
    }
}

/// A context to store the last node id for the monomorphized AST.
#[derive(Debug)]
pub struct MastCtx<B>
where
    B: Backend,
{
    tast: TypeChecker<B>,
    generic_func_scope: Option<usize>,
    // new fully qualified function name as the key, old fully qualified function name as the value
    functions_instantiated: HashMap<FullyQualified, FullyQualified>,
    // new method name as the key, old method name as the value
    methods_instantiated: HashMap<(FullyQualified, String), String>,
    // cache for [PropagatedConstant] values from instantiated methods
    cst_method_cache: HashMap<(FullyQualified, String), PropagatedConstant>,
    // cache for [PropagatedConstant] values from instantiated functions
    cst_fn_cache: HashMap<FullyQualified, PropagatedConstant>,
}

impl<B: Backend> MastCtx<B> {
    pub fn new(tast: TypeChecker<B>) -> Self {
        Self {
            tast,
            generic_func_scope: Some(0),
            functions_instantiated: HashMap::new(),
            methods_instantiated: HashMap::new(),
            cst_method_cache: HashMap::new(),
            cst_fn_cache: HashMap::new(),
        }
    }

    pub fn next_node_id(&mut self) -> usize {
        let new_node_id = self.tast.last_node_id() + 1;
        self.tast.update_node_id(new_node_id);
        new_node_id
    }

    pub fn start_monomorphize_func(&mut self) {
        self.generic_func_scope = Some(self.generic_func_scope.unwrap() + 1);
    }

    pub fn finish_monomorphize_func(&mut self) {
        self.generic_func_scope = Some(self.generic_func_scope.unwrap() - 1);
    }

    pub fn add_monomorphized_fn(
        &mut self,
        old_qualified: FullyQualified,
        new_qualified: FullyQualified,
        fn_info: FnInfo<B>,
    ) {
        self.tast
            .add_monomorphized_fn(new_qualified.clone(), fn_info);
        self.functions_instantiated
            .insert(new_qualified, old_qualified);
    }

    pub fn add_monomorphized_method(
        &mut self,
        struct_qualified: FullyQualified,
        old_method_name: &str,
        method_name: &str,
        fn_info: &FunctionDef,
    ) {
        self.tast
            .add_monomorphized_method(struct_qualified.clone(), method_name, fn_info);

        self.methods_instantiated.insert(
            (struct_qualified, method_name.to_string()),
            old_method_name.to_string(),
        );
    }

    pub fn clear_generic_fns(&mut self) {
        for (new, old) in &self.functions_instantiated {
            // don't remove the instantiated function with no generic arguments
            if new != old {
                self.tast.remove_fn(old);
            }
        }

        for ((struct_qualified, new), old) in &self.methods_instantiated {
            // don't remove the instantiated method with no generic arguments
            if new != old {
                self.tast.remove_method(struct_qualified, old);
            }
        }
    }
}

impl Symbolic {
    /// Evaluate symbolic size to an integer.
    pub fn eval<B: Backend>(&self, gens: &GenericParameters, tast: &TypeChecker<B>) -> BigUint {
        match self {
            Symbolic::Concrete(v) => v.clone(),
            Symbolic::Constant(var) => {
                let qualified = FullyQualified::local(var.value.clone());
                let cst = tast.const_info(&qualified).expect("constant not found");

                // convert to u32
                let bigint: BigUint = cst.value[0].into();
                bigint.try_into().expect("biguint too large")
            }
            Symbolic::Generic(g) => gens.get(&g.value),
            Symbolic::Add(a, b) => a.eval(gens, tast) + b.eval(gens, tast),
            Symbolic::Mul(a, b) => a.eval(gens, tast) * b.eval(gens, tast),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
/// Mast relies on the TAST for the information about the "unresolved" types to monomorphize.
/// Things such as loading the function AST and struct AST from fully qualified names.
/// After monomorphization process, the following data will be updated:
/// - Resolved types. This can be used to determine the size of a type.
/// - Instantiated functions. The circuit writer will load the instantiated function AST by node id.
/// - Monomorphized AST is generated for the circuit writer to walk through and compute.
pub struct Mast<B: Backend>(#[serde(bound = "TypeChecker<B>: Serialize")] pub TypeChecker<B>);

impl<B: Backend> Mast<B> {
    /// Returns the concrete type for the given expression node.
    pub fn expr_type(&self, expr: &Expr) -> Option<&TyKind> {
        self.0.expr_type(expr)
    }

    /// Returns the struct info for the given fully qualified name.
    pub fn struct_info(&self, qualified: &FullyQualified) -> Option<&StructInfo> {
        self.0.struct_info(qualified)
    }

    /// Returns the constant variable info for the given fully qualified name.
    pub fn const_info(&self, qualified: &FullyQualified) -> Option<&ConstInfo<B::Field>> {
        self.0.const_info(qualified)
    }

    /// Returns the function info by fully qualified name.
    pub fn fn_info(&self, qualified: &FullyQualified) -> Option<&FnInfo<B>> {
        self.0.fn_info(qualified)
    }

    // TODO: might want to memoize that at some point
    /// Returns the number of field elements contained in the given type.
    pub(crate) fn size_of(&self, typ: &TyKind) -> usize {
        match typ {
            TyKind::Field { .. } => 1,
            TyKind::Custom { module, name } => {
                let qualified = FullyQualified::new(module, name);
                let struct_info = self
                    .struct_info(&qualified)
                    .expect("bug in the mast: cannot find struct info");

                let mut sum = 0;

                for (_, t) in &struct_info.fields {
                    sum += self.size_of(t);
                }

                sum
            }
            TyKind::Array(typ, len) => (*len as usize) * self.size_of(typ),
            TyKind::GenericSizedArray(_, _) => {
                unreachable!("generic arrays should have been resolved")
            }
            TyKind::Bool => 1,
        }
    }
}
/// Monomorphize the main function.
/// This is the entry point of the monomorphization process.
/// It stores the monomorphized AST at the end.
pub fn monomorphize<B: Backend>(tast: TypeChecker<B>) -> Result<Mast<B>> {
    let mut ctx = MastCtx::new(tast);

    let qualified = FullyQualified::local("main".to_string());
    let mut main_fn = ctx
        .tast
        .fn_info(&qualified)
        .expect("main function not found")
        .clone();

    let mut func_def = match &main_fn.kind {
        // `fn main() { ... }`
        FnKind::Native(function) => function.clone(),
        _ => Err(Error::new(
            "Backend - Monomorphize",
            ErrorKind::UnexpectedError("Main function must be native"),
            Span::default(),
        ))?,
    };

    // create a new typed fn environment to type check the function
    let mut mono_fn_env = MonomorphizedFnEnv::default();

    // store variables and their types in the fn_env
    for arg in &func_def.sig.arguments {
        // store the args' type in the fn environment
        mono_fn_env.store_type(
            &arg.name.value,
            &MTypeInfo::new(&arg.typ.kind, arg.span, None),
        )?;
    }

    // monomorphize main function body
    let (stmts, _) = monomorphize_block(
        &mut ctx,
        &mut mono_fn_env,
        &func_def.body,
        func_def.sig.return_type.as_ref(),
    )?;

    // override the main function AST with the monomorphized version
    func_def.body = stmts;
    main_fn.kind = FnKind::Native(func_def);

    ctx.tast.add_monomorphized_fn(qualified, main_fn.clone());

    ctx.clear_generic_fns();

    Ok(Mast(ctx.tast))
}

/// Recursively monomorphize an expression node.
/// It does two things:
/// - Monomorphize the expression node with the inferred generic values.
/// - Typecheck the resolved type.
fn monomorphize_expr<B: Backend>(
    ctx: &mut MastCtx<B>,
    expr: &Expr,
    mono_fn_env: &mut MonomorphizedFnEnv,
) -> Result<ExprMonoInfo> {
    let expr_mono: ExprMonoInfo = match &expr.kind {
        ExprKind::FieldAccess { lhs, rhs } => {
            let lhs_mono = monomorphize_expr(ctx, lhs, mono_fn_env)?;

            // obtain the type of the field
            let (module, struct_name) = match lhs_mono.typ {
                Some(TyKind::Custom { module, name }) => (module, name),
                _ => Err(Error::new(
                    "Monomorphize Expr",
                    ErrorKind::UnexpectedError("field access must be done on a custom struct"),
                    expr.span,
                ))?,
            };

            // get struct info
            let qualified = FullyQualified::new(&module, &struct_name);
            let struct_info = ctx
                .tast
                .struct_info(&qualified)
                .expect("this struct is not defined, or you're trying to access a field of a struct defined in a third-party library");

            // find field type
            let typ = struct_info
                .fields
                .iter()
                .find(|(name, _)| name == &rhs.value)
                .map(|(_, typ)| typ.clone());

            let mexpr = expr.to_mast(
                ctx,
                &ExprKind::FieldAccess {
                    lhs: Box::new(lhs_mono.expr),
                    rhs: rhs.clone(),
                },
            );

            // propagate the constant value
            let cst = lhs_mono.constant.and_then(|c| match c {
                PropagatedConstant::Custom(map) => map.get(rhs).cloned(),
                _ => None,
            });

            ExprMonoInfo::new(mexpr, typ, cst)
        }

        // `module::fn_name(args)`
        ExprKind::FnCall {
            module,
            fn_name,
            args,
            unsafe_attr,
        } => {
            // compute the observed arguments types
            let mut observed = Vec::with_capacity(args.len());
            for arg in args {
                let node = monomorphize_expr(ctx, arg, mono_fn_env)?;
                observed.push(node);
            }

            // retrieve the function signature
            let old_qualified = FullyQualified::new(module, &fn_name.value);
            let mut fn_info = ctx
                .tast
                .fn_info(&old_qualified)
                .expect("function not found")
                .to_owned();

            let args_mono = observed.clone().into_iter().map(|e| e.expr).collect();

            let resolved_sig = fn_info.resolve_generic_signature(&observed, ctx)?;

            let mono_qualified = FullyQualified::new(module, &resolved_sig.name.value);

            // check if this function is already monomorphized
            if ctx.functions_instantiated.contains_key(&mono_qualified) {
                let mexpr = expr.to_mast(
                    ctx,
                    &ExprKind::FnCall {
                        module: module.clone(),
                        fn_name: resolved_sig.name,
                        args: args_mono,
                        unsafe_attr: *unsafe_attr,
                    },
                );
                let resolved_sig = &fn_info.sig().generics.resolved_sig;

                let typ = resolved_sig
                    .as_ref()
                    .and_then(|sig| sig.return_type.clone().map(|t| t.kind));

                // retrieve the constant value from the cache
                let cst = ctx.cst_fn_cache.get(&mono_qualified).cloned();

                ExprMonoInfo::new(mexpr, typ, cst)
            } else {
                // monomorphize the function call
                let (fn_info_mono, typ, cst) =
                    instantiate_fn_call(ctx, fn_info, &observed, expr.span)?;

                // cache the constant value
                if let Some(cst) = cst.clone() {
                    ctx.cst_fn_cache.insert(mono_qualified.clone(), cst);
                }

                let fn_name_mono = &fn_info_mono.sig().name;
                let mexpr = expr.to_mast(
                    ctx,
                    &ExprKind::FnCall {
                        module: module.clone(),
                        fn_name: fn_name_mono.clone(),
                        args: args_mono,
                        unsafe_attr: *unsafe_attr,
                    },
                );

                let new_qualified = FullyQualified::new(module, &fn_name_mono.value);
                ctx.add_monomorphized_fn(old_qualified, new_qualified, fn_info_mono);

                ExprMonoInfo::new(mexpr, typ, cst)
            }
        }

        // `lhs.method_name(args)`
        ExprKind::MethodCall {
            lhs,
            method_name,
            args,
        } => {
            // retrieve struct name on the lhs
            let lhs_mono = monomorphize_expr(ctx, lhs, mono_fn_env)?;
            let (module, struct_name) = match lhs_mono.clone().typ {
                Some(TyKind::Custom { module, name }) => (module, name),
                _ => return Err(error(ErrorKind::MethodCallOnNonCustomStruct, expr.span)),
            };

            // get struct info
            let struct_qualified = FullyQualified::new(&module, &struct_name);
            let struct_info = ctx
                .tast
                .struct_info(&struct_qualified)
                .ok_or(error(
                    ErrorKind::UndefinedStruct(struct_name.clone()),
                    lhs.span,
                ))?
                .clone();

            // get method info
            let method_type = struct_info
                .methods
                .get(&method_name.value)
                .expect("method not found on custom struct (TODO: better error)");

            let fn_kind = FnKind::Native(method_type.clone());
            let mut fn_info = FnInfo {
                kind: fn_kind,
                is_hint: false,
                span: method_type.span,
            };

            // compute the observed arguments types
            let mut observed = Vec::with_capacity(args.len());
            if let Some(self_arg) = fn_info.sig().arguments.first() {
                if self_arg.name.value == "self" {
                    observed.push(monomorphize_expr(ctx, lhs, mono_fn_env)?);
                }
            }

            let mut args_mono = vec![];
            for arg in args {
                let expr_mono = monomorphize_expr(ctx, arg, mono_fn_env)?;
                observed.push(expr_mono.clone());
                args_mono.push(expr_mono.expr);
            }

            let resolved_sig = fn_info.resolve_generic_signature(&observed, ctx)?;

            // check if this function is already monomorphized
            if ctx
                .methods_instantiated
                .contains_key(&(struct_qualified.clone(), resolved_sig.name.value.clone()))
            {
                let mexpr = expr.to_mast(
                    ctx,
                    &ExprKind::MethodCall {
                        lhs: Box::new(lhs_mono.expr),
                        method_name: resolved_sig.name,
                        args: args_mono,
                    },
                );
                let typ = resolved_sig.return_type.clone().map(|t| t.kind);

                // retrieve the constant value from the cache
                let cst = ctx
                    .cst_method_cache
                    .get(&(struct_qualified.clone(), method_name.value.clone()))
                    .cloned();

                ExprMonoInfo::new(mexpr, typ, cst)
            } else {
                // monomorphize the function call
                let (fn_info_mono, typ, cst) =
                    instantiate_fn_call(ctx, fn_info, &observed, expr.span)?;
                // cache the constant value
                if let Some(cst) = cst.clone() {
                    ctx.cst_method_cache
                        .insert((struct_qualified.clone(), method_name.value.clone()), cst);
                }

                let fn_name_mono = &fn_info_mono.sig().name;
                let mexpr = expr.to_mast(
                    ctx,
                    &ExprKind::MethodCall {
                        lhs: Box::new(lhs_mono.expr),
                        method_name: fn_name_mono.clone(),
                        args: args_mono,
                    },
                );

                let fn_def = fn_info_mono.native().ok_or_else(|| {
                    Error::new(
                        "monomorphize-expr-MethodCall",
                        ErrorKind::UnexpectedError("Function kind is not native"),
                        fn_info_mono.span,
                    )
                })?;

                ctx.tast
                    .add_monomorphized_method(struct_qualified, &fn_name_mono.value, fn_def);

                ExprMonoInfo::new(mexpr, typ, cst)
            }
        }

        ExprKind::Assignment { lhs, rhs } => {
            // compute type of lhs
            let lhs_mono = monomorphize_expr(ctx, lhs, mono_fn_env)?;

            // and is of the same type as the rhs
            let rhs_mono = monomorphize_expr(ctx, rhs, mono_fn_env)?;

            let lhs_typ = lhs_mono.typ.unwrap();
            let rhs_typ = rhs_mono.typ.unwrap();

            if !lhs_typ.match_expected(&rhs_typ, true) {
                return Err(error(
                    ErrorKind::AssignmentTypeMismatch(lhs_typ, rhs_typ),
                    expr.span,
                ));
            }

            let mexpr = expr.to_mast(
                ctx,
                &ExprKind::Assignment {
                    lhs: Box::new(lhs_mono.expr),
                    rhs: Box::new(rhs_mono.expr),
                },
            );

            ExprMonoInfo::new_notype(mexpr)
        }

        ExprKind::BinaryOp {
            op,
            lhs,
            rhs,
            protected,
        } => {
            let lhs_mono = monomorphize_expr(ctx, lhs, mono_fn_env)?;
            let rhs_mono = monomorphize_expr(ctx, rhs, mono_fn_env)?;

            let typ = match op {
                Op2::Equality => Some(TyKind::Bool),
                Op2::Inequality => Some(TyKind::Bool),
                Op2::Addition
                | Op2::Subtraction
                | Op2::Multiplication
                | Op2::Division
                | Op2::BoolAnd
                | Op2::BoolOr => lhs_mono.typ,
            };

            let ExprMonoInfo { expr: lhs_expr, .. } = lhs_mono;
            let ExprMonoInfo { expr: rhs_expr, .. } = rhs_mono;

            // fold constants
            let cst = match (&lhs_expr.kind, &rhs_expr.kind) {
                (ExprKind::BigUInt(lhs), ExprKind::BigUInt(rhs)) => match op {
                    Op2::Addition => Some(lhs + rhs),
                    Op2::Subtraction => Some(lhs - rhs),
                    Op2::Multiplication => Some(lhs * rhs),
                    Op2::Division => Some(lhs / rhs),
                    _ => None,
                },
                _ => None,
            };

            match cst {
                Some(v) => {
                    let mexpr = expr.to_mast(ctx, &ExprKind::BigUInt(v.clone()));

                    ExprMonoInfo::new(mexpr, typ, Some(PropagatedConstant::from(v)))
                }
                // keep as is
                _ => {
                    let mexpr = expr.to_mast(
                        ctx,
                        &ExprKind::BinaryOp {
                            op: op.clone(),
                            protected: *protected,
                            lhs: Box::new(lhs_expr),
                            rhs: Box::new(rhs_expr),
                        },
                    );

                    ExprMonoInfo::new(mexpr, typ, None)
                }
            }
        }

        ExprKind::Negated(inner) => {
            // todo: can constant be negative?
            let inner_mono = monomorphize_expr(ctx, inner, mono_fn_env)?;

            let mexpr = expr.to_mast(ctx, &ExprKind::Negated(Box::new(inner_mono.expr)));

            ExprMonoInfo::new(mexpr, inner_mono.typ, None)
        }

        ExprKind::Not(inner) => {
            let inner_mono = monomorphize_expr(ctx, inner, mono_fn_env)?;

            let mexpr = expr.to_mast(ctx, &ExprKind::Not(Box::new(inner_mono.expr)));

            ExprMonoInfo::new(mexpr, Some(TyKind::Bool), None)
        }

        ExprKind::BigUInt(inner) => {
            let mexpr = expr.to_mast(ctx, &ExprKind::BigUInt(inner.clone()));

            ExprMonoInfo::new(
                mexpr,
                Some(TyKind::Field { constant: true }),
                Some(PropagatedConstant::from(inner.clone())),
            )
        }

        ExprKind::Bool(inner) => {
            let mexpr = expr.to_mast(ctx, &ExprKind::Bool(*inner));

            ExprMonoInfo::new(mexpr, Some(TyKind::Bool), None)
        }

        // mod::path.of.var
        // it could be also a generic variable
        ExprKind::Variable { module, name } => {
            let qualified = FullyQualified::new(module, &name.value);

            let res = if is_generic_parameter(&name.value) {
                let mtype = mono_fn_env.get_type_info(&name.value).unwrap();
                let cst = mtype.constant.clone().unwrap().as_single();
                let mexpr = expr.to_mast(ctx, &ExprKind::BigUInt(BigUint::from(cst)));

                ExprMonoInfo::new(mexpr, Some(mtype.typ.clone()), mtype.constant.clone())
            } else if is_type(&name.value) {
                let mtype = TyKind::Custom {
                    module: module.clone(),
                    name: name.value.clone(),
                };

                let mexpr = expr.to_mast(
                    ctx,
                    &ExprKind::Variable {
                        module: module.clone(),
                        name: name.clone(),
                    },
                );

                ExprMonoInfo::new(mexpr, Some(mtype), None)
            } else if let Some(cst) = ctx.tast.const_info(&qualified) {
                // if it's a variable,
                // check if it's a constant first
                let bigint: BigUint = cst.value[0].into();
                let mexpr = expr.to_mast(ctx, &ExprKind::BigUInt(bigint.clone()));

                ExprMonoInfo::new(
                    mexpr,
                    Some(TyKind::Field { constant: true }),
                    Some(PropagatedConstant::from(bigint)),
                )
            } else {
                // otherwise it's a local variable
                let mexpr = expr.to_mast(
                    ctx,
                    &ExprKind::Variable {
                        module: module.clone(),
                        name: name.clone(),
                    },
                );

                let mtype = mono_fn_env.get_type_info(&name.value).unwrap().clone();
                ExprMonoInfo::new(mexpr, Some(mtype.typ), mtype.constant)
            };

            res
        }

        ExprKind::ArrayAccess { array, idx } => {
            // get type of lhs
            let array_mono = monomorphize_expr(ctx, array, mono_fn_env)?;
            let id_mono = monomorphize_expr(ctx, idx, mono_fn_env)?;

            // get type of element
            let el_typ = match array_mono.typ {
                Some(TyKind::Array(typkind, _)) => Some(*typkind),
                _ => Err(Error::new(
                    "Array Access",
                    ErrorKind::UnexpectedError(
                        "Attempting to access array when type is not an array",
                    ),
                    expr.span,
                ))?,
            };

            let mexpr = expr.to_mast(
                ctx,
                &ExprKind::ArrayAccess {
                    array: Box::new(array_mono.expr),
                    idx: Box::new(id_mono.expr),
                },
            );

            // note that circuit write does the boundary check

            ExprMonoInfo::new(mexpr, el_typ, None)
        }

        ExprKind::ArrayDeclaration(items) => {
            let len: u32 = items.len().try_into().expect("array too large");

            let mut tykind: Option<TyKind> = None;

            let mut items_mono = vec![];

            for item in items {
                let item_mono = monomorphize_expr(ctx, item, mono_fn_env)?;
                items_mono.push(item_mono.clone());

                if let Some(tykind) = &tykind {
                    if !tykind.match_expected(&item_mono.clone().typ.unwrap(), true) {
                        return Err(error(
                            ErrorKind::MismatchType(tykind.clone(), item_mono.typ.unwrap()),
                            expr.span,
                        ));
                    }
                } else {
                    tykind = item_mono.typ;
                }
            }

            let mexpr = expr.to_mast(
                ctx,
                &ExprKind::ArrayDeclaration(items_mono.into_iter().map(|e| e.expr).collect()),
            );

            let item_typ = tykind.expect("expected a value");

            let typ = TyKind::Array(Box::new(item_typ), len);
            ExprMonoInfo::new(mexpr, Some(typ), None)
        }

        ExprKind::IfElse { cond, then_, else_ } => {
            let cond_mono = monomorphize_expr(ctx, cond, mono_fn_env)?;

            // compute type of if/else branches
            let then_mono = monomorphize_expr(ctx, then_, mono_fn_env)?;
            let else_mono = monomorphize_expr(ctx, else_, mono_fn_env)?;

            // make sure that the type of then_ and else_ match
            if then_mono.typ != else_mono.typ {
                Err(Error::new(
                    "If-Else Monomorphization",
                    ErrorKind::UnexpectedError(
                        "`if` branch and `else` branch must have matching types",
                    ),
                    expr.span,
                ))?
            }

            let mexpr = expr.to_mast(
                ctx,
                &ExprKind::IfElse {
                    cond: Box::new(cond_mono.expr),
                    then_: Box::new(then_mono.expr),
                    else_: Box::new(else_mono.expr),
                },
            );

            ExprMonoInfo::new(mexpr, then_mono.typ, None)
        }

        ExprKind::CustomTypeDeclaration { custom, fields } => {
            let CustomType {
                module,
                name,
                span: _,
            } = custom;
            let qualified = FullyQualified::new(module, name);
            let struct_info = ctx
                .tast
                .struct_info(&qualified)
                .ok_or_else(|| error(ErrorKind::UndefinedStruct(name.clone()), expr.span))?;

            let defined_fields = &struct_info.fields.clone();

            let mut fields_mono = vec![];

            for (defined, observed) in defined_fields.iter().zip(fields) {
                let ident = observed.0.clone();
                if defined.0 != ident.value {
                    return Err(error(
                        ErrorKind::InvalidStructField(defined.0.clone(), ident.value.clone()),
                        expr.span,
                    ));
                }

                let observed_mono = &monomorphize_expr(ctx, &observed.1, mono_fn_env)?;
                let typ_mono = observed_mono.typ.as_ref().expect("expected a value");

                if !typ_mono.match_expected(&defined.1, true) {
                    return Err(error(
                        ErrorKind::InvalidStructFieldType(defined.1.clone(), typ_mono.clone()),
                        expr.span,
                    ));
                }

                fields_mono.push((
                    ident,
                    observed_mono.expr.clone(),
                    observed_mono.constant.clone(),
                ));
            }

            let mexpr = expr.to_mast(
                ctx,
                &ExprKind::CustomTypeDeclaration {
                    custom: custom.clone(),
                    // extract a tuple of first two elements
                    fields: fields_mono
                        .iter()
                        .map(|(a, b, _)| (a.clone(), b.clone()))
                        .collect(),
                },
            );

            let cst_fields = {
                let csts = HashMap::from_iter(
                    fields_mono
                        .into_iter()
                        .filter(|(_, _, cst)| cst.is_some())
                        .map(|(ident, _, cst)| (ident, cst.unwrap())),
                );
                if csts.is_empty() {
                    None
                } else {
                    Some(PropagatedConstant::Custom(csts))
                }
            };

            ExprMonoInfo::new(
                mexpr,
                Some(TyKind::Custom {
                    module: module.clone(),
                    name: name.clone(),
                }),
                cst_fields,
            )
        }
        ExprKind::RepeatedArrayInit { item, size } => {
            let item_mono = monomorphize_expr(ctx, item, mono_fn_env)?;
            let size_mono = monomorphize_expr(ctx, size, mono_fn_env)?;

            let item_typ = item_mono.typ.expect("expected a value");
            let mexpr = expr.to_mast(
                ctx,
                &ExprKind::RepeatedArrayInit {
                    item: Box::new(item_mono.expr),
                    size: Box::new(size_mono.expr),
                },
            );

            if let Some(cst) = size_mono.constant {
                let arr_typ = TyKind::Array(
                    Box::new(item_typ),
                    cst.as_single().to_u32().expect("array size too large"),
                );
                ExprMonoInfo::new(mexpr, Some(arr_typ), None)
            } else {
                return Err(error(ErrorKind::InvalidArraySize, expr.span));
            }
        }
    };

    if let Some(typ) = &expr_mono.typ {
        ctx.tast
            .add_monomorphized_type(expr_mono.expr.node_id, typ.clone());
    }

    Ok(expr_mono)
}

/// Monomorphize a block of statements.
pub fn monomorphize_block<B: Backend>(
    ctx: &mut MastCtx<B>,
    mono_fn_env: &mut MonomorphizedFnEnv,
    stmts: &[Stmt],
    expected_return: Option<&Ty>,
) -> Result<(Vec<Stmt>, Option<ExprMonoInfo>)> {
    mono_fn_env.nest();

    let mut ret_expr_mono = None;

    let mut stmts_mono = vec![];

    for stmt in stmts {
        if let Some((stmt, expr_mono)) = monomorphize_stmt(ctx, mono_fn_env, stmt)? {
            stmts_mono.push(stmt);

            // only return stmt can return `ExprMonoInfo` which contains propagated constants
            if expr_mono.is_some() {
                ret_expr_mono = expr_mono;
            }
        }
    }

    let return_typ = if let Some(expr_mono) = ret_expr_mono.clone() {
        expr_mono.typ
    } else {
        None
    };

    // check the return
    if let (Some(expected), Some(observed)) = (expected_return, return_typ.clone()) {
        if !observed.match_expected(&expected.kind, true) {
            return Err(error(
                ErrorKind::ReturnTypeMismatch(observed.clone(), expected.kind.clone()),
                expected.span,
            ));
        }
    };

    mono_fn_env.pop();

    Ok((stmts_mono, ret_expr_mono))
}

/// Monomorphize a statement.
pub fn monomorphize_stmt<B: Backend>(
    ctx: &mut MastCtx<B>,
    mono_fn_env: &mut MonomorphizedFnEnv,
    stmt: &Stmt,
) -> Result<Option<(Stmt, Option<ExprMonoInfo>)>> {
    let res = match &stmt.kind {
        StmtKind::Assign { mutable, lhs, rhs } => {
            let rhs_mono = monomorphize_expr(ctx, rhs, mono_fn_env)?;
            let typ = rhs_mono.typ.as_ref().expect("expected a type");
            let type_info = MTypeInfo::new(typ, lhs.span, rhs_mono.constant);

            // store the type of lhs in the env
            mono_fn_env.store_type(&lhs.value, &type_info)?;

            let stmt_mono = Stmt {
                kind: StmtKind::Assign {
                    mutable: *mutable,
                    lhs: lhs.clone(),
                    rhs: Box::new(rhs_mono.expr),
                },
                span: stmt.span,
            };

            Some((stmt_mono, None))
        }
        StmtKind::ForLoop {
            var,
            argument,
            body,
        } => {
            // enter a new scope
            mono_fn_env.nest();

            let argument_mono = match argument {
                ForLoopArgument::Range(range) => {
                    mono_fn_env.store_type(
                        &var.value,
                        // because we don't unroll the loop in the monomorphized AST,
                        // there is no constant value to propagate.
                        &MTypeInfo::new(&TyKind::Field { constant: true }, var.span, None),
                    )?;

                    let start_mono = monomorphize_expr(ctx, &range.start, mono_fn_env)?;
                    let end_mono = monomorphize_expr(ctx, &range.end, mono_fn_env)?;

                    if start_mono.constant.is_none() || end_mono.constant.is_none() {
                        return Err(error(ErrorKind::InvalidRangeSize, stmt.span));
                    }

                    if start_mono.constant.unwrap().as_single()
                        > end_mono.constant.unwrap().as_single()
                    {
                        return Err(error(ErrorKind::InvalidRangeSize, stmt.span));
                    }

                    let range_mono = Range {
                        start: start_mono.expr,
                        end: end_mono.expr,
                        span: range.span,
                    };
                    ForLoopArgument::Range(range_mono)
                }
                ForLoopArgument::Iterator(iterator) => {
                    let iterator_mono = monomorphize_expr(ctx, iterator, mono_fn_env)?;
                    let array_element_type = match iterator_mono.typ {
                        Some(TyKind::Array(t, _)) => t,
                        _ => Err(Error::new(
                            "Monomorphize Stmt - ForLoopArgument",
                            ErrorKind::UnexpectedError(
                                "Expected array while monomorphizing iterator's type",
                            ),
                            stmt.span,
                        ))?,
                    };

                    mono_fn_env.store_type(
                        &var.value,
                        &MTypeInfo::new(&array_element_type, var.span, None),
                    )?;
                    ForLoopArgument::Iterator(Box::new(iterator_mono.expr))
                }
            };

            let (stmts_mono, _) = monomorphize_block(ctx, mono_fn_env, body, None)?;
            let loop_stmt_mono = Stmt {
                kind: StmtKind::ForLoop {
                    var: var.clone(),
                    argument: argument_mono,
                    body: stmts_mono,
                },
                span: stmt.span,
            };

            // exit the scope
            mono_fn_env.pop();

            Some((loop_stmt_mono, None))
        }
        StmtKind::Expr(expr) => {
            let expr_mono = monomorphize_expr(ctx, expr, mono_fn_env)?;
            let stmt_mono = Stmt {
                kind: StmtKind::Expr(Box::new(expr_mono.expr)),
                span: stmt.span,
            };

            Some((stmt_mono, None))
        }
        StmtKind::Return(res) => {
            let expr_mono = monomorphize_expr(ctx, res, mono_fn_env)?;
            let stmt_mono = Stmt {
                kind: StmtKind::Return(Box::new(expr_mono.expr.clone())),
                span: stmt.span,
            };

            Some((stmt_mono, Some(expr_mono)))
        }
        StmtKind::Comment(_) => None,
    };

    Ok(res)
}

/// Overall, the function call checking process is as follows:
/// 1. resolve generic values from observed function args
/// 2. propagate the resolved values within the function body
/// 3. type check the observed return and the resolved expected return
pub fn instantiate_fn_call<B: Backend>(
    ctx: &mut MastCtx<B>,
    fn_info: FnInfo<B>,
    args: &[ExprMonoInfo],
    span: Span,
) -> Result<(FnInfo<B>, Option<TyKind>, Option<PropagatedConstant>)> {
    ctx.start_monomorphize_func();

    let fn_sig = fn_info.sig();

    // canonicalize the arguments depending on method call or not
    let expected: Vec<_> = fn_sig.arguments.iter().collect();

    // check argument length
    if expected.len() != args.len() {
        return Err(error(
            ErrorKind::MismatchFunctionArguments(args.len(), expected.len()),
            span,
        ));
    }

    // create a context for the function call
    let mono_fn_env = &mut MonomorphizedFnEnv::new();

    // store the values for generic parameters in the env
    for gen in &fn_sig.generics.names() {
        let val = fn_sig.generics.get(gen);
        mono_fn_env.store_type(
            gen,
            &MTypeInfo::new(
                &TyKind::Field { constant: true },
                span,
                Some(PropagatedConstant::from(val)),
            ),
        )?;
    }

    // store the types of the arguments in the env
    for (sig_arg, mono_info) in expected.iter().zip(args) {
        let arg_name = &sig_arg.name.value;

        // generic parameters should have been stored in the env
        if is_generic_parameter(arg_name) {
            continue;
        }

        let typ = mono_info.typ.as_ref().expect("expected a value");
        mono_fn_env.store_type(
            arg_name,
            &MTypeInfo::new(typ, mono_info.expr.span, mono_info.constant.clone()),
        )?;
    }

    let sig_typed = fn_info.resolved_sig();
    let ret_typed = sig_typed.return_type.clone();

    // construct the monomorphized function AST
    let (func_def, mono_info) = match fn_info.kind {
        FnKind::BuiltIn(_, handle, ignore_arg_types) => (
            FnInfo {
                kind: FnKind::BuiltIn(sig_typed, handle, ignore_arg_types),
                ..fn_info
            },
            // todo: we will need to propagate the constant value from builtin function as well
            None,
        ),
        FnKind::Native(fn_def) => {
            let (stmts_typed, mono_info) =
                monomorphize_block(ctx, mono_fn_env, &fn_def.body, ret_typed.as_ref())?;

            (
                FnInfo {
                    kind: FnKind::Native(FunctionDef {
                        sig: sig_typed,
                        body: stmts_typed,
                        ..fn_def
                    }),
                    is_hint: fn_info.is_hint,
                    span: fn_info.span,
                },
                mono_info,
            )
        }
    };

    ctx.finish_monomorphize_func();

    let cst = mono_info.and_then(|c| c.constant);

    Ok((func_def, ret_typed.map(|t| t.kind), cst))
}
pub fn error(kind: ErrorKind, span: Span) -> Error {
    Error::new("mast", kind, span)
}
