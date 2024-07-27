use num_bigint::BigUint;
use std::collections::{HashMap, HashSet};

use crate::{
    backends::Backend,
    constants::Span,
    error::{Error, ErrorKind, Result},
    imports::FnKind,
    parser::{
        types::{FnArg, FnSig, GenericParameters, Range, Stmt, StmtKind, Symbolic, Ty, TyKind},
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

    // todo: see if we can do constant folding on the expression nodes.
    // - it is possible to remove this field, as the constant value can be extracted from folded expression node
    /// Numeric value of the expression
    /// applicable to BigInt type
    pub constant: Option<u32>,
}

impl ExprMonoInfo {
    pub fn new(expr: Expr, typ: Option<TyKind>, value: Option<u32>) -> Self {
        if value.is_some() && !matches!(typ, Some(TyKind::BigInt)) {
            panic!("value can only be set for BigInt type");
        }

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
    pub value: Option<u32>,

    /// The span of the variable declaration.
    pub span: Span,
}

impl MTypeInfo {
    pub fn new(typ: &TyKind, span: Span, value: Option<u32>) -> Self {
        Self {
            typ: typ.clone(),
            span,
            value,
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

        // remove variables as we exit the scope
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
    pub fn store_type(&mut self, ident: &str, type_info: &MTypeInfo) -> Result<()> {
        match self
            .vars
            .insert(ident.to_string(), (self.current_scope, type_info.clone()))
        {
            Some(_) => Err(Error::new(
                "type-checker",
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

/// A context to store the last node id for the monomorphized AST.
#[derive(Debug)]
pub struct MastCtx<B>
where
    B: Backend,
{
    last_node_id: usize,
    tast: TypeChecker<B>,
    node_types: HashMap<usize, TyKind>,
    node_functions: HashMap<usize, FnInfo<B>>,
}

impl<B: Backend> MastCtx<B> {
    pub fn new(tast: TypeChecker<B>) -> Self {
        Self {
            last_node_id: 0,
            tast,
            node_types: HashMap::new(),
            node_functions: HashMap::new(),
        }
    }

    pub fn next_node_id(&mut self) -> usize {
        self.last_node_id += 1;
        self.last_node_id
    }
}

impl Symbolic {
    /// Evaluate symbolic size to an integer.
    pub fn eval(&self, mono_fn_env: &MonomorphizedFnEnv) -> u32 {
        match self {
            Symbolic::Concrete(v) => *v,
            Symbolic::Generic(g) => mono_fn_env.get_type_info(&g.value).unwrap().value.unwrap(),
            Symbolic::Add(a, b) => a.eval(mono_fn_env) + b.eval(mono_fn_env),
            Symbolic::Mul(a, b) => a.eval(mono_fn_env) * b.eval(mono_fn_env),
        }
    }
}

impl GenericParameters {
    /// Return all generic parameter names
    pub fn names(&self) -> HashSet<String> {
        self.0.keys().cloned().collect()
    }

    /// Add an unbound generic parameter
    pub fn add(&mut self, name: String) {
        self.0.insert(name, None);
    }

    /// Get the value of a generic parameter
    pub fn get(&self, name: &str) -> u32 {
        self.0
            .get(name)
            .expect("generic parameter not found")
            .expect("generic value not assigned")
    }

    /// Bind a generic parameter to a value
    pub fn assign(&mut self, name: &String, value: u32, span: Span) -> Result<()> {
        let existing = self.0.get(name);
        match existing {
            Some(Some(v)) => {
                if *v == value {
                    return Ok(());
                }

                Err(Error::new(
                    "mast",
                    ErrorKind::ConflictGenericValue(name.to_string(), *v, value),
                    span,
                ))
            }
            Some(None) => {
                self.0.insert(name.to_string(), Some(value));
                Ok(())
            }
            None => Err(Error::new(
                "mast",
                ErrorKind::UnexpectedGenericParameter(name.to_string()),
                span,
            )),
        }
    }
}

#[derive(Debug)]
/// Mast relies on the TAST for the information about the "unresolved" types to monomorphize.
/// Things such as loading the function AST and struct AST from fully qualified names.
/// After monomorphization process, the following data will be updated:
/// - Resolved types. This can be used to determine the size of a type.
/// - Instantiated functions. The circuit writer will load the instantiated function AST by node id.
/// - Monomorphized AST is generated for the circuit writer to walk through and compute.
pub struct Mast<B>
where
    B: Backend,
{
    tast: TypeChecker<B>,

    /// Mapping from node id to resolved type
    node_types: HashMap<usize, TyKind>,

    /// Mapping from node id to instantiated functions
    node_functions: HashMap<usize, FnInfo<B>>,

    /// Monomorphized AST of the main function
    main_fn_ast: FunctionDef,
}

impl<B: Backend> Mast<B> {
    /// Returns the concrete type for the given expression node.
    pub fn expr_type(&self, expr: &Expr) -> Option<&TyKind> {
        self.node_types.get(&expr.node_id)
    }

    /// Returns the instantiated function for the given expression node, which should be [FnCall] node.
    pub fn expr_fn(&self, expr: &Expr) -> Option<&FnInfo<B>> {
        self.node_functions.get(&expr.node_id)
    }

    /// Returns the struct info for the given fully qualified name.
    pub fn struct_info(&self, qualified: &FullyQualified) -> Option<&StructInfo> {
        self.tast.struct_info(qualified)
    }

    /// Returns the constant variable info for the given fully qualified name.
    pub fn const_info(&self, qualified: &FullyQualified) -> Option<&ConstInfo<B::Field>> {
        self.tast.const_info(qualified)
    }

    /// Returns the AST for monomorphized main function
    pub fn ast(&self) -> &FunctionDef {
        &self.main_fn_ast
    }

    // TODO: might want to memoize that at some point
    /// Returns the number of field elements contained in the given type.
    pub(crate) fn size_of(&self, typ: &TyKind) -> usize {
        match typ {
            TyKind::Field => 1,
            TyKind::Custom { module, name } => {
                let qualified = FullyQualified::new(module, name);
                let struct_info = self
                    .struct_info(&qualified)
                    .expect("bug in the type checker: cannot find struct info");

                let mut sum = 0;

                for (_, t) in &struct_info.fields {
                    sum += self.size_of(t);
                }

                sum
            }
            TyKind::BigInt => 1,
            TyKind::Array(typ, len) => (*len as usize) * self.size_of(typ),
            TyKind::GenericArray(_, _) => unreachable!("generic arrays should have been resolved"),
            TyKind::Bool => 1,
        }
    }
}
/// Monomorphize the main function.
/// This is the entry point of the monomorphization process.
/// It stores the monomorphized AST at the end.
pub fn monomorphize<B: Backend>(tast: TypeChecker<B>) -> Result<Mast<B>> {
    let qualified = FullyQualified::local("main".to_string());
    let main_fn = tast.fn_info(&qualified).expect("main function not found");

    let func_def = match &main_fn.kind {
        // `fn main() { ... }`
        FnKind::Native(function) => function.clone(),

        _ => panic!("main function must be native"),
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

    // the output value returned by the main function is also a main_args with a special name (public_output)
    if let Some(typ) = &func_def.sig.return_type {
        match typ.kind {
            TyKind::Field => {
                mono_fn_env.store_type(
                    "public_output".as_ref(),
                    &MTypeInfo::new(&typ.kind, typ.span, None),
                )?;
            }
            TyKind::Array(_, _) => {
                mono_fn_env.store_type(
                    "public_output".as_ref(),
                    &MTypeInfo::new(&typ.kind, typ.span, None),
                )?;
            }
            _ => unimplemented!(),
        }
    }

    let mut ctx = MastCtx::new(tast);

    // inferring for main function body
    let (stmts_mono, _) = monomorphize_block(
        &mut ctx,
        &mut mono_fn_env,
        &func_def.body,
        func_def.sig.return_type.as_ref(),
    )?;

    let main_fn_ast = FunctionDef {
        sig: func_def.sig,
        body: stmts_mono,
        span: func_def.span,
    };

    Ok(Mast {
        tast: ctx.tast,
        node_types: ctx.node_types,
        node_functions: ctx.node_functions,
        main_fn_ast,
    })
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
                _ => panic!("field access must be done on a custom struct"),
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

            let cst = None;
            ExprMonoInfo::new(mexpr, typ, cst)
        }

        // `module::fn_name(args)`
        ExprKind::FnCall {
            module,
            fn_name,
            args,
        } => {
            // compute the observed arguments types
            let mut observed = Vec::with_capacity(args.len());
            for arg in args {
                let node = monomorphize_expr(ctx, arg, mono_fn_env)?;
                observed.push(node);
            }

            // retrieve the function signature
            let qualified = FullyQualified::new(module, &fn_name.value);
            let fn_info = ctx
                .tast
                .fn_info(&qualified)
                .expect("function not found")
                .to_owned();

            let args_mono = observed.clone().into_iter().map(|e| e.expr).collect();

            let mexpr = expr.to_mast(
                ctx,
                &ExprKind::FnCall {
                    module: module.clone(),
                    fn_name: fn_name.clone(),
                    args: args_mono,
                },
            );

            // instantiate the function call
            let (fn_info_mono, typ) = instantiate_fn_call(ctx, fn_info, &observed, expr.span)?;

            ctx.node_functions.insert(mexpr.node_id, fn_info_mono);

            // assume the function call won't return constant value
            ExprMonoInfo::new(mexpr, typ, None)
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
            let qualified = FullyQualified::new(&module, &struct_name);
            let struct_info = ctx
                .tast
                .struct_info(&qualified)
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
            let fn_info = FnInfo {
                kind: fn_kind,
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

            let mexpr = expr.to_mast(
                ctx,
                &ExprKind::MethodCall {
                    lhs: Box::new(lhs_mono.expr),
                    method_name: method_name.clone(),
                    args: args_mono,
                },
            );

            // instantiate the function call
            let (fn_info_mono, typ) = instantiate_fn_call(ctx, fn_info, &observed, expr.span)?;
            ctx.node_functions.insert(mexpr.node_id, fn_info_mono);

            // assume the function call won't return constant value
            ExprMonoInfo::new(mexpr, typ, None)
        }

        ExprKind::Assignment { lhs, rhs } => {
            // compute type of lhs
            let lhs_mono = monomorphize_expr(ctx, lhs, mono_fn_env)?;

            // and is of the same type as the rhs
            let rhs_mono = monomorphize_expr(ctx, rhs, mono_fn_env)?;

            if !rhs_mono.typ.unwrap().same_as(&lhs_mono.typ.unwrap()) {
                panic!("lhs type doesn't match rhs type");
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

            let cst = match (lhs_mono.constant, rhs_mono.constant) {
                (Some(lhs), Some(rhs)) => match op {
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
                    let mexpr = expr.to_mast(ctx, &ExprKind::BigUInt(BigUint::from(v)));

                    ExprMonoInfo::new(mexpr, typ, Some(v))
                }
                None => {
                    let mexpr = expr.to_mast(
                        ctx,
                        &ExprKind::BinaryOp {
                            op: op.clone(),
                            protected: *protected,
                            lhs: Box::new(lhs_mono.expr),
                            rhs: Box::new(rhs_mono.expr),
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

            ExprMonoInfo::new(mexpr, Some(TyKind::Field), None)
        }

        ExprKind::Not(inner) => {
            let inner_mono = monomorphize_expr(ctx, inner, mono_fn_env)?;

            let mexpr = expr.to_mast(ctx, &ExprKind::Not(Box::new(inner_mono.expr)));

            ExprMonoInfo::new(mexpr, Some(TyKind::Bool), None)
        }

        ExprKind::BigUInt(inner) => {
            let cst: u32 = inner.try_into().expect("biguint too large");
            let mexpr = expr.to_mast(ctx, &ExprKind::BigUInt(inner.clone()));

            ExprMonoInfo::new(mexpr, Some(TyKind::BigInt), Some(cst))
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
                let mexpr =
                    expr.to_mast(ctx, &ExprKind::BigUInt(BigUint::from(mtype.value.unwrap())));

                ExprMonoInfo::new(mexpr, Some(mtype.typ.clone()), mtype.value)
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
                let cst: u32 = bigint.clone().try_into().expect("biguint too large");
                let mexpr = expr.to_mast(ctx, &ExprKind::BigUInt(bigint));

                ExprMonoInfo::new(mexpr, Some(TyKind::BigInt), Some(cst))
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
                ExprMonoInfo::new(mexpr, Some(mtype.typ), mtype.value)
            };

            res
        }

        ExprKind::ArrayAccess { array, idx } => {
            // get type of lhs
            let array_mono = monomorphize_expr(ctx, array, mono_fn_env)?;
            let id_mono = monomorphize_expr(ctx, idx, mono_fn_env)?;

            // get type of element
            let el_typ = match array_mono.typ.unwrap() {
                TyKind::Array(typkind, _) => Some(*typkind),
                _ => panic!("not an array"),
            };

            let mexpr = expr.to_mast(
                ctx,
                &ExprKind::ArrayAccess {
                    array: Box::new(array_mono.expr),
                    idx: Box::new(id_mono.expr),
                },
            );

            // todo: check the bounds of the array

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
                    if !tykind.same_as(&item_mono.clone().typ.unwrap()) {
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
                panic!("`if` branch and `else` branch must have matching types");
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

                if !typ_mono.same_as(&defined.1) {
                    return Err(error(
                        ErrorKind::InvalidStructFieldType(defined.1.clone(), typ_mono.clone()),
                        expr.span,
                    ));
                }

                fields_mono.push((ident, observed_mono.expr.clone()));
            }

            let mexpr = expr.to_mast(
                ctx,
                &ExprKind::CustomTypeDeclaration {
                    custom: custom.clone(),
                    fields: fields_mono,
                },
            );

            ExprMonoInfo::new(
                mexpr,
                Some(TyKind::Custom {
                    module: module.clone(),
                    name: name.clone(),
                }),
                None,
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
                let arr_typ = TyKind::Array(Box::new(item_typ), cst);
                ExprMonoInfo::new(mexpr, Some(arr_typ), None)
            } else {
                return Err(error(ErrorKind::InvalidArraySize, expr.span));
            }
        }
    };

    if let Some(typ) = &expr_mono.typ {
        ctx.node_types.insert(expr_mono.expr.node_id, typ.clone());
    }

    Ok(expr_mono)
}

/// Monomorphize a block of statements.
pub fn monomorphize_block<B: Backend>(
    ctx: &mut MastCtx<B>,
    mono_fn_env: &mut MonomorphizedFnEnv,
    stmts: &[Stmt],
    expected_return: Option<&Ty>,
) -> Result<(Vec<Stmt>, Option<TyKind>)> {
    mono_fn_env.nest();

    let mut return_typ = None;

    let mut stmts_mono = vec![];

    for stmt in stmts {
        if let Some((stmt, ret_typ)) = monomorphize_stmt(ctx, mono_fn_env, stmt)? {
            stmts_mono.push(stmt);

            if ret_typ.is_some() {
                return_typ = ret_typ;
            }
        }
    }

    // check the return
    if let (Some(expected), Some(observed)) = (expected_return, return_typ.clone()) {
        if !observed.same_as(&expected.kind) {
            return Err(error(
                ErrorKind::ReturnTypeMismatch(observed.clone(), expected.kind.clone()),
                expected.span,
            ));
        }
    };

    mono_fn_env.pop();

    Ok((stmts_mono, return_typ))
}

/// Monomorphize a statement.
pub fn monomorphize_stmt<B: Backend>(
    ctx: &mut MastCtx<B>,
    mono_fn_env: &mut MonomorphizedFnEnv,
    stmt: &Stmt,
) -> Result<Option<(Stmt, Option<TyKind>)>> {
    let res = match &stmt.kind {
        StmtKind::Assign { mutable, lhs, rhs } => {
            let rhs_mono = monomorphize_expr(ctx, rhs, mono_fn_env)?;
            let typ = rhs_mono.typ.as_ref().expect("expected a type");
            let type_info = MTypeInfo::new(typ, lhs.span, None);

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
        StmtKind::ForLoop { var, range, body } => {
            mono_fn_env.store_type(
                &var.value,
                // because we don't unroll the loop in the monomorphized AST
                // no constant value even if it's a BigInt
                &MTypeInfo::new(&TyKind::BigInt, var.span, None),
            )?;

            let start_mono = monomorphize_expr(ctx, &range.start, mono_fn_env)?;
            let end_mono = monomorphize_expr(ctx, &range.end, mono_fn_env)?;

            if start_mono.constant.is_none() || end_mono.constant.is_none() {
                return Err(error(ErrorKind::InvalidRangeSize, stmt.span));
            }

            if start_mono.constant.unwrap() > end_mono.constant.unwrap() {
                return Err(error(ErrorKind::InvalidRangeSize, stmt.span));
            }

            let range_mono = Range {
                start: start_mono.expr,
                end: end_mono.expr,
                span: range.span,
            };

            let (stmts_mono, _) = monomorphize_block(ctx, mono_fn_env, body, None)?;
            let loop_stmt_mono = Stmt {
                kind: StmtKind::ForLoop {
                    var: var.clone(),
                    range: range_mono,
                    body: stmts_mono,
                },
                span: stmt.span,
            };

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
                kind: StmtKind::Return(Box::new(expr_mono.expr)),
                span: stmt.span,
            };

            Some((stmt_mono, expr_mono.typ))
        }
        StmtKind::Comment(_) => None,
    };

    Ok(res)
}

/// Overall, the function call check process is as follows:
/// 1. infer generic values from function args
/// 2. evaluate generic return types using inferred values
/// 3. evaluate function body using inferred values
/// 4. type check the observed return and the inferred expected return (handled by check_body)
/// 5. return an reconstructed FunctionDef AST
pub fn instantiate_fn_call<B: Backend>(
    ctx: &mut MastCtx<B>,
    fn_info: FnInfo<B>,
    args: &[ExprMonoInfo],
    span: Span,
) -> Result<(FnInfo<B>, Option<TyKind>)> {
    let (fn_sig, stmts) = match &fn_info.kind {
        FnKind::BuiltIn(sig, _) => {
            let mut sig = sig.clone();
            // resolve generic values
            sig.resolve_generic_values(args)?;

            (sig, Vec::<Stmt>::new())
        }
        FnKind::Native(func) => {
            let mut sig = func.sig.clone();
            // resolve generic values
            sig.resolve_generic_values(args)?;

            (sig, func.body.clone())
        }
    };

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
        mono_fn_env.store_type(gen, &MTypeInfo::new(&TyKind::BigInt, span, Some(val)))?;
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
            &MTypeInfo::new(typ, mono_info.expr.span, mono_info.constant),
        )?;
    }

    // reconstruct FnArgs using the observed types
    let fn_args_typed = expected
        .iter()
        .zip(args)
        .map(|(arg, mono_info)| FnArg {
            name: arg.name.clone(),
            attribute: arg.attribute.clone(),
            span: arg.span,
            typ: Ty {
                kind: mono_info.typ.clone().expect("expected a type"),
                span: arg.typ.span,
            },
        })
        .collect();

    // evaluate generic return types using inferred values
    let ret_ty = match &fn_sig.return_type {
        Some(ret_ty) => match &ret_ty.kind {
            TyKind::GenericArray(typ, size) => {
                let val = size.eval(mono_fn_env);
                let tykind = TyKind::Array(typ.clone(), val);
                Some(Ty {
                    kind: tykind,
                    span: ret_ty.span,
                })
            }
            _ => Some(ret_ty.clone()),
        },
        None => None,
    };

    // construct the monomorphized function AST
    let func_def = match fn_info.kind {
        FnKind::BuiltIn(_, handle) => {
            let sig_typed = FnSig {
                arguments: fn_args_typed,
                return_type: ret_ty.clone(),
                ..fn_sig
            };
            FnInfo {
                kind: FnKind::BuiltIn(sig_typed, handle),
                span: fn_info.span,
            }
        }
        FnKind::Native(fn_def) => {
            let (stmts, typ) = monomorphize_block(ctx, mono_fn_env, &stmts, ret_ty.as_ref())?;

            let ret_typ = match typ {
                Some(t) => Some(Ty {
                    kind: t,
                    span: fn_def.sig.return_type.as_ref().unwrap().span,
                }),
                None => None,
            };

            FnInfo {
                kind: FnKind::Native(FunctionDef {
                    sig: FnSig {
                        arguments: fn_args_typed,
                        return_type: ret_typ,
                        ..fn_sig
                    },
                    body: stmts,
                    span: fn_def.span,
                }),
                span: fn_info.span,
            }
        }
    };

    Ok((func_def, ret_ty.map(|t| t.kind)))
}
pub fn error(kind: ErrorKind, span: Span) -> Error {
    Error::new("mast", kind, span)
}
