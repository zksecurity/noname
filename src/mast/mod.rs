use num_bigint::BigUint;
use std::collections::{HashMap, HashSet};

use crate::{
    backends::Backend,
    constants::Span,
    error::{Error, ErrorKind, Result},
    imports::FnKind,
    parser::{
        types::{
            FnArg, FnSig, GenericParameters, GenericValue, Range, Stmt, StmtKind, Symbolic, Ty,
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
#[derive(Debug, Default)]
pub struct MastCtx {
    pub last_node_id: usize,
}

impl MastCtx {
    pub fn new() -> Self {
        Self { last_node_id: 0 }
    }

    pub fn next_node_id(&mut self) -> usize {
        self.last_node_id += 1;
        self.last_node_id
    }
}

impl Symbolic {
    /// Evaluate symbolic size to an integer.
    pub fn eval(&self, typed_fn_env: &MonomorphizedFnEnv) -> u32 {
        match self {
            Symbolic::Concrete(v) => *v,
            Symbolic::Generic(g) => typed_fn_env.get_type_info(&g.value).unwrap().value.unwrap(),
            Symbolic::Add(a, b) => a.eval(typed_fn_env) + b.eval(typed_fn_env),
            Symbolic::Mul(a, b) => a.eval(typed_fn_env) * b.eval(typed_fn_env),
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
        self.0.insert(name, GenericValue::Unbound);
    }

    /// Bind a generic parameter to a value
    pub fn bind(&mut self, name: String, value: u32, span: Span) -> Result<()> {
        let existing = self.0.get(&name);
        match existing {
            Some(GenericValue::Bound(v)) => {
                if *v == value {
                    return Ok(());
                }

                Err(Error::new(
                    "mast",
                    ErrorKind::ConflictGenericValue(name, *v, value),
                    span,
                ))
            }
            Some(GenericValue::Unbound) => Ok(()),
            _ => Err(Error::new(
                "mast",
                ErrorKind::UnexpectedGenericParameter(name),
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

    ctx: MastCtx,

    /// Monomorphized AST of the main function
    pub main_fn_ast: Option<FunctionDef>,
}

impl<B: Backend> Mast<B> {
    pub fn new(tast: TypeChecker<B>) -> Self {
        Self {
            tast,
            node_types: HashMap::new(),
            node_functions: HashMap::new(),
            ctx: MastCtx::new(),
            main_fn_ast: None,
        }
    }

    pub fn expr_type(&self, expr: &Expr) -> Option<&TyKind> {
        self.node_types.get(&expr.node_id)
    }

    pub fn expr_fn(&self, expr: &Expr) -> Option<&FnInfo<B>> {
        self.node_functions.get(&expr.node_id)
    }

    pub fn struct_info(&self, qualified: &FullyQualified) -> Option<&StructInfo> {
        self.tast.struct_info(qualified)
    }

    pub fn const_info(&self, qualified: &FullyQualified) -> Option<&ConstInfo<B::Field>> {
        self.tast.const_info(qualified)
    }

    pub fn size_of(&self, typ: &TyKind) -> usize {
        self.tast.size_of(typ)
    }

    /// Monomorphize the main function.
    /// This is the entry point of the monomorphization process.
    /// It stores the monomorphized AST at the end.
    pub fn monomorphize(&mut self) -> Result<()> {
        // store mtype in mast
        let qualified = FullyQualified::local("main".to_string());
        let main_fn = self
            .tast
            .fn_info(&qualified)
            .ok_or(self.error(ErrorKind::NoMainFunction, Span::default()))?;

        let func_def = match &main_fn.kind {
            // `fn main() { ... }`
            FnKind::Native(function) => function.clone(),

            _ => panic!("main function must be native"),
        };

        // create a new typed fn environment to type check the function
        let mut typed_fn_env = MonomorphizedFnEnv::default();

        // store variables and their types in the fn_env
        for arg in &func_def.sig.arguments {
            // store the args' type in the fn environment
            typed_fn_env.store_type(
                &arg.name.value,
                &MTypeInfo::new(&arg.typ.kind, arg.span, None),
            )?;
        }

        // the output value returned by the main function is also a main_args with a special name (public_output)
        if let Some(typ) = &func_def.sig.return_type {
            match typ.kind {
                TyKind::Field => {
                    typed_fn_env.store_type(
                        "public_output".as_ref(),
                        &MTypeInfo::new(&typ.kind, typ.span, None),
                    )?;
                }
                TyKind::Array(_, _) => {
                    typed_fn_env.store_type(
                        "public_output".as_ref(),
                        &MTypeInfo::new(&typ.kind, typ.span, None),
                    )?;
                }
                _ => unimplemented!(),
            }
        }

        typed_fn_env.nest();
        // inferring for main function body
        let (stmts_mono, _) = self.monomorphize_block(
            &mut typed_fn_env,
            &func_def.body,
            func_def.sig.return_type.as_ref(),
        )?;
        typed_fn_env.pop();

        let mast = FunctionDef {
            sig: func_def.sig,
            body: stmts_mono,
            span: func_def.span,
        };

        self.main_fn_ast = Some(mast);

        Ok(())
    }

    /// Recursively monomorphize an expression node.
    /// It does two things:
    /// - Monomorphize the expression node with the inferred generic values.
    /// - Typecheck the resolved type.
    fn monomorphize_expr(
        &mut self,
        expr: &Expr,
        typed_fn_env: &mut MonomorphizedFnEnv,
    ) -> Result<ExprMonoInfo> {
        let expr_mono: ExprMonoInfo = match &expr.kind {
            ExprKind::FieldAccess { lhs, rhs } => {
                let lhs_mono = self.monomorphize_expr(lhs, typed_fn_env)?;

                // obtain the type of the field
                let (module, struct_name) = match lhs_mono.typ {
                    Some(TyKind::Custom { module, name }) => (module, name),
                    _ => panic!("field access must be done on a custom struct"),
                };

                // get struct info
                let qualified = FullyQualified::new(&module, &struct_name);
                let struct_info = self
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
                    &mut self.ctx,
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
                    let node = self.monomorphize_expr(arg, typed_fn_env)?;
                    observed.push((node, arg.span));
                }

                // retrieve the function signature
                let qualified = FullyQualified::new(module, &fn_name.value);
                let fn_info = self
                    .tast
                    .fn_info(&qualified)
                    .expect("function not found")
                    .to_owned();

                let args_mono = observed.clone().into_iter().map(|e| e.0.expr).collect();

                let mexpr = expr.to_mast(
                    &mut self.ctx,
                    &ExprKind::FnCall {
                        module: module.clone(),
                        fn_name: fn_name.clone(),
                        args: args_mono,
                    },
                );

                // instantiate the function call
                let (fn_info_mono, typ) =
                    self.instantiate_fn_call(fn_info, &observed, expr.span)?;

                self.node_functions.insert(mexpr.node_id, fn_info_mono);

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
                let lhs_mono = self.monomorphize_expr(lhs, typed_fn_env)?;
                let (module, struct_name) = match lhs_mono.clone().typ {
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

                // compute the observed arguments types
                let mut observed = Vec::with_capacity(args.len());
                if let Some(self_arg) = fn_info.sig().arguments.first() {
                    if self_arg.name.value == "self" {
                        observed.push((self.monomorphize_expr(lhs, typed_fn_env)?, lhs.span));
                    }
                }

                let mut args_mono = vec![];
                for arg in args {
                    let expr_mono = self.monomorphize_expr(arg, typed_fn_env)?;
                    observed.push((expr_mono.clone(), arg.span));
                    args_mono.push(expr_mono.expr);
                }

                let mexpr = expr.to_mast(
                    &mut self.ctx,
                    &ExprKind::MethodCall {
                        lhs: Box::new(lhs_mono.expr),
                        method_name: method_name.clone(),
                        args: args_mono,
                    },
                );

                // instantiate the function call
                let (fn_info_mono, typ) =
                    self.instantiate_fn_call(fn_info, &observed, expr.span)?;
                self.node_functions.insert(mexpr.node_id, fn_info_mono);

                // assume the function call won't return constant value
                ExprMonoInfo::new(mexpr, typ, None)
            }

            ExprKind::Assignment { lhs, rhs } => {
                // compute type of lhs
                let lhs_mono = self.monomorphize_expr(lhs, typed_fn_env)?;

                // and is of the same type as the rhs
                let rhs_mono = self.monomorphize_expr(rhs, typed_fn_env)?;

                if !rhs_mono.typ.unwrap().same_as(&lhs_mono.typ.unwrap()) {
                    panic!("lhs type doesn't match rhs type");
                }

                let mexpr = expr.to_mast(
                    &mut self.ctx,
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
                let lhs_mono = self.monomorphize_expr(lhs, typed_fn_env)?;
                let rhs_mono = self.monomorphize_expr(rhs, typed_fn_env)?;

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
                        let mexpr =
                            expr.to_mast(&mut self.ctx, &ExprKind::BigUInt(BigUint::from(v)));

                        ExprMonoInfo::new(mexpr, typ, Some(v))
                    }
                    None => {
                        let mexpr = expr.to_mast(
                            &mut self.ctx,
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
                let inner_mono = self.monomorphize_expr(inner, typed_fn_env)?;

                let mexpr =
                    expr.to_mast(&mut self.ctx, &ExprKind::Negated(Box::new(inner_mono.expr)));

                ExprMonoInfo::new(mexpr, Some(TyKind::Field), None)
            }

            ExprKind::Not(inner) => {
                let inner_mono = self.monomorphize_expr(inner, typed_fn_env)?;

                let mexpr = expr.to_mast(&mut self.ctx, &ExprKind::Not(Box::new(inner_mono.expr)));

                ExprMonoInfo::new(mexpr, Some(TyKind::Bool), None)
            }

            ExprKind::BigUInt(inner) => {
                let cst: u32 = inner.try_into().expect("biguint too large");
                let mexpr = expr.to_mast(&mut self.ctx, &ExprKind::BigUInt(inner.clone()));

                ExprMonoInfo::new(mexpr, Some(TyKind::BigInt), Some(cst))
            }

            ExprKind::Bool(inner) => {
                let mexpr = expr.to_mast(&mut self.ctx, &ExprKind::Bool(*inner));

                ExprMonoInfo::new(mexpr, Some(TyKind::Bool), None)
            }

            // mod::path.of.var
            // it could be also a generic variable
            ExprKind::Variable { module, name } => {
                let qualified = FullyQualified::new(module, &name.value);

                let res = if is_generic_parameter(&name.value) {
                    let mtype = typed_fn_env.get_type_info(&name.value).unwrap();
                    let mexpr = expr.to_mast(
                        &mut self.ctx,
                        &ExprKind::BigUInt(BigUint::from(mtype.value.unwrap())),
                    );

                    ExprMonoInfo::new(mexpr, Some(mtype.typ.clone()), mtype.value)
                } else if is_type(&name.value) {
                    let mtype = TyKind::Custom {
                        module: module.clone(),
                        name: name.value.clone(),
                    };

                    let mexpr = expr.to_mast(
                        &mut self.ctx,
                        &ExprKind::Variable {
                            module: module.clone(),
                            name: name.clone(),
                        },
                    );

                    ExprMonoInfo::new(mexpr, Some(mtype), None)
                } else if let Some(cst) = self.tast.const_info(&qualified) {
                    // if it's a variable,
                    // check if it's a constant first
                    let bigint: BigUint = cst.value[0].into();
                    let cst: u32 = bigint.clone().try_into().expect("biguint too large");
                    let mexpr = expr.to_mast(&mut self.ctx, &ExprKind::BigUInt(bigint));

                    ExprMonoInfo::new(mexpr, Some(TyKind::BigInt), Some(cst))
                } else {
                    // otherwise it's a local variable
                    let mexpr = expr.to_mast(
                        &mut self.ctx,
                        &ExprKind::Variable {
                            module: module.clone(),
                            name: name.clone(),
                        },
                    );

                    let mtype = typed_fn_env.get_type_info(&name.value).unwrap().clone();
                    ExprMonoInfo::new(mexpr, Some(mtype.typ), mtype.value)
                };

                res
            }

            ExprKind::ArrayAccess { array, idx } => {
                // get type of lhs
                let array_mono = self.monomorphize_expr(array, typed_fn_env)?;
                let id_mono = self.monomorphize_expr(idx, typed_fn_env)?;

                // get type of element
                let el_typ = match array_mono.typ.unwrap() {
                    TyKind::Array(typkind, _) => Some(*typkind),
                    _ => panic!("not an array"),
                };

                let mexpr = expr.to_mast(
                    &mut self.ctx,
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
                    let item_mono = self.monomorphize_expr(item, typed_fn_env)?;
                    items_mono.push(item_mono.clone());

                    if let Some(tykind) = &tykind {
                        if !tykind.same_as(&item_mono.clone().typ.unwrap()) {
                            return Err(self.error(
                                ErrorKind::MismatchType(tykind.clone(), item_mono.typ.unwrap()),
                                expr.span,
                            ));
                        }
                    } else {
                        tykind = item_mono.typ;
                    }
                }

                let mexpr = expr.to_mast(
                    &mut self.ctx,
                    &ExprKind::ArrayDeclaration(items_mono.into_iter().map(|e| e.expr).collect()),
                );

                let item_typ = tykind.expect("expected a value");

                let typ = TyKind::Array(Box::new(item_typ), len);
                ExprMonoInfo::new(mexpr, Some(typ), None)
            }

            ExprKind::IfElse { cond, then_, else_ } => {
                let cond_mono = self.monomorphize_expr(cond, typed_fn_env)?;

                // compute type of if/else branches
                let then_mono = self.monomorphize_expr(then_, typed_fn_env)?;
                let else_mono = self.monomorphize_expr(else_, typed_fn_env)?;

                // make sure that the type of then_ and else_ match
                if then_mono.typ != else_mono.typ {
                    panic!("`if` branch and `else` branch must have matching types");
                }

                let mexpr = expr.to_mast(
                    &mut self.ctx,
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
                let struct_info = self.tast.struct_info(&qualified).ok_or_else(|| {
                    self.error(ErrorKind::UndefinedStruct(name.clone()), expr.span)
                })?;

                let defined_fields = &struct_info.fields.clone();

                let mut fields_mono = vec![];

                for (defined, observed) in defined_fields.iter().zip(fields) {
                    let ident = observed.0.clone();
                    if defined.0 != ident.value {
                        return Err(self.error(
                            ErrorKind::InvalidStructField(defined.0.clone(), ident.value.clone()),
                            expr.span,
                        ));
                    }

                    let observed_mono = &self.monomorphize_expr(&observed.1, typed_fn_env)?;
                    let typ_mono = observed_mono.typ.as_ref().expect("expected a value");

                    if !typ_mono.same_as(&defined.1) {
                        return Err(self.error(
                            ErrorKind::InvalidStructFieldType(defined.1.clone(), typ_mono.clone()),
                            expr.span,
                        ));
                    }

                    fields_mono.push((ident, observed_mono.expr.clone()));
                }

                let mexpr = expr.to_mast(
                    &mut self.ctx,
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
            ExprKind::RepeatedArrayDeclaration { item, size } => {
                let item_mono = self.monomorphize_expr(item, typed_fn_env)?;
                let size_mono = self.monomorphize_expr(size, typed_fn_env)?;

                let item_typ = item_mono.typ.expect("expected a value");
                let mexpr = expr.to_mast(
                    &mut self.ctx,
                    &ExprKind::RepeatedArrayDeclaration {
                        item: Box::new(item_mono.expr),
                        size: Box::new(size_mono.expr),
                    },
                );

                if let Some(cst) = size_mono.constant {
                    let arr_typ = TyKind::Array(Box::new(item_typ), cst);
                    ExprMonoInfo::new(mexpr, Some(arr_typ), None)
                } else {
                    return Err(self.error(ErrorKind::InvalidArraySize, expr.span));
                }
            }
        };

        if let Some(typ) = &expr_mono.typ {
            self.node_types.insert(expr_mono.expr.node_id, typ.clone());
        }

        Ok(expr_mono)
    }

    /// Monomorphize a block of statements.
    pub fn monomorphize_block(
        &mut self,
        typed_fn_env: &mut MonomorphizedFnEnv,
        stmts: &[Stmt],
        expected_return: Option<&Ty>,
    ) -> Result<(Vec<Stmt>, Option<TyKind>)> {
        let mut return_typ = None;

        let mut stmts_mono = vec![];

        for stmt in stmts {
            if let Some((stmt, ret_typ)) = self.monomorphize_stmt(typed_fn_env, stmt)? {
                stmts_mono.push(stmt);

                if ret_typ.is_some() {
                    return_typ = ret_typ;
                }
            }
        }

        // check the return
        if let (Some(expected), Some(observed)) = (expected_return, return_typ.clone()) {
            if !observed.same_as(&expected.kind) {
                return Err(self.error(
                    ErrorKind::ReturnTypeMismatch(observed.clone(), expected.kind.clone()),
                    expected.span,
                ));
            }
        };

        Ok((stmts_mono, return_typ))
    }

    /// Monomorphize a statement.
    pub fn monomorphize_stmt(
        &mut self,
        typed_fn_env: &mut MonomorphizedFnEnv,
        stmt: &Stmt,
    ) -> Result<Option<(Stmt, Option<TyKind>)>> {
        let res = match &stmt.kind {
            StmtKind::Assign { mutable, lhs, rhs } => {
                let rhs_mono = self.monomorphize_expr(rhs, typed_fn_env)?;
                let typ = rhs_mono.typ.as_ref().expect("expected a type");
                let type_info = MTypeInfo::new(typ, lhs.span, None);

                // store the type of lhs in the env
                typed_fn_env.store_type(&lhs.value, &type_info)?;

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
                typed_fn_env.store_type(
                    &var.value,
                    // because we don't unroll the loop in the monomorphized AST
                    // no constant value even if it's a BigInt
                    &MTypeInfo::new(&TyKind::BigInt, var.span, None),
                )?;

                let start_mono = self.monomorphize_expr(&range.start, typed_fn_env)?;
                let end_mono = self.monomorphize_expr(&range.end, typed_fn_env)?;

                if start_mono.constant.is_none() || end_mono.constant.is_none() {
                    return Err(self.error(ErrorKind::InvalidRangeSize, stmt.span));
                }

                if start_mono.constant.unwrap() > end_mono.constant.unwrap() {
                    return Err(self.error(ErrorKind::InvalidRangeSize, stmt.span));
                }

                let range_mono = Range {
                    start: start_mono.expr,
                    end: end_mono.expr,
                    span: range.span,
                };

                let (stmts_mono, _) = self.monomorphize_block(typed_fn_env, body, None)?;
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
                let expr_mono = self.monomorphize_expr(expr, typed_fn_env)?;
                let stmt_mono = Stmt {
                    kind: StmtKind::Expr(Box::new(expr_mono.expr)),
                    span: stmt.span,
                };

                Some((stmt_mono, None))
            }
            StmtKind::Return(res) => {
                let expr_mono = self.monomorphize_expr(res, typed_fn_env)?;
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
    pub fn instantiate_fn_call(
        &mut self,
        fn_info: FnInfo<B>,
        args: &[(ExprMonoInfo, Span)],
        span: Span,
    ) -> Result<(FnInfo<B>, Option<TyKind>)> {
        let (fn_sig, stmts) = match &fn_info.kind {
            FnKind::BuiltIn(sig, _) => (sig, Vec::<Stmt>::new()),
            FnKind::Native(func) => (&func.sig, func.body.clone()),
        };

        // canonicalize the arguments depending on method call or not
        let expected: Vec<_> = fn_sig.arguments.iter().collect();

        // check argument length
        if expected.len() != args.len() {
            return Err(self.error(
                ErrorKind::MismatchFunctionArguments(args.len(), expected.len()),
                span,
            ));
        }

        // create a context for the function call
        let typed_fn_env = &mut MonomorphizedFnEnv::new();

        // to bind the generic values
        let mut generics = fn_sig.generics.clone();

        // infer the generic values from the observed types
        for (sig_arg, (type_info, span)) in expected.iter().zip(args) {
            let arg_name = &sig_arg.name.value;
            match &sig_arg.typ.kind {
                TyKind::GenericArray(_, sym_size) => {
                    let gen_name = match sym_size {
                        Symbolic::Generic(g) => &g.value,
                        _ => panic!("only allow a single generic parameter for an argument"),
                    };

                    // infer the array size from the observed type
                    let arr_type = type_info.typ.as_ref().expect("expected a value");
                    let size = match arr_type {
                        TyKind::Array(_, size) => size,
                        _ => panic!("expected array type"),
                    };

                    // bind generic value
                    generics.bind(gen_name.to_string(), *size, sig_arg.span)?;

                    // store the inferred value for generic parameter
                    let gen_mty = MTypeInfo::new(&TyKind::BigInt, *span, Some(*size));
                    typed_fn_env.store_type(gen_name, &gen_mty)?;

                    // store concrete array type for the argument name
                    let arr_mty = MTypeInfo::new(arr_type, *span, None);
                    typed_fn_env.store_type(arg_name, &arr_mty)?;
                }
                _ => {
                    let typ = type_info.typ.as_ref().expect("expected a value");
                    let cst = type_info.constant;

                    if is_generic_parameter(arg_name) {
                        if cst.is_none() {
                            return Err(self.error(
                                ErrorKind::GenericValueExpected(arg_name.to_string()),
                                *span,
                            ));
                        }

                        // bind generic value
                        generics.bind(arg_name.to_string(), cst.unwrap(), sig_arg.span)?;
                    }

                    // store the type of the argument in the env
                    typed_fn_env.store_type(arg_name, &MTypeInfo::new(typ, *span, cst))?;
                }
            }
        }

        // reconstruct FnArgs using the observed types
        let fn_args_typed = expected
            .iter()
            .zip(args)
            .map(|(arg, (mono_info, _))| FnArg {
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
                    let val = size.eval(typed_fn_env);
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
            FnKind::BuiltIn(sig, handle) => {
                let sig_typed = FnSig {
                    name: sig.name.clone(),
                    kind: sig.kind.clone(),
                    generics,
                    arguments: fn_args_typed,
                    return_type: ret_ty.clone(),
                };
                FnInfo {
                    kind: FnKind::BuiltIn(sig_typed, handle),
                    span: fn_info.span,
                }
            }
            FnKind::Native(fn_def) => {
                let (stmts, typ) =
                    self.monomorphize_block(typed_fn_env, &stmts, ret_ty.as_ref())?;

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
                            name: fn_def.sig.name.clone(),
                            kind: fn_def.sig.kind.clone(),
                            generics,
                            arguments: fn_args_typed,
                            return_type: ret_typ,
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

    pub fn error(&self, kind: ErrorKind, span: Span) -> Error {
        Error::new("mast", kind, span)
    }
}
