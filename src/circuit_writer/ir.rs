use ark_ff::{One, PrimeField, Zero};
use circ::{
    ir::term::{
        leaf_term, precomp::PreComp, term, BoolNaryOp, BvBinOp, IntBinOp, IntBinPred, IntNaryOp,
        Op, PfNaryOp, PfUnOp, Sort, Term, Value,
    },
    term,
};
use fxhash::FxHashMap;
use kimchi::o1_utils::FieldHelpers;
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{
    backends::{Backend, BackendField},
    constants::Span,
    error::{Error, ErrorKind, Result},
    imports::FnKind,
    mast::PropagatedConstant,
    parser::{
        types::{ForLoopArgument, FunctionDef, Stmt, StmtKind, TyKind},
        Expr, ExprKind, Op2,
    },
    syntax::is_type,
    type_checker::{ConstInfo, FnInfo, FullyQualified, StructInfo, TypeChecker},
};

/// Same as [crate::var::Var], but with Term instead of ConstOrCell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Var {
    pub cvars: Vec<Term>,

    pub span: Span,
}

impl Var {
    pub fn new(cvars: Vec<Term>, span: Span) -> Self {
        Self { cvars, span }
    }

    pub fn new_cvar(cvar: Term, span: Span) -> Self {
        Self {
            cvars: vec![cvar],
            span,
        }
    }

    pub fn new_var(cvar: Term, span: Span) -> Self {
        Self {
            cvars: vec![cvar],
            span,
        }
    }

    pub fn new_constant<F: BackendField>(cst: F, span: Span) -> Self {
        let cvar = leaf_term(Op::new_const(Value::Field(cst.to_circ_field())));

        Self {
            cvars: vec![cvar],
            span,
        }
    }

    pub fn new_bool(b: bool, span: Span) -> Self {
        let cvar = leaf_term(Op::new_const(Value::Bool(b)));

        Self {
            cvars: vec![cvar],
            span,
        }
    }

    pub fn new_constant_typ<F: BackendField>(cst_info: &ConstInfo<F>, span: Span) -> Self {
        let ConstInfo { value, typ: _ } = cst_info;
        let cvars = value
            .iter()
            .cloned()
            .map(|f| leaf_term(Op::new_const(Value::Field(f.to_circ_field()))))
            .collect();

        Self { cvars, span }
    }

    pub fn len(&self) -> usize {
        self.cvars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cvars.is_empty()
    }

    pub fn get(&self, idx: usize) -> Option<&Term> {
        if idx < self.cvars.len() {
            Some(&self.cvars[idx])
        } else {
            None
        }
    }

    pub fn constant<B: Backend>(&self) -> Option<BigUint> {
        if self.cvars.len() == 1 {
            let env = fxhash::FxHashMap::default();
            Some(IRWriter::<B>::eval_ir(&env, &self.cvars[0])[0].to_biguint())
        } else {
            None
        }
    }

    pub fn range(&self, start: usize, len: usize) -> &[Term] {
        &self.cvars[start..(start + len)]
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Term> {
        self.cvars.iter()
    }
}

/// Same as [crate::var::VarInfo], but with Term based Var.
pub enum VarOrRef {
    Var(Var),

    Ref {
        var_name: String,
        start: usize,
        len: usize,
    },
}

impl VarOrRef {
    pub(crate) fn constant<B: Backend>(&self) -> Option<BigUint> {
        match self {
            VarOrRef::Var(var) => var.constant::<B>(),
            VarOrRef::Ref { .. } => None,
        }
    }

    pub(crate) fn value<B: Backend>(self, ir_writer: &IRWriter<B>, fn_env: &FnEnv) -> Var {
        match self {
            VarOrRef::Var(var) => var,
            VarOrRef::Ref {
                var_name,
                start,
                len,
            } => {
                let var_info = ir_writer.get_local_var(fn_env, &var_name);
                let cvars = var_info.var.range(start, len).to_vec();
                Var::new(cvars, var_info.var.span)
            }
        }
    }

    pub(crate) fn from_var_info(var_name: String, var_info: VarInfo) -> Self {
        if var_info.mutable {
            Self::Ref {
                var_name,
                start: 0,
                len: var_info.var.len(),
            }
        } else {
            Self::Var(var_info.var)
        }
    }

    pub(crate) fn narrow(&self, start: usize, len: usize) -> Self {
        match self {
            VarOrRef::Var(var) => {
                let cvars = var.range(start, len).to_vec();
                VarOrRef::Var(Var::new(cvars, var.span))
            }

            //      old_start
            //      |
            //      v
            // |----[-----------]-----| <-- var.cvars
            //       <--------->
            //         old_len
            //
            //
            //          start
            //          |
            //          v
            //      |---[-----]-|
            //           <--->
            //            len
            //
            VarOrRef::Ref {
                var_name,
                start: old_start,
                len: old_len,
            } => {
                // ensure that the new range is contained in the older range
                assert!(start < *old_len); // lower bound
                assert!(start + len <= *old_len); // upper bound
                assert!(len > 0); // empty range not allowed

                Self::Ref {
                    var_name: var_name.clone(),
                    start: old_start + start,
                    len,
                }
            }
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            VarOrRef::Var(var) => var.len(),
            VarOrRef::Ref { len, .. } => *len,
        }
    }
}

/// Same as [crate::fn_env::VarInfo], but with Term based Var.
/// Information about a variable.
#[derive(Debug, Clone)]
pub struct VarInfo {
    /// The variable.
    pub var: Var,

    pub mutable: bool,

    /// We keep track of the type of variables, eventhough we're not in the typechecker anymore,
    /// because we need to know the type for method calls.
    pub typ: Option<TyKind>,
}

impl VarInfo {
    pub fn new(var: Var, mutable: bool, typ: Option<TyKind>) -> Self {
        Self { var, mutable, typ }
    }

    pub fn reassign(&self, var: Var) -> Self {
        Self {
            var,
            mutable: self.mutable,
            typ: self.typ.clone(),
        }
    }

    pub fn reassign_range(&self, var: Var, start: usize, len: usize) -> Self {
        // sanity check
        assert_eq!(var.len(), len);

        // create new cvars by modifying a specific range
        let mut cvars = self.var.cvars.clone();
        let cvars_range = &mut cvars[start..start + len];
        cvars_range.clone_from_slice(&var.cvars);

        let var = Var::new(cvars, self.var.span);

        Self {
            var,
            mutable: self.mutable,
            typ: self.typ.clone(),
        }
    }
}

/// Same as [crate::fn_env::FnEnv], but with Term based VarInfo.
#[derive(Default, Debug, Clone)]
pub struct FnEnv {
    current_scope: usize,

    vars: HashMap<String, (usize, VarInfo)>,
}

impl FnEnv {
    /// Creates a new FnEnv
    pub fn new() -> Self {
        Self {
            current_scope: 0,
            vars: HashMap::new(),
        }
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
    fn is_in_scope(&self, prefix_scope: usize) -> bool {
        self.current_scope >= prefix_scope
    }

    /// Stores type information about a local variable.
    /// Note that we forbid shadowing at all scopes.
    pub fn add_local_var(&mut self, var_name: String, var_info: VarInfo) {
        let scope = self.current_scope;

        if self
            .vars
            .insert(var_name.clone(), (scope, var_info))
            .is_some()
        {
            panic!("type checker error: var `{var_name}` already exists");
        }
    }

    /// Retrieves type information on a variable, given a name.
    /// If the variable is not in scope, return false.
    pub fn get_local_var(&self, var_name: &str) -> VarInfo {
        let (scope, var_info) = self
            .vars
            .get(var_name)
            .unwrap_or_else(|| panic!("type checking bug: local variable `{var_name}` not found"));
        if !self.is_in_scope(*scope) {
            panic!("type checking bug: local variable `{var_name}` not in scope");
        }

        var_info.clone()
    }

    pub fn reassign_local_var(&mut self, var_name: &str, var: Var) {
        // get the scope first, we don't want to modify that
        let (scope, var_info) = self
            .vars
            .get(var_name)
            .expect("type checking bug: local variable for reassigning not found");

        if !self.is_in_scope(*scope) {
            panic!("type checking bug: local variable for reassigning not in scope");
        }

        if !var_info.mutable {
            panic!("type checking bug: local variable for reassigning is not mutable");
        }

        let var_info = var_info.reassign(var);
        self.vars.insert(var_name.to_string(), (*scope, var_info));
    }

    /// Same as [Self::reassign_var], but only reassigns a specific range of the variable.
    pub fn reassign_var_range(&mut self, var_name: &str, var: Var, start: usize, len: usize) {
        // get the scope first, we don't want to modify that
        let (scope, var_info) = self
            .vars
            .get(var_name)
            .expect("type checking bug: local variable for reassigning not found");

        if !self.is_in_scope(*scope) {
            panic!("type checking bug: local variable for reassigning not in scope");
        }

        if !var_info.mutable {
            panic!("type checking bug: local variable for reassigning is not mutable");
        }

        let var_info = var_info.reassign_range(var, start, len);
        self.vars.insert(var_name.to_string(), (*scope, var_info));
    }
}

#[derive(Debug)]
/// This converts the MAST to circ IR.
/// Currently it is only for hint functions.
pub struct IRWriter<B: Backend> {
    pub typed: TypeChecker<B>,
    pub logs: Vec<VarInfo>,
}

impl<B: Backend> IRWriter<B> {
    /// Same as circuit_writer::compile_stmt
    fn compile_stmt(&mut self, fn_env: &mut FnEnv, stmt: &Stmt) -> Result<Option<VarOrRef>> {
        match &stmt.kind {
            StmtKind::Assign { mutable, lhs, rhs } => {
                // compute the rhs
                let rhs_var = self
                    .compute_expr(fn_env, rhs)?
                    .ok_or_else(|| self.error(ErrorKind::CannotComputeExpression, stmt.span))?;

                // obtain the actual values
                let rhs_var = rhs_var.value(self, fn_env);

                let typ = self.expr_type(rhs).cloned();
                let var_info = VarInfo::new(rhs_var, *mutable, typ);

                // store the new variable
                self.add_local_var(fn_env, lhs.value.clone(), var_info)?;
            }

            StmtKind::ForLoop {
                var,
                argument,
                body,
            } => {
                match argument {
                    ForLoopArgument::Range(range) => {
                        // compute the start and end of the range
                        let start_bg: BigUint = self
                            .compute_expr(fn_env, &range.start)?
                            .ok_or_else(|| {
                                self.error(ErrorKind::CannotComputeExpression, range.start.span)
                            })?
                            .constant::<B>()
                            .expect("expected constant");
                        let start: u32 = start_bg.try_into().map_err(|_| {
                            self.error(ErrorKind::InvalidRangeSize, range.start.span)
                        })?;

                        let end_bg: BigUint = self
                            .compute_expr(fn_env, &range.end)?
                            .ok_or_else(|| {
                                self.error(ErrorKind::CannotComputeExpression, range.end.span)
                            })?
                            .constant::<B>()
                            .expect("expected constant");
                        let end: u32 = end_bg
                            .try_into()
                            .map_err(|_| self.error(ErrorKind::InvalidRangeSize, range.end.span))?;

                        // compute for the for loop block
                        for ii in start..end {
                            fn_env.nest();

                            let cst_var = Var::new_constant(B::Field::from(ii), var.span);
                            let var_info = VarInfo::new(
                                cst_var,
                                false,
                                Some(TyKind::Field { constant: true }),
                            );
                            self.add_local_var(fn_env, var.value.clone(), var_info)?;

                            self.compile_block(fn_env, body)?;

                            fn_env.pop();
                        }
                    }
                    ForLoopArgument::Iterator(iterator) => {
                        let iterator_var = self
                            .compute_expr(fn_env, iterator)?
                            .expect("array access on non-array");

                        let array_typ = self
                            .expr_type(iterator)
                            .cloned()
                            .expect("cannot find type of array");

                        let (elem_type, array_len) = match array_typ {
                            TyKind::Array(ty, array_len) => (ty, array_len),
                            _ => Err(Error::new(
                                "compile-stmt",
                                ErrorKind::UnexpectedError("expected array"),
                                stmt.span,
                            ))?,
                        };

                        // compute the size of each element in the array
                        let len = self.size_of(&elem_type);

                        for idx in 0..array_len {
                            // compute the real index
                            let idx = idx as usize;
                            let start = idx * len;

                            fn_env.nest();

                            // add the variable to the inner enviroment corresponding
                            // to iterator[idx]
                            let indexed_var = iterator_var.narrow(start, len).value(self, fn_env);
                            let var_info =
                                VarInfo::new(indexed_var.clone(), false, Some(*elem_type.clone()));
                            self.add_local_var(fn_env, var.value.clone(), var_info)?;

                            self.compile_block(fn_env, body)?;

                            fn_env.pop();
                        }
                    }
                }
            }
            StmtKind::Expr(expr) => {
                // compute the expression
                let var = self.compute_expr(fn_env, expr)?;

                // make sure it does not return any value.
                assert!(var.is_none());
            }
            StmtKind::Return(expr) => {
                let var = self
                    .compute_expr(fn_env, expr)?
                    .ok_or_else(|| self.error(ErrorKind::CannotComputeExpression, stmt.span))?;

                // we already checked in type checking that this is not an early return
                return Ok(Some(var));
            }
            StmtKind::Comment(_) => (),
        }

        Ok(None)
    }

    /// Same as circuit_writer::compile_block
    /// might return something?
    fn compile_block(&mut self, fn_env: &mut FnEnv, stmts: &[Stmt]) -> Result<Option<Var>> {
        fn_env.nest();
        for stmt in stmts {
            let res = self.compile_stmt(fn_env, stmt)?;
            if let Some(var) = res {
                // a block doesn't return a pointer, only values
                let var = var.value(self, fn_env);

                // we already checked for early returns in type checking
                return Ok(Some(var));
            }
        }
        fn_env.pop();
        Ok(None)
    }

    pub fn compile_hint_function_call(
        &mut self,
        function: &FunctionDef,
        args: Vec<crate::circuit_writer::fn_env::VarInfo<B::Field, B::Var>>,
    ) -> Result<Vec<crate::var::Value<B>>> {
        assert!(!function.is_main());

        // create new fn_env
        let fn_env = &mut FnEnv::new();

        // set arguments
        assert_eq!(function.sig.arguments.len(), args.len());

        // create circ var terms for the arguments
        let mut named_args = vec![];
        for (arg, observed) in function.sig.arguments.iter().zip(args) {
            let name = &arg.name.value;
            // create a list of terms corresponding to the observed
            let cvars = observed.var.cvars.iter().enumerate().map(|(i, v)| {
                // internal var name for IR
                let name = format!("{}_{}", name, i);
                // map between circ IR variables and noname [ConstOrCell]
                named_args.push((name.clone(), v.clone()));

                match v {
                    crate::var::ConstOrCell::Const(cst) => {
                        leaf_term(Op::new_const(Value::Field(cst.to_circ_field())))
                    }
                    crate::var::ConstOrCell::Cell(_) => leaf_term(Op::new_var(
                        name.clone(),
                        Sort::Field(B::Field::to_circ_type()),
                    )),
                }
            });

            let var = Var::new(cvars.collect(), observed.var.span);

            // add as local var
            let var_info = VarInfo::new(var, false, Some(arg.typ.kind.clone()));
            self.add_local_var(fn_env, name.clone(), var_info)?;
        }

        // compile it and potentially return a return value
        let ir = self.compile_block(fn_env, &function.body)?;
        if ir.is_none() {
            return Ok(vec![]);
        }

        let logs = self.logs.clone();
        let logs_terms: Vec<Term> = logs
            .into_iter()
            .map(|v| term(Op::Tuple, v.var.cvars))
            .collect();

        self.logs.clear();

        let res = ir.unwrap().cvars.into_iter().enumerate().map(|(i, v)| {
            // With the current setup to calculate symbolic values, the [compute_val] can only compute for one symbolic variable,
            // it has to evaluate each symbolic variable separately from a hint function.
            // Thus, this could introduce some performance overhead if the hint returns multiple symbolic variables.
            // Maybe this can be batched and cached in the [compute_val] function.

            // Each compiled IR can contain multiple terms, as hint function output could be array or struct.
            // Each term needs to be evaluated separately.
            // For logs, there could be multiple logs for a compiled hint function.
            // To avoid redundant logs, here we only evaluate log terms once with the first term.
            if i == 0 {
                crate::var::Value::HintIR(v, named_args.clone(), logs_terms.clone())
            } else {
                crate::var::Value::HintIR(v, named_args.clone(), Vec::new())
            }
        });

        Ok(res.collect())
    }

    /// Evaluate a single IR term.
    pub fn eval_ir(
        env: &FxHashMap<String, circ::ir::term::Value>,
        t: &circ::ir::term::Term,
    ) -> Vec<B::Field> {
        let mut precomp = PreComp::new();
        // For hint evaluation purpose, precomp only has only one output and no connections with other parts,
        // so just use a dummy output var name.
        precomp.add_output("x".to_string(), t.clone());
        // evaluate and get the only one output
        let eval_map = precomp.eval(env);
        let value = eval_map.get("x").unwrap();
        // convert to field
        match value {
            circ::ir::term::Value::Field(f) => {
                let bytes = f.i().to_digits::<u8>(rug::integer::Order::Lsf);
                // todo: should we allow field overflow in hint evaluation?
                vec![B::Field::from_le_bytes_mod_order(&bytes)]
            }
            circ::ir::term::Value::Bool(b) => {
                if *b {
                    vec![B::Field::one()]
                } else {
                    vec![B::Field::zero()]
                }
            }
            circ::ir::term::Value::BitVector(bv) => {
                let bytes = bv.uint().to_digits::<u8>(rug::integer::Order::Lsf);
                // todo: should we allow field overflow in hint evaluation?
                vec![B::Field::from_le_bytes_mod_order(&bytes)]
            }
            circ::ir::term::Value::Int(int) => {
                let bytes = int.to_digits::<u8>(rug::integer::Order::Lsf);
                vec![B::Field::from_le_bytes_mod_order(&bytes)]
            }
            circ::ir::term::Value::Tuple(v) => {
                let mut res = Vec::new();
                for v in v {
                    match v {
                        circ::ir::term::Value::Field(f) => {
                            let bytes = f.i().to_digits::<u8>(rug::integer::Order::Lsf);
                            res.push(B::Field::from_le_bytes_mod_order(&bytes));
                        }
                        circ::ir::term::Value::Bool(b) => {
                            if *b {
                                res.push(B::Field::one());
                            } else {
                                res.push(B::Field::zero());
                            }
                        }
                        circ::ir::term::Value::BitVector(bv) => {
                            let bytes = bv.uint().to_digits::<u8>(rug::integer::Order::Lsf);
                            res.push(B::Field::from_le_bytes_mod_order(&bytes));
                        }
                        circ::ir::term::Value::Int(int) => {
                            let bytes = int.to_digits::<u8>(rug::integer::Order::Lsf);
                            res.push(B::Field::from_le_bytes_mod_order(&bytes));
                        }
                        circ::ir::term::Value::Tuple(_) => {
                            panic!("nested tuple is not supported");
                        }
                        _ => panic!("unexpected output type"),
                    }
                }
                res
            }
            _ => panic!("unexpected output type"),
        }
    }

    /// This is used in MAST phase to fold constant values.
    pub fn evaluate(
        &mut self,
        function: &FunctionDef,
        args: Vec<PropagatedConstant>,
    ) -> Result<PropagatedConstant> {
        assert!(!function.is_main());

        // create new fn_env
        let fn_env = &mut FnEnv::new();

        // set arguments
        assert_eq!(function.sig.arguments.len(), args.len());

        // create circ var terms for the arguments
        for (arg, observed) in function.sig.arguments.iter().zip(args) {
            let name = &arg.name.value;
            match observed {
                PropagatedConstant::Single(cst) => {
                    let f = B::Field::from(cst);
                    let cvar = leaf_term(Op::new_const(Value::Field(f.to_circ_field())));
                    let var = Var::new(vec![cvar], arg.name.span);
                    let var_info = VarInfo::new(var, false, Some(arg.typ.kind.clone()));
                    self.add_local_var(fn_env, name.clone(), var_info).unwrap();
                }
                _ => unimplemented!(),
            }
        }

        // compile it and potentially return a return value
        let ir = self.compile_block(fn_env, &function.body)?;

        let res: Vec<_> = ir
            .unwrap()
            .cvars
            .into_iter()
            .flat_map(|f| {
                // because all the arguments are assumed to be constants,
                // so no need to pass the arguments in env
                let env = fxhash::FxHashMap::default();
                Self::eval_ir(&env, &f)
            })
            .collect();

        Ok(PropagatedConstant::from(res[0].to_biguint()))
    }

    fn compile_native_function_call(
        &mut self,
        function: &FunctionDef,
        args: Vec<VarInfo>,
    ) -> Result<Option<Var>> {
        assert!(!function.is_main());

        // create new fn_env
        let fn_env = &mut FnEnv::new();

        // set arguments
        assert_eq!(function.sig.arguments.len(), args.len());

        for (name, var_info) in function.sig.arguments.iter().zip(args) {
            self.add_local_var(fn_env, name.name.value.clone(), var_info)?;
        }

        // compile it and potentially return a return value
        self.compile_block(fn_env, &function.body)
    }

    fn compute_expr(&mut self, fn_env: &mut FnEnv, expr: &Expr) -> Result<Option<VarOrRef>> {
        match &expr.kind {
            // `module::fn_name(args)`
            ExprKind::FnCall {
                module,
                fn_name,
                args,
                ..
            } => {
                // sanity check
                if fn_name.value == "main" {
                    Err(self.error(ErrorKind::RecursiveMain, expr.span))?
                }

                // retrieve the function in the env
                let qualified = FullyQualified::new(module, &fn_name.value);
                let fn_info = self
                    .fn_info(&qualified)
                    .ok_or_else(|| {
                        self.error(
                            ErrorKind::UndefinedFunction(fn_name.value.clone()),
                            fn_name.span,
                        )
                    })?
                    .clone();

                // compute the arguments
                // module::fn_name(args)
                //                 ^^^^
                let mut vars = Vec::with_capacity(args.len());
                for arg in args {
                    // get the variable behind the expression
                    let var = self
                        .compute_expr(fn_env, arg)?
                        .ok_or_else(|| self.error(ErrorKind::CannotComputeExpression, arg.span))?;

                    // we pass variables by values always
                    let var = var.value(self, fn_env);

                    let typ = self.expr_type(arg).cloned();
                    let mutable = false;
                    let var_info = VarInfo::new(var, mutable, typ);

                    vars.push(var_info);
                }

                match &fn_info.kind {
                    // assert() <-- for example
                    FnKind::BuiltIn(sig, ..) => {
                        if sig.name.value == "log" {
                            self.logs.push(vars[0].clone());

                            Ok(None)
                        } else {
                            Err(self.error(
                                ErrorKind::InvalidFnCall(
                                    "builtin functions not allowed in hint functions.",
                                ),
                                expr.span,
                            ))
                        }
                    }
                    // fn_name(args)
                    // ^^^^^^^
                    FnKind::Native(func) => {
                        // module::fn_name(args)
                        // ^^^^^^
                        // only allow calling hint functions
                        if !func.is_hint {
                            return Err(self.error(
                                ErrorKind::InvalidFnCall("only hint functions allowed"),
                                expr.span,
                            ));
                        }

                        self.compile_native_function_call(func, vars)
                            .map(|r| r.map(VarOrRef::Var))
                    }
                }
            }

            ExprKind::FieldAccess { lhs, rhs } => {
                // get var behind lhs
                let lhs_var = self
                    .compute_expr(fn_env, lhs)?
                    .ok_or_else(|| self.error(ErrorKind::CannotComputeExpression, lhs.span))?;

                // get struct info behind lhs
                let lhs_struct = self
                    .expr_type(lhs)
                    .ok_or_else(|| self.error(ErrorKind::CannotComputeExpression, lhs.span))?;

                let (module, self_struct) = match lhs_struct {
                    TyKind::Custom { module, name } => (module, name),
                    _ => Err(Error::new(
                        "compute-expr",
                        ErrorKind::UnexpectedError(
                            "could not figure out struct implementing that method call",
                        ),
                        lhs.span,
                    ))?,
                };

                let qualified = FullyQualified::new(module, self_struct);
                let struct_info = self
                    .struct_info(&qualified)
                    .expect("struct info not found for custom struct");

                // find range of field
                let mut start = 0;
                let mut len = 0;
                for (field, field_typ, _attribute) in &struct_info.fields {
                    if field == &rhs.value {
                        len = self.size_of(field_typ);
                        break;
                    }

                    start += self.size_of(field_typ);
                }

                // narrow the variable to the given range
                let var = lhs_var.narrow(start, len);
                Ok(Some(var))
            }

            // `Thing.method(args)` or `thing.method(args)`
            ExprKind::MethodCall {
                lhs,
                method_name,
                args,
            } => {
                // figure out the name of the custom struct
                let lhs_typ = self.expr_type(lhs).expect("method call on what?").clone();

                let (module, struct_name) = match &lhs_typ {
                    TyKind::Custom { module, name } => (module, name),
                    _ => {
                        return Err(self.error(
                            ErrorKind::UnexpectedError("method call only work on custom types"),
                            lhs.span,
                        ))
                    }
                };

                // get var of `self`
                // (might be `None` if it's a static method call)
                let self_var = self.compute_expr(fn_env, lhs)?;

                // find method info
                let qualified = FullyQualified::new(module, struct_name);
                let struct_info = self
                    .struct_info(&qualified)
                    .ok_or(self.error(
                        ErrorKind::UnexpectedError("struct not found"),
                        method_name.span,
                    ))?
                    .clone();
                let func = struct_info
                    .methods
                    .get(&method_name.value)
                    .expect("could not find method");

                // if method has a `self` argument, manually add it to the list of argument
                let mut vars = vec![];
                if let Some(first_arg) = func.sig.arguments.first() {
                    if first_arg.name.value == "self" {
                        let self_var = self_var.ok_or_else(|| {
                            self.error(ErrorKind::NotAStaticMethod, method_name.span)
                        })?;

                        let mutable = false;
                        let self_var = self_var.value(self, fn_env);

                        let self_var_info = VarInfo::new(self_var, mutable, Some(lhs_typ.clone()));
                        vars.insert(0, self_var_info);
                    }
                } else {
                    assert!(self_var.is_none());
                }

                // compute the arguments
                for arg in args {
                    let var = self
                        .compute_expr(fn_env, arg)?
                        .ok_or_else(|| self.error(ErrorKind::CannotComputeExpression, arg.span))?;

                    let mutable = false;
                    let var = var.value(self, fn_env);

                    let typ = self.expr_type(arg).cloned();
                    let var_info = VarInfo::new(var, mutable, typ);

                    vars.push(var_info);
                }

                // execute method
                self.compile_native_function_call(func, vars)
                    .map(|r| r.map(VarOrRef::Var))
            }

            ExprKind::IfElse { cond, then_, else_ } => {
                let cond = self
                    .compute_expr(fn_env, cond)?
                    .unwrap()
                    .value(self, fn_env)
                    .cvars[0]
                    .clone();
                let then_ = self
                    .compute_expr(fn_env, then_)?
                    .unwrap()
                    .value(self, fn_env)
                    .cvars[0]
                    .clone();
                let else_ = self
                    .compute_expr(fn_env, else_)?
                    .unwrap()
                    .value(self, fn_env)
                    .cvars[0]
                    .clone();

                let ite_ir = term![Op::Ite; cond, then_, else_];

                let res = Var::new_cvar(ite_ir, expr.span);
                Ok(Some(VarOrRef::Var(res)))
            }

            ExprKind::Assignment { lhs, rhs } => {
                // figure out the local var  of lhs
                let lhs = self.compute_expr(fn_env, lhs)?.unwrap();

                // figure out the var of what's on the right
                let rhs = self.compute_expr(fn_env, rhs)?.unwrap();
                let rhs_var = match rhs {
                    VarOrRef::Var(var) => var,
                    VarOrRef::Ref {
                        var_name,
                        start,
                        len,
                    } => {
                        let var_info = self.get_local_var(fn_env, &var_name);
                        let cvars = var_info.var.range(start, len).to_vec();
                        Var::new(cvars, var_info.var.span)
                    }
                };

                // replace the left with the right
                match lhs {
                    VarOrRef::Var(_) => Err(Error::new(
                        "compute-expr",
                        ErrorKind::UnexpectedError("can't reassign this non-mutable variable"),
                        expr.span,
                    ))?,
                    VarOrRef::Ref {
                        var_name,
                        start,
                        len,
                    } => {
                        fn_env.reassign_var_range(&var_name, rhs_var, start, len);
                    }
                }

                Ok(None)
            }

            ExprKind::BinaryOp { op, lhs, rhs, .. } => {
                let lhs = self.compute_expr(fn_env, lhs)?.unwrap();
                let rhs = self.compute_expr(fn_env, rhs)?.unwrap();

                let lhs = lhs.value(self, fn_env);
                let rhs = rhs.value(self, fn_env);

                let res = match op {
                    Op2::Addition => {
                        let t: Term = term![Op::PfNaryOp(PfNaryOp::Add); lhs.cvars[0].clone(), rhs.cvars[0].clone()];
                        Var::new_cvar(t, expr.span)
                    }
                    Op2::Subtraction => {
                        let t: Term = term![Op::PfNaryOp(PfNaryOp::Add); lhs.cvars[0].clone(), term![Op::PfUnOp(PfUnOp::Neg); rhs.cvars[0].clone()]];
                        Var::new_cvar(t, expr.span)
                    }
                    Op2::Multiplication => {
                        let t: Term = term![Op::PfNaryOp(PfNaryOp::Mul); lhs.cvars[0].clone(), rhs.cvars[0].clone()];
                        Var::new_cvar(t, expr.span)
                    }
                    Op2::Equality => {
                        let t: Term = term![Op::Eq; lhs.cvars[0].clone(), rhs.cvars[0].clone()];
                        Var::new_cvar(t, expr.span)
                    }
                    Op2::Inequality => {
                        let t: Term = term![Op::Not; term![Op::Eq; lhs.cvars[0].clone(), rhs.cvars[0].clone()]];
                        Var::new_cvar(t, expr.span)
                    }
                    Op2::BoolAnd => {
                        let t: Term = term![Op::BoolNaryOp(BoolNaryOp::And); lhs.cvars[0].clone(), rhs.cvars[0].clone()];
                        Var::new_cvar(t, expr.span)
                    }
                    Op2::BoolOr => {
                        let t: Term = term![Op::BoolNaryOp(BoolNaryOp::Or); lhs.cvars[0].clone(), rhs.cvars[0].clone()];
                        Var::new_cvar(t, expr.span)
                    }
                    Op2::Division => {
                        // convert to int
                        let a_int = term![Op::PfToInt; lhs.cvars[0].clone()];
                        let b_int = term![Op::PfToInt; rhs.cvars[0].clone()];
                        // division
                        let t = term![Op::IntBinOp(IntBinOp::Div); a_int, b_int];

                        // convert back to field
                        let t = term![Op::IntToPf(B::Field::to_circ_type()); t];

                        Var::new_cvar(t, expr.span)
                    }
                    Op2::Rem => {
                        let bit_len = B::Field::MODULUS_BIT_SIZE as usize;
                        let a_bv = term![Op::PfToBv(bit_len); lhs.cvars[0].clone()];
                        let b_bv = term![Op::PfToBv(bit_len); rhs.cvars[0].clone()];
                        let t = term![Op::BvBinOp(BvBinOp::Urem); a_bv.clone(), b_bv.clone()];
                        let t = term![Op::UbvToPf(Box::new(B::Field::to_circ_type())); t];

                        Var::new_var(t, expr.span)
                    }
                    Op2::LShift => {
                        let bit_len = B::Field::MODULUS_BIT_SIZE as usize;
                        let a_bv = term![Op::PfToBv(bit_len); lhs.cvars[0].clone()];
                        let b_bv = term![Op::PfToBv(bit_len); rhs.cvars[0].clone()];
                        // if the shift result is larger than the bit length, it will be truncated:
                        // https://github.com/circify/circ/blob/4aa36e479fe15fb444cc9190e0cb5a1a493ee221/src/ir/term/bv.rs#L96
                        // todo: should we allow field overflow in the hint calculation?
                        let t = term![Op::BvBinOp(BvBinOp::Shl); a_bv, b_bv];
                        // convert back to field
                        let t = term![Op::UbvToPf(Box::new(B::Field::to_circ_type())); t];
                        Var::new_var(t, expr.span)
                    }
                    Op2::LessThan => {
                        let a_int = term![Op::PfToInt; lhs.cvars[0].clone()];
                        let b_int = term![Op::PfToInt; rhs.cvars[0].clone()];
                        let t = term![Op::IntBinPred(IntBinPred::Lt); a_int, b_int];
                        Var::new_var(t, expr.span)
                    }
                    Op2::Pow => {
                        let base_int = term![Op::PfToInt; lhs.cvars[0].clone()];
                        let folded = circ::ir::opt::cfold::fold(&rhs.cvars[0].clone(), &[]);
                        let exp = match &folded.as_value_opt().unwrap() {
                            v => (**v).as_pf().i().to_u32().unwrap(),
                            _ => unreachable!(),
                        };

                        let result = if exp == 0 {
                            let var = Var::new_constant(B::Field::from(1u32), expr.span);
                            term![Op::PfToInt; var.cvars[0].clone()]
                        } else {
                            let mut acc = base_int.clone();
                            for _ in 1..exp {
                                acc = term![Op::IntNaryOp(IntNaryOp::Mul); acc, base_int.clone()];
                            }
                            acc
                        };
                        // convert back to field
                        let converted = term![Op::IntToPf(B::Field::to_circ_type()); result];
                        Var::new_var(converted, expr.span)
                    }
                };
                Ok(Some(VarOrRef::Var(res)))
            }

            ExprKind::Negated(b) => {
                let var = self.compute_expr(fn_env, b)?.unwrap();

                let var = var.value(self, fn_env);

                let t: Term = term![Op::PfUnOp(PfUnOp::Neg); var.cvars[0].clone()];
                let res = Var::new_cvar(t, expr.span);
                Ok(Some(VarOrRef::Var(res)))
            }

            ExprKind::Not(b) => {
                let var = self.compute_expr(fn_env, b)?.unwrap();
                let var = var.value(self, fn_env);

                let t: Term = term![Op::Not; var.cvars[0].clone()];
                let res = Var::new_cvar(t, expr.span);
                Ok(Some(VarOrRef::Var(res)))
            }

            ExprKind::BigUInt(b) => {
                let ff = B::Field::from(b.to_owned());

                let v = Var::new_constant(ff, expr.span);
                Ok(Some(VarOrRef::Var(v)))
            }

            ExprKind::Bool(b) => {
                let v = Var::new_bool(*b, expr.span);
                Ok(Some(VarOrRef::Var(v)))
            }

            ExprKind::StringLiteral(s) => {
                // chars as field elements from asci;;
                let fr: Vec<B::Field> = s.chars().map(|char| B::Field::from(char as u8)).collect();
                let cvars = fr
                    .iter()
                    .map(|&f| leaf_term(Op::new_const(Value::Field(f.to_circ_field()))))
                    .collect();

                Ok(Some(VarOrRef::Var(Var::new(cvars, expr.span))))
            }

            ExprKind::Variable { module, name } => {
                // if it's a type we return nothing
                // (most likely what follows is a static method call)
                if is_type(&name.value) {
                    return Ok(None);
                }

                // search for constants first
                let qualified = FullyQualified::new(module, &name.value);
                let var_info = if let Some(cst_info) = self.const_info(&qualified) {
                    let var = Var::new_constant_typ(cst_info, name.span);
                    VarInfo::new(var, false, Some(cst_info.typ.kind.clone()))
                } else {
                    // if no constant found, look in the function's scope
                    // remember: we can do this because the type checker already checked that we didn't shadow constant vars
                    self.get_local_var(fn_env, &name.value)
                };

                let res = VarOrRef::from_var_info(name.value.clone(), var_info);
                Ok(Some(res))
            }

            ExprKind::ArrayOrTupleAccess { container, idx } => {
                // retrieve var of container
                let var = self
                    .compute_expr(fn_env, container)?
                    .expect("container access on non-container");

                // compute the index
                let idx_var = self
                    .compute_expr(fn_env, idx)?
                    .ok_or_else(|| self.error(ErrorKind::CannotComputeExpression, expr.span))?;
                let idx = idx_var
                    .constant::<B>()
                    .ok_or_else(|| self.error(ErrorKind::ExpectedConstant, expr.span))?;
                let idx: usize = idx.try_into().unwrap();

                // retrieve the type of the elements in the container
                let container_typ = self
                    .expr_type(container)
                    .expect("cannot find type of container");

                // actual starting index for narrowing the var depends on the cotainer
                // for arrays it is just idx * elem_size as all elements are of same size
                // while for tuples we have to sum the sizes of all types up to that index
                let (start, len) = match container_typ {
                    TyKind::Array(ty, array_len) => {
                        if idx >= (*array_len as usize) {
                            return Err(self.error(
                                ErrorKind::ArrayIndexOutOfBounds(idx, *array_len as usize - 1),
                                expr.span,
                            ));
                        }
                        let len = self.size_of(ty);
                        let start = idx * self.size_of(ty);
                        (start, len)
                    }

                    TyKind::Tuple(typs) => {
                        let mut starting_idx = 0;
                        for i in 0..idx {
                            starting_idx += self.size_of(&typs[i]);
                        }
                        (starting_idx, self.size_of(&typs[idx]))
                    }
                    _ => Err(Error::new(
                        "compute-expr",
                        ErrorKind::UnexpectedError("expected container"),
                        expr.span,
                    ))?,
                };

                // compute the real index
                let start = idx * len;

                // out-of-bound checks
                if start >= var.len() || start + len > var.len() {
                    return Err(self.error(
                        ErrorKind::ArrayIndexOutOfBounds(start, var.len()),
                        expr.span,
                    ));
                }

                // index into the var
                let var = var.narrow(start, len);

                //
                Ok(Some(var))
            }

            ExprKind::ArrayDeclaration(items) => {
                let mut cvars = vec![];

                for item in items {
                    let var = self.compute_expr(fn_env, item)?.unwrap();
                    let to_extend = var.value(self, fn_env).cvars.clone();
                    cvars.extend(to_extend);
                }

                let var = VarOrRef::Var(Var::new(cvars, expr.span));

                Ok(Some(var))
            }

            ExprKind::CustomTypeDeclaration { custom: _, fields } => {
                // create the struct by just concatenating all of its cvars
                let mut cvars = vec![];
                for (_field, rhs) in fields {
                    let var = self.compute_expr(fn_env, rhs)?.unwrap();
                    let to_extend = var.value(self, fn_env).cvars.clone();
                    cvars.extend(to_extend);
                }
                let var = VarOrRef::Var(Var::new(cvars, expr.span));

                //
                Ok(Some(var))
            }
            ExprKind::RepeatedArrayInit { item, size } => {
                let size = self
                    .compute_expr(fn_env, size)?
                    .ok_or_else(|| self.error(ErrorKind::CannotComputeExpression, expr.span))?;
                let size = size
                    .constant::<B>()
                    .ok_or_else(|| self.error(ErrorKind::ExpectedConstant, expr.span))?;
                let size: usize = size.try_into().unwrap();

                let mut cvars = vec![];
                for _ in 0..size {
                    let var = self.compute_expr(fn_env, item)?.unwrap();
                    let to_extend = var.value(self, fn_env).cvars.clone();
                    cvars.extend(to_extend);
                }

                let var = VarOrRef::Var(Var::new(cvars, expr.span));
                Ok(Some(var))
            }

            ExprKind::TupleDeclaration(items) => {
                let mut cvars = vec![];

                for item in items {
                    let var = self.compute_expr(fn_env, item)?.unwrap();
                    let to_extend = var.value(self, fn_env).cvars.clone();
                    cvars.extend(to_extend);
                }

                let var = VarOrRef::Var(Var::new(cvars, expr.span));

                Ok(Some(var))
            }
        }
    }

    pub fn expr_type(&self, expr: &Expr) -> Option<&TyKind> {
        self.typed.expr_type(expr)
    }

    pub fn struct_info(&self, qualified: &FullyQualified) -> Option<&StructInfo> {
        self.typed.struct_info(qualified)
    }

    pub fn fn_info(&self, qualified: &FullyQualified) -> Option<&FnInfo<B>> {
        self.typed.fn_info(qualified)
    }

    pub fn const_info(&self, qualified: &FullyQualified) -> Option<&ConstInfo<B::Field>> {
        self.typed.const_info(qualified)
    }

    pub fn size_of(&self, typ: &TyKind) -> usize {
        self.typed.size_of(typ)
    }

    pub fn add_local_var(
        &self,
        fn_env: &mut FnEnv,
        var_name: String,
        var_info: VarInfo,
    ) -> Result<()> {
        // check for consts first
        let qualified = FullyQualified::local(var_name.clone());
        if let Some(_cst_info) = self.typed.const_info(&qualified) {
            Err(Error::new("add-local-var", ErrorKind::UnexpectedError("type checker bug: we already have a constant with the same name (`{var_name}`)!"), Span::default()))?
        }
        fn_env.add_local_var(var_name, var_info);
        Ok(())
    }

    pub fn get_local_var(&self, fn_env: &FnEnv, var_name: &str) -> VarInfo {
        // check for consts first
        let qualified = FullyQualified::local(var_name.to_string());
        if let Some(cst_info) = self.typed.const_info(&qualified) {
            let var = Var::new_constant_typ(cst_info, cst_info.typ.span);
            return VarInfo::new(var, false, Some(TyKind::Field { constant: true }));
        }

        // then check for local variables
        fn_env.get_local_var(var_name)
    }

    pub fn error(&self, kind: ErrorKind, span: Span) -> Error {
        Error::new("ir-generation", kind, span)
    }
}
