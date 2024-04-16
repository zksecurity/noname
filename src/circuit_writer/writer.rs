use std::{
    fmt::{self, Display, Formatter},
    ops::Neg,
};

use ark_ff::{Field, One, Zero};
use kimchi::circuits::polynomials::generic::{GENERIC_COEFFS, GENERIC_REGISTERS};
use kimchi::circuits::wires::Wire;
use num_bigint::BigUint;
use num_traits::Num as _;
use serde::{Deserialize, Serialize};

use crate::{
    backends::Backend,
    circuit_writer::{CircuitWriter, DebugInfo, FnEnv, VarInfo},
    constants::{Span, NUM_REGISTERS},
    constraints::{boolean, field},
    error::{ErrorKind, Result},
    imports::FnKind,
    parser::{
        types::{FunctionDef, Stmt, StmtKind, TyKind},
        Expr, ExprKind, Op2,
    },
    syntax::is_type,
    type_checker::FullyQualified,
    var::{CellVar, ConstOrCell, Value, Var, VarOrRef},
};

//
// Data structures
//

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GateKind {
    Zero,
    DoubleGeneric,
    Poseidon,
}

impl From<GateKind> for kimchi::circuits::gate::GateType {
    fn from(gate_kind: GateKind) -> Self {
        use kimchi::circuits::gate::GateType::*;
        match gate_kind {
            GateKind::Zero => Zero,
            GateKind::DoubleGeneric => Generic,
            GateKind::Poseidon => Poseidon,
        }
    }
}

// TODO: this could also contain the span that defined the gate!
#[derive(Debug, Serialize, Deserialize)]
pub struct Gate<B>
where
    B: Backend,
{
    /// Type of gate
    pub typ: GateKind,

    /// Coefficients
    #[serde(skip)]
    pub coeffs: Vec<B::Field>,
}

impl<B: Backend> Gate<B> {
    pub fn to_kimchi_gate(&self, row: usize) -> kimchi::circuits::gate::CircuitGate<B::Field> {
        kimchi::circuits::gate::CircuitGate {
            typ: self.typ.into(),
            wires: Wire::for_row(row),
            coeffs: self.coeffs.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Cell {
    pub row: usize,
    pub col: usize,
}

impl Display for Cell {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({},{})", self.row, self.col)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Wiring {
    /// Not yet wired (just indicates the position of the cell itself)
    NotWired(AnnotatedCell),
    /// The wiring (associated to different spans)
    Wired(Vec<AnnotatedCell>),
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq)]
pub struct AnnotatedCell {
    pub(crate) cell: Cell,
    pub(crate) debug: DebugInfo,
}

impl PartialEq for AnnotatedCell {
    fn eq(&self, other: &Self) -> bool {
        self.cell == other.cell
    }
}

impl PartialOrd for AnnotatedCell {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.cell.partial_cmp(&other.cell)
    }
}

impl Ord for AnnotatedCell {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cell.cmp(&other.cell)
    }
}

//
// Circuit Writer (also used by witness generation)
//

impl<B: Backend> CircuitWriter<B> {
    /// Returns the compiled gates of the circuit.
    pub fn compiled_gates(&self) -> &[Gate<B>] {
        if !self.finalized {
            unreachable!();
        }
        &self.gates
    }

    fn compile_stmt(
        &mut self,
        fn_env: &mut FnEnv<B::Field>,
        stmt: &Stmt,
    ) -> Result<Option<VarOrRef<B>>> {
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
                // TODO: do we really need to store that in the scope? That's not an actual var in the scope that's an internal var...
                self.add_local_var(fn_env, lhs.value.clone(), var_info);
            }

            StmtKind::ForLoop { var, range, body } => {
                for ii in range.range() {
                    fn_env.nest();

                    let cst_var = Var::new_constant(ii.into(), var.span);
                    let var_info = VarInfo::new(cst_var, false, Some(TyKind::Field));
                    self.add_local_var(fn_env, var.value.clone(), var_info);

                    self.compile_block(fn_env, body)?;

                    fn_env.pop();
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

    /// might return something?
    fn compile_block(
        &mut self,
        fn_env: &mut FnEnv<B::Field>,
        stmts: &[Stmt],
    ) -> Result<Option<Var<B::Field>>> {
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

    fn compile_native_function_call(
        &mut self,
        function: &FunctionDef,
        args: Vec<VarInfo<B::Field>>,
    ) -> Result<Option<Var<B::Field>>> {
        assert!(!function.is_main());

        // create new fn_env
        let fn_env = &mut FnEnv::new();

        // set arguments
        assert_eq!(function.sig.arguments.len(), args.len());

        for (name, var_info) in function.sig.arguments.iter().zip(args) {
            self.add_local_var(fn_env, name.name.value.clone(), var_info);
        }

        // compile it and potentially return a return value
        self.compile_block(fn_env, &function.body)
    }

    pub(crate) fn constrain_inputs_to_main(
        &mut self,
        input: &[ConstOrCell<B::Field>],
        input_typ: &TyKind,
        span: Span,
    ) -> Result<()> {
        match input_typ {
            TyKind::Field => (),
            TyKind::Bool => {
                assert_eq!(input.len(), 1);
                boolean::check(self, &input[0], span);
            }
            TyKind::Array(tykind, _) => {
                let el_size = self.size_of(tykind);
                for el in input.chunks(el_size) {
                    self.constrain_inputs_to_main(el, tykind, span)?;
                }
            }
            TyKind::Custom {
                module,
                name: struct_name,
            } => {
                let qualified = FullyQualified::new(module, &struct_name);
                let struct_info = self
                    .struct_info(&qualified)
                    .ok_or(self.error(ErrorKind::UnexpectedError("struct not found"), span))?
                    .clone();

                let mut offset = 0;
                for (_field_name, field_typ) in &struct_info.fields {
                    let len = self.size_of(field_typ);
                    let range = offset..(offset + len);
                    self.constrain_inputs_to_main(&input[range], field_typ, span)?;
                    offset += len;
                }
            }
            TyKind::BigInt => unreachable!(),
        };
        Ok(())
    }

    /// Compile a function. Used to compile `main()` only for now
    pub(crate) fn compile_main_function(
        &mut self,
        fn_env: &mut FnEnv<B::Field>,
        function: &FunctionDef,
    ) -> Result<()> {
        assert!(function.is_main());

        // compile the block
        let returned = self.compile_block(fn_env, &function.body)?;

        // we're expecting something returned?
        match (function.sig.return_type.as_ref(), returned) {
            (None, None) => Ok(()),
            (Some(expected), None) => Err(self.error(ErrorKind::MissingReturn, expected.span)),
            (None, Some(returned)) => Err(self.error(ErrorKind::UnexpectedReturn, returned.span)),
            (Some(_expected), Some(returned)) => {
                // make sure there are no constants in the returned value
                let mut returned_cells = vec![];
                for r in &returned.cvars {
                    match r {
                        ConstOrCell::Cell(c) => returned_cells.push(c),
                        ConstOrCell::Const(_) => {
                            return Err(self.error(ErrorKind::ConstantInOutput, returned.span))
                        }
                    }
                }

                // store the return value in the public input that was created for that ^
                let public_output = self
                    .public_output
                    .as_ref()
                    .expect("bug in the compiler: missing public output");

                for (pub_var, ret_var) in public_output.cvars.iter().zip(returned_cells) {
                    // replace the computation of the public output vars with the actual variables being returned here
                    let var_idx = pub_var.idx().unwrap();
                    let prev = self
                        .witness_vars
                        .insert(var_idx, Value::PublicOutput(Some(*ret_var)));
                    assert!(prev.is_some());
                }

                Ok(())
            }
        }
    }

    fn compute_expr(
        &mut self,
        fn_env: &mut FnEnv<B::Field>,
        expr: &Expr,
    ) -> Result<Option<VarOrRef<B>>> {
        match &expr.kind {
            // `module::fn_name(args)`
            ExprKind::FnCall {
                module,
                fn_name,
                args,
            } => {
                // sanity check
                if fn_name.value == "main" {
                    return Err(self.error(ErrorKind::RecursiveMain, expr.span));
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
                    let mutable = false; // TODO: mut keyword in arguments?
                    let var_info = VarInfo::new(var, mutable, typ);

                    vars.push(var_info);
                }

                let res = match &fn_info.kind {
                    // assert() <-- for example
                    FnKind::BuiltIn(_sig, handle) => {
                        let res = handle(self, &vars, expr.span);
                        res.map(|r| r.map(VarOrRef::Var))
                    }

                    // fn_name(args)
                    // ^^^^^^^
                    FnKind::Native(func) => {
                        // module::fn_name(args)
                        // ^^^^^^
                        self.compile_native_function_call(&func, vars)
                            .map(|r| r.map(VarOrRef::Var))
                    }
                };

                //
                res
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
                    _ => {
                        panic!("could not figure out struct implementing that method call")
                    }
                };

                let qualified = FullyQualified::new(module, self_struct);
                let struct_info = self
                    .struct_info(&qualified)
                    .expect("struct info not found for custom struct");

                // find range of field
                let mut start = 0;
                let mut len = 0;
                for (field, field_typ) in &struct_info.fields {
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

                        // TODO: for now we pass `self` by value as well
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

                    // TODO: for now we pass `self` by value as well
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
                    .value(self, fn_env);
                let then_ = self
                    .compute_expr(fn_env, then_)?
                    .unwrap()
                    .value(self, fn_env);
                let else_ = self
                    .compute_expr(fn_env, else_)?
                    .unwrap()
                    .value(self, fn_env);

                let res = field::if_else(self, &cond, &then_, &else_, expr.span);

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
                    VarOrRef::Var(_) => panic!("can't reassign this non-mutable variable"),
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
                    Op2::Addition => field::add(self, &lhs[0], &rhs[0], expr.span),
                    Op2::Subtraction => field::sub(self, &lhs[0], &rhs[0], expr.span),
                    Op2::Multiplication => field::mul(self, &lhs[0], &rhs[0], expr.span),
                    Op2::Equality => field::equal(self, &lhs, &rhs, expr.span),
                    Op2::BoolAnd => boolean::and(self, &lhs[0], &rhs[0], expr.span),
                    Op2::BoolOr => boolean::or(self, &lhs[0], &rhs[0], expr.span),
                    Op2::Division => todo!(),
                };

                Ok(Some(VarOrRef::Var(res)))
            }

            ExprKind::Negated(b) => {
                let var = self.compute_expr(fn_env, b)?.unwrap();

                let var = var.value(self, fn_env);

                todo!()
            }

            ExprKind::Not(b) => {
                let var = self.compute_expr(fn_env, b)?.unwrap();

                let var = var.value(self, fn_env);

                let res = boolean::not(self, &var[0], expr.span.merge_with(b.span));
                Ok(Some(VarOrRef::Var(res)))
            }

            ExprKind::BigInt(b) => {
                let biguint = BigUint::from_str_radix(b, 10).expect("failed to parse number.");
                let ff = B::Field::try_from(biguint).map_err(|_| {
                    self.error(ErrorKind::CannotConvertToField(b.to_string()), expr.span)
                })?;

                let res = VarOrRef::Var(Var::new_constant(ff, expr.span));
                Ok(Some(res))
            }

            ExprKind::Bool(b) => {
                let value = if *b {
                    B::Field::one()
                } else {
                    B::Field::zero()
                };
                let res = VarOrRef::Var(Var::new_constant(value, expr.span));
                Ok(Some(res))
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

            ExprKind::ArrayAccess { array, idx } => {
                // retrieve var of array
                let var = self
                    .compute_expr(fn_env, array)?
                    .expect("array access on non-array");

                // compute the index
                let idx_var = self
                    .compute_expr(fn_env, idx)?
                    .ok_or_else(|| self.error(ErrorKind::CannotComputeExpression, expr.span))?;
                let idx = idx_var
                    .constant()
                    .ok_or_else(|| self.error(ErrorKind::ExpectedConstant, expr.span))?;
                let idx: BigUint = idx.into();
                let idx: usize = idx.try_into().unwrap();

                // retrieve the type of the elements in the array
                let array_typ = self.expr_type(array).expect("cannot find type of array");

                let elem_type = match array_typ {
                    TyKind::Array(ty, array_len) => {
                        if idx >= (*array_len as usize) {
                            return Err(self.error(
                                ErrorKind::ArrayIndexOutOfBounds(idx, *array_len as usize - 1),
                                expr.span,
                            ));
                        }
                        ty
                    }
                    _ => panic!("expected array"),
                };

                // compute the size of each element in the array
                let len = self.size_of(elem_type);

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
        }
    }

    // TODO: dead code?
    pub fn compute_constant(&self, var: CellVar, span: Span) -> Result<B::Field> {
        match &self.witness_vars.get(&var.index) {
            Some(Value::Constant(c)) => Ok(*c),
            Some(Value::LinearCombination(lc, cst)) => {
                let mut res = *cst;
                for (coeff, var) in lc {
                    res += self.compute_constant(*var, span)? * *coeff;
                }
                Ok(res)
            }
            Some(Value::Mul(lhs, rhs)) => {
                let lhs = self.compute_constant(*lhs, span)?;
                let rhs = self.compute_constant(*rhs, span)?;
                Ok(lhs * rhs)
            }
            _ => Err(self.error(ErrorKind::ExpectedConstant, span)),
        }
    }

    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }

    // TODO: we should cache constants to avoid creating a new variable for each constant
    /// This should be called only when you want to constrain a constant for real.
    /// Gates that handle constants should always make sure to call this function when they want them constrained.
    pub fn add_constant(
        &mut self,
        label: Option<&'static str>,
        value: B::Field,
        span: Span,
    ) -> CellVar {
        if let Some(cvar) = self.cached_constants.get(&value) {
            return *cvar;
        }

        let var = self.backend.new_internal_var(Value::Constant(value), span);
        self.cached_constants.insert(value, var);

        let zero = B::Field::zero();
        self.add_generic_gate(
            label.unwrap_or("hardcode a constant"),
            vec![Some(var)],
            vec![B::Field::one(), zero, zero, zero, value.neg()],
            span,
        );

        var
    }

    /// creates a new gate, and the associated row in the witness/execution trace.
    // TODO: add_gate instead of gates?
    pub fn add_gate(
        &mut self,
        note: &'static str,
        typ: GateKind,
        vars: Vec<Option<CellVar>>,
        coeffs: Vec<B::Field>,
        span: Span,
    ) {
        // sanitize
        assert!(coeffs.len() <= NUM_REGISTERS);
        assert!(vars.len() <= NUM_REGISTERS);

        // construct the execution trace with vars, for the witness generation
        self.rows_of_vars.push(vars.clone());

        // get current row
        // important: do that before adding the gate below
        let row = self.gates.len();

        // add gate
        self.gates.push(Gate { typ, coeffs });

        // add debug info related to that gate
        let debug_info = DebugInfo {
            span,
            note: note.to_string(),
        };
        self.debug_info.push(debug_info.clone());

        // wiring (based on vars)
        for (col, var) in vars.iter().enumerate() {
            if let Some(var) = var {
                let curr_cell = Cell { row, col };
                let annotated_cell = AnnotatedCell {
                    cell: curr_cell,
                    debug: debug_info.clone(),
                };

                self.wiring
                    .entry(var.index)
                    .and_modify(|w| match w {
                        Wiring::NotWired(old_cell) => {
                            *w = Wiring::Wired(vec![old_cell.clone(), annotated_cell.clone()])
                        }
                        Wiring::Wired(ref mut cells) => {
                            cells.push(annotated_cell.clone());
                        }
                    })
                    .or_insert(Wiring::NotWired(annotated_cell));
            }
        }
    }

    pub fn add_public_inputs(&mut self, name: String, num: usize, span: Span) -> Var<B::Field> {
        let mut cvars = Vec::with_capacity(num);

        for idx in 0..num {
            // create the var
            let cvar = self.backend.new_internal_var(Value::External(name.clone(), idx), span);
            cvars.push(ConstOrCell::Cell(cvar));

            // create the associated generic gate
            self.add_gate(
                "add public input",
                GateKind::DoubleGeneric,
                vec![Some(cvar)],
                vec![B::Field::one()],
                span,
            );
        }

        self.public_input_size += num;

        Var::new(cvars, span)
    }

    pub fn add_public_outputs(&mut self, num: usize, span: Span) {
        assert!(self.public_output.is_none());

        let mut cvars = Vec::with_capacity(num);
        for _ in 0..num {
            // create the var
            let cvar = self.backend.new_internal_var(Value::PublicOutput(None), span);
            cvars.push(ConstOrCell::Cell(cvar));

            // create the associated generic gate
            self.add_generic_gate(
                "add public output",
                vec![Some(cvar)],
                vec![B::Field::one()],
                span,
            );
        }
        self.public_input_size += num;

        // store it
        let res = Var::new(cvars, span);
        self.public_output = Some(res);
    }

    pub fn add_private_inputs(&mut self, name: String, num: usize, span: Span) -> Var<B::Field> {
        let mut cvars = Vec::with_capacity(num);

        for idx in 0..num {
            // create the var
            let cvar = self.backend.new_internal_var(Value::External(name.clone(), idx), span);
            cvars.push(ConstOrCell::Cell(cvar));
            self.private_input_indices.push(cvar.index);
        }

        Var::new(cvars, span)
    }

    pub(crate) fn add_generic_gate(
        &mut self,
        label: &'static str,
        mut vars: Vec<Option<CellVar>>,
        mut coeffs: Vec<B::Field>,
        span: Span,
    ) {
        // padding
        let coeffs_padding = GENERIC_COEFFS.checked_sub(coeffs.len()).unwrap();
        coeffs.extend(std::iter::repeat(B::Field::zero()).take(coeffs_padding));

        let vars_padding = GENERIC_REGISTERS.checked_sub(vars.len()).unwrap();
        vars.extend(std::iter::repeat(None).take(vars_padding));

        // if the double gate optimization is not set, just add the gate
        if !self.double_generic_gate_optimization {
            self.add_gate(label, GateKind::DoubleGeneric, vars, coeffs, span);
            return;
        }

        // only add a double generic gate if we have two of them
        if let Some(generic_gate) = self.pending_generic_gate.take() {
            coeffs.extend(generic_gate.coeffs);
            vars.extend(generic_gate.vars);

            // TODO: what to do with the label and span?

            self.add_gate(label, GateKind::DoubleGeneric, vars, coeffs, span);
        } else {
            // otherwise queue it
            self.pending_generic_gate = Some(PendingGate {
                label,
                coeffs,
                vars,
                span,
            });
        }
    }
}

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub(crate) struct PendingGate<F>
where
    F: Field,
{
    pub label: &'static str,
    #[serde(skip)]
    pub coeffs: Vec<F>,
    pub vars: Vec<Option<CellVar>>,
    pub span: Span,
}
