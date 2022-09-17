use std::{
    collections::{HashMap, HashSet},
    fmt::{self, Display, Formatter},
    ops::Neg,
    vec,
};

use ark_ff::{One, Zero};
use num_bigint::BigUint;
use num_traits::Num as _;

use crate::{
    asm, boolean,
    constants::{Field, Span, NUM_REGISTERS},
    error::{Error, ErrorKind, Result},
    field,
    imports::FnKind,
    parser::{
        AttributeKind, Expr, ExprKind, FnArg, FnSig, Function, Op2, RootKind, Stmt, StmtKind,
        TyKind,
    },
    syntax::is_const,
    type_checker::{TypedGlobalEnv, TAST},
    var::{CellVar, ConstOrCell, Value, Var, VarKind},
    witness::CompiledCircuit,
};

//
// Data structures
//

/// The environment of the module/program.
#[derive(Debug)]
pub struct GlobalEnv {
    typed: TypedGlobalEnv,

    /// Constants defined in the module/program.
    constants: HashMap<String, Var>,
}

impl GlobalEnv {
    /// Creates a global environment from the one created by the type checker.
    pub fn new(typed_global_env: TypedGlobalEnv) -> Self {
        Self {
            typed: typed_global_env,
            constants: HashMap::new(),
        }
    }

    /// Stores type information about a local variable.
    /// Note that we forbid shadowing at all scopes.
    pub fn add_constant(&mut self, name: String, constant: Field, span: Span) {
        let var = Var::new_constant(constant, span);
        if self.constants.insert(name.clone(), var).is_some() {
            panic!("constant `{name}` already exists");
        }
    }

    /// Retrieves type information on a constantiable, given a name.
    /// If the constantiable is not in scope, return false.
    // TODO: return an error no?
    pub fn get_constant(&self, ident: &str) -> &Var {
        let constant = self
            .constants
            .get(ident)
            .unwrap_or_else(|| panic!("constant `{ident}` not found"));

        constant
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GateKind {
    DoubleGeneric,
    Poseidon,
}

impl From<GateKind> for kimchi::circuits::gate::GateType {
    fn from(gate_kind: GateKind) -> Self {
        use kimchi::circuits::gate::GateType::*;
        match gate_kind {
            GateKind::DoubleGeneric => Generic,
            GateKind::Poseidon => Poseidon,
        }
    }
}

// TODO: this could also contain the span that defined the gate!
#[derive(Debug)]
pub struct Gate {
    /// Type of gate
    pub typ: GateKind,

    /// col -> (row, col)
    // TODO: do we want to do an external wiring instead?
    //    wiring: HashMap<u8, (u64, u8)>,

    /// Coefficients
    pub coeffs: Vec<Field>,

    /// The place in the original source code that created that gate.
    pub span: Span,

    /// A note on why this was added
    pub note: &'static str,
}

impl Gate {
    pub fn to_kimchi_gate(&self, row: usize) -> kimchi::circuits::gate::CircuitGate<Field> {
        kimchi::circuits::gate::CircuitGate {
            typ: self.typ.into(),
            wires: kimchi::circuits::wires::Wire::new(row),
            coeffs: self.coeffs.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Cell {
    pub row: usize,
    pub col: usize,
}

impl Display for Cell {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({},{})", self.row, self.col)
    }
}

#[derive(Debug, Clone)]
pub enum Wiring {
    /// Not yet wired (just indicates the position of the cell itself)
    NotWired(Cell),
    /// The wiring (associated to different spans)
    Wired(Vec<(Cell, Span)>),
}

//
// Circuit Writer (also used by witness generation)
//

#[derive(Default, Debug)]
pub struct CircuitWriter {
    /// The source code that created this circuit.
    /// Useful for debugging and displaying user errors.
    pub source: String,

    /// Once this is set, you can generate a witness (and can't modify the circuit?)
    pub finalized: bool,

    /// This is used to give a distinct number to each variable during circuit generation.
    pub next_variable: usize,

    /// This is how you compute the value of each variable during witness generation.
    /// It is created during circuit generation.
    pub witness_vars: HashMap<usize, Value>,

    /// The execution trace table with vars as placeholders.
    /// It is created during circuit generation,
    /// and used by the witness generator.
    pub rows_of_vars: Vec<Vec<Option<CellVar>>>,

    /// The gates created by the circuit generation.
    gates: Vec<Gate>,

    /// The wiring of the circuit.
    /// It is created during circuit generation.
    pub wiring: HashMap<usize, Wiring>,

    /// Size of the public input.
    pub public_input_size: usize,

    /// If a public output is set, this will be used to store its [Var].
    /// The public output generation works as follows:
    /// 1. This cvar is created and inserted in the circuit (gates) during compilation of the public input
    ///    (as the public output is the end of the public input)
    /// 2. When the `return` statement of the circuit is parsed,
    ///    it will set this `public_output` variable again to the correct vars.
    /// 3. During witness generation, the public output computation
    ///    is delayed until the very end.
    pub public_output: Option<Var>,

    /// Indexes used by the private inputs
    /// (this is useful to check that they appear in the circuit)
    pub private_input_indices: Vec<usize>,

    /// Used during the witness generation to check
    /// public and private inputs given by the prover.
    pub main: (FnSig, Span),
}

impl CircuitWriter {
    pub fn generate_circuit(tast: TAST, code: &str) -> Result<CompiledCircuit> {
        // if there's no main function, then return an error
        let TAST {
            ast,
            typed_global_env,
        } = tast;

        if !typed_global_env.has_main {
            return Err(Error {
                kind: ErrorKind::NoMainFunction,
                span: Span::default(),
            });
        }

        let (main_sig, main_span) = {
            let fn_info = typed_global_env.functions.get("main").cloned().unwrap();

            (fn_info.sig().clone(), fn_info.span)
        };

        let mut circuit_writer = CircuitWriter {
            source: code.to_string(),
            main: (main_sig, main_span),
            ..CircuitWriter::default()
        };

        // make sure we can't call that several times
        if circuit_writer.finalized {
            panic!("circuit already finalized (TODO: return a proper error");
        }

        let global_env = &mut GlobalEnv::new(typed_global_env);

        for root in &ast.0 {
            match &root.kind {
                // imports (already dealt with in type checker)
                RootKind::Use(_path) => (),

                // `const THING = 42;`
                RootKind::Const(cst) => {
                    global_env.add_constant(cst.name.value.clone(), cst.value, cst.span);
                }

                // `fn main() { ... }`
                RootKind::Function(function) => {
                    // create the env
                    let mut fn_env = FnEnv::default();

                    // we only compile the main function
                    if !function.is_main() {
                        continue;
                    }

                    // if there are no arguments, return an error
                    // TODO: should we check this in the type checker?
                    if function.sig.arguments.is_empty() {
                        return Err(Error {
                            kind: ErrorKind::NoArgsInMain,
                            span: function.span,
                        });
                    }

                    // create public and private inputs
                    for FnArg {
                        attribute,
                        name,
                        typ,
                        ..
                    } in &function.sig.arguments
                    {
                        let len = match &typ.kind {
                            TyKind::Field => 1,
                            TyKind::Array(typ, len) => {
                                if !matches!(**typ, TyKind::Field) {
                                    unimplemented!();
                                }
                                *len as usize
                            }
                            TyKind::Bool => 1,
                            _ => unimplemented!(),
                        };

                        let var = if let Some(attr) = attribute {
                            if !matches!(attr.kind, AttributeKind::Pub) {
                                return Err(Error {
                                    kind: ErrorKind::InvalidAttribute(attr.kind),
                                    span: attr.span,
                                });
                            }
                            circuit_writer.add_public_inputs(name.value.clone(), len, name.span)
                        } else {
                            circuit_writer.add_private_inputs(name.value.clone(), len, name.span)
                        };

                        // add argument variable to the ast env
                        fn_env.add_var(name.value.clone(), var);
                    }

                    // create public output
                    if let Some(typ) = &function.sig.return_type {
                        if typ.kind != TyKind::Field {
                            unimplemented!();
                        }

                        // create it
                        circuit_writer.add_public_outputs(1, typ.span);
                    }

                    // compile function
                    circuit_writer.compile_main_function(&global_env, &mut fn_env, function)?;
                }

                // struct definition (already dealt with in type checker)
                RootKind::Struct(_struct) => (),

                // ignore comments
                // TODO: we could actually preserve the comment in the ASM!
                RootKind::Comment(_comment) => (),
            }
        }

        // for sanity check, we make sure that every cellvar created has ended up in a gate
        let mut written_vars = HashSet::new();
        for row in &circuit_writer.rows_of_vars {
            row.iter().flatten().for_each(|cvar| {
                written_vars.insert(cvar.index);
            });
        }

        for var in 0..circuit_writer.next_variable {
            if !written_vars.contains(&var) {
                if circuit_writer.private_input_indices.contains(&var) {
                    return Err(Error {
                        kind: ErrorKind::PrivateInputNotUsed,
                        span: circuit_writer.main.1,
                    });
                } else {
                    panic!("there's a bug in the circuit_writer, some cellvar does not end up being a cellvar in the circuit!");
                }
            }
        }

        // we finalized!
        circuit_writer.finalized = true;

        Ok(CompiledCircuit::new(circuit_writer))
    }

    /// Returns the compiled gates of the circuit.
    pub fn compiled_gates(&self) -> &[Gate] {
        if !self.finalized {
            panic!("Circuit not finalized yet!");
        }
        &self.gates
    }

    fn compile_stmt(
        &mut self,
        global_env: &GlobalEnv,
        fn_env: &mut FnEnv,
        stmt: &Stmt,
    ) -> Result<Option<Var>> {
        match &stmt.kind {
            StmtKind::Assign { lhs, rhs, .. } => {
                // compute the rhs
                let var = self.compute_expr(global_env, fn_env, rhs)?.ok_or(Error {
                    kind: ErrorKind::CannotComputeExpression,
                    span: stmt.span,
                })?;

                // store the new variable
                // TODO: do we really need to store that in the scope? That's not an actual var in the scope that's an internal var...
                fn_env.add_var(lhs.value.clone(), var);
            }
            StmtKind::For { var, range, body } => {
                for ii in range.range() {
                    fn_env.nest();

                    let cst_var = Var::new_constant(ii.into(), var.span);
                    fn_env.add_var(var.value.clone(), cst_var);

                    self.compile_block(global_env, fn_env, body)?;

                    fn_env.pop();
                }
            }
            StmtKind::Expr(expr) => {
                // compute the expression
                let var = self.compute_expr(global_env, fn_env, expr)?;

                // make sure it does not return any value.
                assert!(var.is_none());
            }
            StmtKind::Return(expr) => {
                let var = self.compute_expr(global_env, fn_env, expr)?.ok_or(Error {
                    kind: ErrorKind::CannotComputeExpression,
                    span: stmt.span,
                })?;

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
        global_env: &GlobalEnv,
        fn_env: &mut FnEnv,
        stmts: &[Stmt],
    ) -> Result<Option<Var>> {
        fn_env.nest();
        for stmt in stmts {
            let res = self.compile_stmt(global_env, fn_env, stmt)?;
            if res.is_some() {
                // we already checked for early returns in type checking
                return Ok(res);
            }
        }
        fn_env.pop();
        Ok(None)
    }

    fn compile_native_function_call(
        &mut self,
        global_env: &GlobalEnv,
        function: &Function,
        args: Vec<Var>,
        span: Span,
    ) -> Result<Option<Var>> {
        assert!(!function.is_main());

        // create new fn_env
        let fn_env = &mut FnEnv::new();

        // set arguments
        assert_eq!(function.sig.arguments.len(), args.len());

        for (name, var) in function.sig.arguments.iter().zip(args) {
            fn_env.add_var(name.name.value.clone(), var);
        }

        // compile it and potentially return a return value
        self.compile_block(global_env, fn_env, &function.body)
    }

    /// Compile a function. Used to compile `main()` only for now
    fn compile_main_function(
        &mut self,
        global_env: &GlobalEnv,
        fn_env: &mut FnEnv,
        function: &Function,
    ) -> Result<()> {
        assert!(function.is_main());

        // compile the block
        let returned = self.compile_block(global_env, fn_env, &function.body)?;

        // we're expecting something returned?
        match (function.sig.return_type.as_ref(), returned) {
            (None, None) => Ok(()),
            (Some(expected), None) => Err(Error {
                kind: ErrorKind::MissingReturn,
                span: expected.span,
            }),
            (None, Some(returned)) => Err(Error {
                kind: ErrorKind::UnexpectedReturn,
                span: returned.span,
            }),
            (Some(_expected), Some(returned)) => {
                // make sure there are no constants in the returned value
                let mut returned_cells = vec![];
                for r in returned.into_const_or_cells() {
                    match r {
                        ConstOrCell::Cell(c) => returned_cells.push(c),
                        ConstOrCell::Const(_) => {
                            return Err(Error {
                                kind: ErrorKind::ConstantInOutput,
                                span: returned.span,
                            })
                        }
                    }
                }

                // store the return value in the public input that was created for that ^
                let public_output = self
                    .public_output
                    .as_ref()
                    .expect("bug in the compiler: missing public output");

                for (pub_var, ret_var) in public_output
                    .into_const_or_cells()
                    .into_iter()
                    .zip(returned_cells)
                {
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

    pub fn asm(&self, debug: bool) -> String {
        asm::generate_asm(&self.source, &self.gates, &self.wiring, debug)
    }

    pub fn new_internal_var(&mut self, val: Value, span: Span) -> CellVar {
        // create new var
        let var = CellVar::new(self.next_variable, span);
        self.next_variable += 1;

        // store it in the circuit_writer
        self.witness_vars.insert(var.index, val);

        var
    }

    fn compute_expr(
        &mut self,
        global_env: &GlobalEnv,
        fn_env: &mut FnEnv,
        expr: &Expr,
    ) -> Result<Option<Var>> {
        match &expr.kind {
            ExprKind::FnCall { name, args } => {
                // compute the arguments
                let mut vars = Vec::with_capacity(args.len());
                for arg in args {
                    let var = self.compute_expr(global_env, fn_env, arg)?.ok_or(Error {
                        kind: ErrorKind::CannotComputeExpression,
                        span: arg.span,
                    })?;
                    vars.push(var);
                }

                // retrieve the function in the env
                if name.len() == 1 {
                    // functions present in the scope
                    let fn_name = &name.path[0].value;
                    let fn_info = global_env.typed.functions.get(fn_name).ok_or(Error {
                        kind: ErrorKind::UndefinedFunction(fn_name.clone()),
                        span: name.span,
                    })?;

                    match &fn_info.kind {
                        FnKind::BuiltIn(_, handle) => handle(self, &vars, expr.span),
                        FnKind::Native(func) => {
                            self.compile_native_function_call(global_env, func, vars, expr.span)
                        }
                        FnKind::Main(_) => {
                            return Err(Error {
                                kind: ErrorKind::RecursiveMain,
                                span: expr.span,
                            });
                        }
                    }
                } else if name.len() == 2 {
                    // check module present in the scope
                    let module = &name.path[0];
                    let fn_name = &name.path[1];
                    let module = global_env.typed.modules.get(&module.value).ok_or(Error {
                        kind: ErrorKind::UndefinedModule(module.value.clone()),
                        span: module.span,
                    })?;
                    let fn_info = module.functions.get(&fn_name.value).ok_or(Error {
                        kind: ErrorKind::UndefinedFunction(fn_name.value.clone()),
                        span: fn_name.span,
                    })?;

                    match &fn_info.kind {
                        FnKind::BuiltIn(_, handle) => handle(self, &vars, expr.span),
                        FnKind::Native(_) => todo!(),
                        FnKind::Main(_) => {
                            return Err(Error {
                                kind: ErrorKind::RecursiveMain,
                                span: expr.span,
                            });
                        }
                    }
                } else {
                    Err(Error {
                        kind: ErrorKind::InvalidFnCall("sub-sub modules unsupported"),
                        span: name.span,
                    })
                }
            }
            ExprKind::Assignment { lhs, rhs } => {
                // figure out the local var name of lhs
                let lhs_name = match &lhs.kind {
                    ExprKind::Identifier(n) => n,
                    ExprKind::ArrayAccess(_, _) => todo!(),
                    _ => panic!("type checker error"),
                };

                // figure out the var of what's on the right
                let rhs = self.compute_expr(global_env, fn_env, rhs)?.unwrap();

                // replace the left with the right
                fn_env.reassign_var(lhs_name, rhs);

                Ok(None)
            }
            ExprKind::Op(op, lhs, rhs) => match op {
                Op2::Addition => {
                    let lhs = self.compute_expr(global_env, fn_env, lhs)?.unwrap();
                    let rhs = self.compute_expr(global_env, fn_env, rhs)?.unwrap();

                    Ok(Some(field::add(self, lhs, rhs, expr.span)))
                }
                Op2::Subtraction => todo!(),
                Op2::Multiplication => todo!(),
                Op2::Division => todo!(),
                Op2::Equality => {
                    let lhs = self.compute_expr(global_env, fn_env, lhs)?.unwrap();
                    let rhs = self.compute_expr(global_env, fn_env, rhs)?.unwrap();

                    Ok(Some(field::equal(self, &lhs.kind, &rhs.kind, expr.span)))
                }
                Op2::BoolAnd => {
                    let lhs = self.compute_expr(global_env, fn_env, lhs)?.unwrap();
                    let rhs = self.compute_expr(global_env, fn_env, rhs)?.unwrap();

                    Ok(Some(boolean::and(self, lhs, rhs, expr.span)))
                }
                Op2::BoolOr => todo!(),
                Op2::BoolNot => todo!(),
            },
            ExprKind::Negated(b) => {
                let var = self.compute_expr(global_env, fn_env, b)?.unwrap();
                Ok(Some(boolean::neg(self, var, expr.span)))
            }
            ExprKind::BigInt(b) => {
                let biguint = BigUint::from_str_radix(b, 10).expect("failed to parse number.");
                let ff = Field::try_from(biguint).map_err(|_| Error {
                    kind: ErrorKind::CannotConvertToField(b.to_string()),
                    span: expr.span,
                })?;

                Ok(Some(Var::new_constant(ff, expr.span)))
            }
            ExprKind::Bool(b) => {
                let value = if *b { Field::one() } else { Field::zero() };
                Ok(Some(Var::new_constant(value, expr.span)))
            }
            ExprKind::Identifier(name) => {
                let var = if is_const(name) {
                    global_env.get_constant(name).clone()
                } else {
                    fn_env.get_var(name).clone()
                };
                Ok(Some(var))
            }
            ExprKind::ArrayAccess(path, expr) => {
                // retrieve the CircuitVar at the path
                let array: Var = if path.len() == 1 {
                    // var present in the scope
                    let name = &path.path[0].value;
                    fn_env.get_var(name).clone()
                } else if path.len() == 2 {
                    // check module present in the scope
                    let module = &path.path[0];
                    let _name = &path.path[1];
                    let _module = global_env.typed.modules.get(&module.value).ok_or(Error {
                        kind: ErrorKind::UndefinedModule(module.value.clone()),
                        span: module.span,
                    })?;
                    unimplemented!()
                } else {
                    return Err(Error {
                        kind: ErrorKind::InvalidPath,
                        span: path.span,
                    });
                };

                // compute the index
                let idx_var = self.compute_expr(global_env, fn_env, expr)?.ok_or(Error {
                    kind: ErrorKind::CannotComputeExpression,
                    span: expr.span,
                })?;

                // the index must be a constant!!
                let idx = idx_var.constant().ok_or(Error {
                    kind: ErrorKind::ExpectedConstant,
                    span: expr.span,
                })?;

                let idx: BigUint = idx.into();
                let idx: usize = idx.try_into().unwrap();

                // index into the CircuitVar (and prevent out of bounds)
                let res = array.get(idx).cloned().ok_or(Error {
                    kind: ErrorKind::ArrayIndexOutOfBounds(idx, array.len()),
                    span: expr.span,
                })?;

                //
                Ok(Some(Var::new(res, expr.span)))
            }
            ExprKind::ArrayDeclaration(items) => {
                // we only support arrays of Field elements at the moment.
                // not sure yet how we can support any other type of arrays...
                // perhaps we could abstract an array as an list of cellvars (each item contains one cellvars)
                let mut vars = Vec::with_capacity(items.len());

                for item in items {
                    let var = self.compute_expr(global_env, fn_env, item)?.unwrap();
                    vars.push(var.kind.clone());
                    /*
                    to support other types:
                     in the loop: vars.push(var);
                     after: vars.flatten() (or smthg like that)
                     and save the information in
                     */
                }

                let var = Var::new_array(vars, expr.span);

                Ok(Some(var))
            }
            ExprKind::CustomTypeDeclaration(name, fields) => {
                let mut vars = HashMap::new();
                for (name, rhs) in fields {
                    let var = self.compute_expr(global_env, fn_env, rhs)?.unwrap();
                    vars.insert(name.value.clone(), var.kind.clone());
                }
                let var = Var::new_struct(vars, expr.span);

                fn_env.add_var(name.clone(), var.clone());

                Ok(Some(var))
            }
            ExprKind::StructAccess(name, field) => {
                let stru = fn_env.get_var(&name.value);

                let fields = stru.fields().expect("type-checker bug: expected struct");

                let var = fields.get(&field.value).cloned().ok_or(Error {
                    kind: ErrorKind::UndefinedField(name.value.clone(), field.value.clone()),
                    span: field.span,
                })?;

                Ok(Some(Var::new(var, expr.span)))
            }
        }
    }

    // TODO: dead code?
    pub fn compute_constant(&self, var: CellVar, span: Span) -> Result<Field> {
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
            _ => Err(Error {
                kind: ErrorKind::ExpectedConstant,
                span,
            }),
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
        value: Field,
        span: Span,
    ) -> CellVar {
        let var = self.new_internal_var(Value::Constant(value), span);

        let zero = Field::zero();
        self.add_gate(
            label.unwrap_or("hardcode a constant"),
            GateKind::DoubleGeneric,
            vec![Some(var)],
            vec![Field::one(), zero, zero, zero, value.neg()],
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
        coeffs: Vec<Field>,
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
        self.gates.push(Gate {
            typ,
            coeffs,
            span,
            note,
        });

        // wiring (based on vars)
        for (col, var) in vars.iter().enumerate() {
            if let Some(var) = var {
                let curr_cell = Cell { row, col };
                self.wiring
                    .entry(var.index)
                    .and_modify(|w| match w {
                        Wiring::NotWired(cell) => {
                            *w = Wiring::Wired(vec![(*cell, var.span), (curr_cell, span)])
                        }
                        Wiring::Wired(ref mut cells) => {
                            cells.push((curr_cell, span));
                        }
                    })
                    .or_insert(Wiring::NotWired(curr_cell));
            }
        }
    }

    pub fn add_public_inputs(&mut self, name: String, num: usize, span: Span) -> Var {
        let mut vars = Vec::with_capacity(num);

        for idx in 0..num {
            // create the var
            let cvar = self.new_internal_var(Value::External(name.clone(), idx), span);
            vars.push(VarKind::new_cell(cvar));

            // create the associated generic gate
            self.add_gate(
                "add public input",
                GateKind::DoubleGeneric,
                vec![Some(cvar)],
                vec![Field::one()],
                span,
            );
        }

        self.public_input_size += num;

        // TODO: support more than arrays and fields
        if num == 1 {
            Var::new(vars[0].clone(), span)
        } else {
            Var::new_array(vars, span)
        }
    }

    pub fn add_public_outputs(&mut self, num: usize, span: Span) {
        assert!(self.public_output.is_none());

        let mut vars = Vec::with_capacity(num);
        for _ in 0..num {
            // create the var
            let cvar = self.new_internal_var(Value::PublicOutput(None), span);
            vars.push(VarKind::new_cell(cvar));

            // create the associated generic gate
            self.add_gate(
                "add public output",
                GateKind::DoubleGeneric,
                vec![Some(cvar)],
                vec![Field::one()],
                span,
            );
        }
        self.public_input_size += num;

        // store it
        // TODO: support more than arrays and fields
        let res = if num == 1 {
            Var::new(vars[0].clone(), span)
        } else {
            Var::new_array(vars, span)
        };
        self.public_output = Some(res);
    }

    pub fn add_private_inputs(&mut self, name: String, num: usize, span: Span) -> Var {
        let mut vars = Vec::with_capacity(num);

        for idx in 0..num {
            // create the var
            let cvar = self.new_internal_var(Value::External(name.clone(), idx), span);
            vars.push(VarKind::new_cell(cvar));
            self.private_input_indices.push(cvar.index);
        }

        // TODO: support more than arrays and fields
        if num == 1 {
            Var::new(vars[0].clone(), span)
        } else {
            Var::new_array(vars, span)
        }
    }
}

//
// Local Environment
//

/// Is used to store functions' scoped variables.
/// This include inputs and output of the function,  as well as local variables.
/// You can think of it as a function call stack.
#[derive(Default, Debug, Clone)]
pub struct FnEnv {
    /// The current nesting level.
    /// Starting at 0 (top level), and increasing as we go into a block.
    current_scope: usize,

    /// Used by the private and public inputs,
    /// and any other external variables created in the circuit
    /// This needs to be garbage collected when we exit a scope.
    /// Note that usize here represents the scope the variable was created in.
    vars: HashMap<String, (usize, Var)>,
}

impl FnEnv {
    /// Creates a new FnEnv
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
    pub fn add_var(&mut self, name: String, var: Var) {
        if self
            .vars
            .insert(name.clone(), (self.current_scope, var))
            .is_some()
        {
            panic!("type checker error: var {name} already exists");
        }
    }

    /// Retrieves type information on a variable, given a name.
    /// If the variable is not in scope, return false.
    // TODO: return an error no?
    pub fn get_var(&self, ident: &str) -> &Var {
        let (scope, var) = self
            .vars
            .get(ident)
            .unwrap_or_else(|| panic!("type checking bug: local variable {ident} not found"));
        if !self.is_in_scope(*scope) {
            panic!("type checking bug: local variable not in scope");
        }

        var
    }

    pub fn reassign_var(&mut self, ident: &str, var: Var) {
        // get the scope first, we don't want to modify that
        let (scope, _) = self
            .vars
            .get(ident)
            .expect("type checking bug: local variable for reassigning not found");

        if !self.is_in_scope(*scope) {
            panic!("type checking bug: local variable for reassigning not in scope");
        }

        self.vars.insert(ident.to_string(), (*scope, var));
    }
}
