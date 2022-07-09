use std::{
    collections::HashMap,
    fmt::{self, Display, Formatter},
    ops::Neg,
    vec,
};

use ark_ff::{One, PrimeField, Zero};
use itertools::Itertools as _;
use num_bigint::BigUint;
use num_traits::Num as _;

use crate::{
    asm,
    constants::{Span, COLUMNS},
    error::{Error, ErrorKind},
    field::{Field, PrettyField as _},
    parser::{
        AttributeKind, Expr, ExprKind, FuncArg, Function, FunctionSig, Ident, Op2, Path, RootKind,
        Stmt, StmtKind, Ty, TyKind, AST,
    },
    stdlib::{self, parse_fn_sigs, ImportedModule, BUILTIN_FNS},
};

//
// Data structures
//

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
// TODO: should a var also contain a span?
/// An internal variable is a variable that is created from a linear combination of external variables.
/// It, most of the time, ends up being a cell in the circuit.
/// That is, unless it's unused?
pub struct CellVar(usize);

/// A variable's actual value in the witness can be computed in different ways.
pub enum Value {
    /// Either it's a hint and can be computed from the outside.
    Hint(Box<dyn Fn(&Compiler, &WitnessEnv) -> Result<Field, Error>>),

    /// Or it's a constant.
    Constant(Field),

    /// Or it's a linear combination of internal circuit variables (+ a constant).
    // TODO: probably values of internal variables should be cached somewhere
    LinearCombination(Vec<(Field, CellVar)>, Field),

    /// A public or private input to the function
    /// There's an index associated to a variable name, as the variable could be composed of several field elements.
    External(String, usize),

    PublicOutput(Option<CellVar>),
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[value]")
    }
}

/// The primitive type in a circuit is a field element.
/// But since we want to be able to deal with custom types
/// that are built on top of field elements,
/// we abstract any variable in the circuit as a [CircuitVar]
/// which can contain one or more variables.
#[derive(Debug, Clone)]
pub struct CircuitVar {
    pub vars: Vec<CellVar>,
}

impl CircuitVar {
    pub fn new(vars: Vec<CellVar>) -> Self {
        Self { vars }
    }

    pub fn len(&self) -> usize {
        self.vars.len()
    }

    pub fn var(&self, i: usize) -> Option<CellVar> {
        self.vars.get(i).cloned()
    }
}

/// Represents an expression or variable in a program
#[derive(Debug, Clone)]
pub enum Var {
    /// Either a constant
    Constant(Field),
    /// Or a [CircuitVar]
    CircuitVar(CircuitVar),
}

impl Var {
    pub fn is_constant(&self) -> bool {
        matches!(self, Var::Constant(..))
    }

    pub fn new_circuit_var(vars: Vec<CellVar>) -> Self {
        Var::CircuitVar(CircuitVar::new(vars))
    }

    pub fn new_constant(value: Field) -> Self {
        Var::Constant(value)
    }
}

/// the equivalent of [CircuitVar] but for witness generation
#[derive(Debug, Clone)]
pub struct CircuitValue {
    pub values: Vec<Field>,
}

impl CircuitValue {
    pub fn new(values: Vec<Field>) -> Self {
        Self { values }
    }
}

pub struct Witness(Vec<[Field; COLUMNS]>);

impl Witness {
    /// kimchi uses a transposed witness
    pub fn to_kimchi_witness(&self) -> [Vec<Field>; COLUMNS] {
        let transposed = vec![Vec::with_capacity(self.0.len()); COLUMNS];
        let mut transposed: [_; COLUMNS] = transposed.try_into().unwrap();
        for row in &self.0 {
            for (col, field) in row.iter().enumerate() {
                transposed[col].push(*field);
            }
        }
        transposed
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn debug(&self) {
        for (row, values) in self.0.iter().enumerate() {
            let values = values.iter().map(|v| v.pretty()).join(" | ");
            println!("{row} - {values}");
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GateKind {
    DoubleGeneric,
    Poseidon,
}

impl Into<kimchi::circuits::gate::GateType> for GateKind {
    fn into(self) -> kimchi::circuits::gate::GateType {
        use kimchi::circuits::gate::GateType::*;
        match self {
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
}

impl Gate {
    pub fn to_kimchi_gate(&self, row: usize) -> kimchi::circuits::gate::CircuitGate<Field> {
        kimchi::circuits::gate::CircuitGate {
            typ: self.typ.into(),
            // TODO: wiring!!
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
    NotWired(Cell),
    Wired(Vec<Cell>),
}

//
// Compiler
//

#[derive(Default, Debug)]
pub struct Compiler {
    /// The source code that created this circuit. Useful for debugging and displaying errors.
    pub source: String,

    /// Once this is set, you can generate a witness (and can't modify the circuit?)
    // TODO: is this useful?
    pub finalized: bool,

    /// This is used to give a distinct number to each variable.
    pub next_variable: usize,

    /// This is how you compute the value of each variable, for witness generation.
    pub witness_vars: HashMap<CellVar, Value>,

    /// The wiring of the circuit.
    pub wiring: HashMap<CellVar, Wiring>,

    /// This is used to compute the witness row by row.
    witness_rows: Vec<Vec<Option<CellVar>>>,

    /// the arguments expected by main (I think it's used by the witness generator to make sure we passed the arguments)
    pub main_args: HashMap<String, TyKind>,

    /// The gates created by the circuit
    // TODO: replace by enum and merge with finalized?
    gates: Vec<Gate>,

    /// Size of the public input.
    pub public_input_size: usize,

    /// If a public output is set, this will be used to store its [CircuitVar] (cvar).
    /// The public output generation works as follows:
    /// 1. This cvar is created and inserted in the circuit (gates) during compilation of the public input
    ///    (as the public output is the end of the public input)
    /// 2. When the `return` statement of the circuit is parsed,
    ///    it will set this `public_output` variable again to the correct vars.
    /// 3. During witness generation, the public output computation
    ///    is delayed until the very end.
    pub public_output: Option<CircuitVar>,

    /// Size of the private input.
    // TODO: bit weird isn't it?
    pub private_input_size: usize,
}

impl Compiler {
    // TODO: perhaps don't return Self, but return a new type that only contains what's necessary to create the witness?
    pub fn analyze_and_compile(
        mut ast: AST,
        code: &str,
        debug: bool,
    ) -> Result<(String, Self), Error> {
        let mut compiler = Compiler::default();
        compiler.source = code.to_string();

        let env = &mut Environment::default();

        // inject some utility functions in the scope
        // TODO: should we really import them by default?
        {
            let builtin_functions = parse_fn_sigs(&BUILTIN_FNS);
            for (sig, func) in builtin_functions {
                env.functions
                    .insert(sig.name.value.clone(), FuncInScope::BuiltIn(sig, func));
            }
        }

        // this will type check everything
        compiler.type_check(env, &mut ast)?;

        // this will convert everything to {gates, wiring, witness_vars}
        compiler.compile(env, &ast, debug)
    }

    pub fn compiled_gates(&self) -> &[Gate] {
        if !self.finalized {
            panic!("Circuit not finalized yet!");
        }
        &self.gates
    }

    fn compile(
        mut self,
        env: &mut Environment,
        ast: &AST,
        debug: bool,
    ) -> Result<(String, Self), Error> {
        for root in &ast.0 {
            match &root.kind {
                // `use crypto::poseidon;`
                RootKind::Use(_path) => {
                    // we already dealt with that in the type check pass
                    ()
                }

                // `fn main() { ... }`
                RootKind::Function(function) => {
                    if !function.is_main() {
                        unimplemented!();
                    }

                    // create public and private inputs
                    for FuncArg {
                        attribute,
                        name,
                        typ,
                        ..
                    } in &function.arguments
                    {
                        let len = match &typ.kind {
                            TyKind::Field => 1,
                            TyKind::Array(typ, len) => {
                                if !matches!(**typ, TyKind::Field) {
                                    unimplemented!();
                                }
                                *len as usize
                            }
                            _ => unimplemented!(),
                        };

                        let cvar = if let Some(attr) = attribute {
                            if !matches!(attr.kind, AttributeKind::Pub) {
                                return Err(Error {
                                    kind: ErrorKind::InvalidAttribute(attr.kind),
                                    span: attr.span,
                                });
                            }
                            self.public_inputs(name.value.clone(), len, name.span)
                        } else {
                            self.private_inputs(name.value.clone(), len)
                        };

                        env.variables.insert(name.value.clone(), cvar);
                    }

                    // create public output
                    if let Some(typ) = &function.return_type {
                        if typ.kind != TyKind::Field {
                            unimplemented!();
                        }

                        // create it
                        self.public_outputs(1, typ.span);
                    }

                    // compile function
                    self.compile_function(env, &function)?;
                }

                // ignore comments
                // TODO: we could actually preserve the comment in the ASM!
                RootKind::Comment(_comment) => (),
            }
        }

        self.finalized = true;

        Ok((self.asm(debug), self))
    }

    fn compile_function(
        &mut self,
        env: &mut Environment,
        function: &Function,
    ) -> Result<(), Error> {
        for stmt in &function.body {
            match &stmt.kind {
                StmtKind::Assign { lhs, rhs } => {
                    // compute the rhs
                    let var = self.compute_expr(env, rhs)?.ok_or(Error {
                        kind: ErrorKind::CannotComputeExpression,
                        span: stmt.span,
                    })?;

                    // store the new variable
                    // TODO: do we really need to store that in the scope? That's not an actual var in the scope that's an internal var...
                    env.variables.insert(lhs.value.clone(), var);
                }
                StmtKind::Expr(expr) => {
                    // compute the expression
                    let var = self.compute_expr(env, expr)?;

                    // make sure it does not return any value.
                    assert!(var.is_none());
                }
                StmtKind::Return(res) => {
                    if !function.is_main() {
                        unimplemented!();
                    }

                    let cvar = self.compute_expr(env, res)?.ok_or(Error {
                        kind: ErrorKind::CannotComputeExpression,
                        span: stmt.span,
                    })?;

                    // store the return value in the public input that was created for that ^
                    let public_output = self.public_output.as_ref().ok_or(Error {
                        kind: ErrorKind::NoPublicOutput,
                        span: stmt.span,
                    })?;

                    for (pub_var, ret_var) in public_output.vars.iter().zip(&cvar.vars) {
                        // replace the computation of the public output vars with the actual variables being returned here
                        assert!(self
                            .witness_vars
                            .insert(*pub_var, Value::PublicOutput(Some(ret_var.clone())))
                            .is_some());
                    }
                }
                StmtKind::Comment(_) => todo!(),
            }
        }

        Ok(())
    }

    //     pub fn constrain(compiler: &mut Compiler)

    // TODO: how to pass arguments?
    pub fn generate_witness(
        &self,
        args: HashMap<&str, CircuitValue>,
    ) -> Result<(Witness, Vec<Field>), Error> {
        let mut witness = vec![];
        let mut env = WitnessEnv::default();

        // create the argument's variables?
        for (name, typ) in &self.main_args {
            // TODO: support more types
            assert_eq!(typ, &TyKind::Field);

            let val = args.get(name.as_str()).ok_or(Error {
                kind: ErrorKind::MissingArg(name.clone()),
                span: (0, 0),
            })?;

            env.add_value(name.clone(), val.clone());
        }

        // compute each rows' vars, except for the deferred ones (public output)
        let mut public_outputs_vars: Vec<(usize, CellVar)> = vec![];

        for (row, witness_row) in self.witness_rows.iter().enumerate() {
            // create the witness row
            let mut res = [Field::zero(); COLUMNS];
            for (col, var) in witness_row.iter().enumerate() {
                let val = if let Some(var) = var {
                    // if it's a public output, defer it's computation
                    if matches!(self.witness_vars[&var], Value::PublicOutput(_)) {
                        public_outputs_vars.push((row, *var));
                        Field::zero()
                    } else {
                        self.compute_var(&env, *var)?
                    }
                } else {
                    Field::zero()
                };
                res[col] = val;
            }

            //
            witness.push(res);
        }

        // compute public output at last
        let mut public_output = vec![];

        for (row, var) in public_outputs_vars {
            let val = self.compute_var(&env, var)?;
            witness[row][0] = val;
            public_output.push(val);
        }

        //
        assert_eq!(witness.len(), self.gates.len());
        Ok((Witness(witness), public_output))
    }

    pub fn asm(&self, debug: bool) -> String {
        asm::generate_asm(&self.source, &self.gates, &self.wiring, debug)
    }

    pub fn new_internal_var(&mut self, val: Value) -> CellVar {
        // create new var
        let var = CellVar(self.next_variable);
        self.next_variable += 1;

        // store it in the compiler
        self.witness_vars.insert(var, val);

        var
    }

    fn compute_expr(&mut self, env: &Environment, expr: &Expr) -> Result<Option<Var>, Error> {
        let cvar = match &expr.kind {
            ExprKind::FnCall { name, args } => {
                // retrieve the function in the env
                let func: FuncType = if name.len() == 1 {
                    // functions present in the scope
                    let fn_name = &name.path[0].value;
                    match env.functions.get(fn_name).ok_or(Error {
                        kind: ErrorKind::UndefinedFunction(fn_name.clone()),
                        span: name.span,
                    })? {
                        crate::ast::FuncInScope::BuiltIn(_, func) => func.clone(),
                        crate::ast::FuncInScope::Library(_, _) => todo!(),
                    }
                } else if name.len() == 2 {
                    // check module present in the scope
                    let module = &name.path[0];
                    let fn_name = &name.path[1];
                    let module = env.modules.get(&module.value).ok_or(Error {
                        kind: ErrorKind::UndefinedModule(module.value.clone()),
                        span: module.span,
                    })?;
                    let (_, func) = module.functions.get(&fn_name.value).ok_or(Error {
                        kind: ErrorKind::UndefinedFunction(fn_name.value.clone()),
                        span: fn_name.span,
                    })?;
                    func.clone()
                } else {
                    return Err(Error {
                        kind: ErrorKind::InvalidFnCall("sub-sub modules unsupported"),
                        span: name.span,
                    });
                };

                // compute the arguments
                let mut vars = Vec::with_capacity(args.len());
                for arg in args {
                    let var = self.compute_expr(env, arg)?.ok_or(Error {
                        kind: ErrorKind::CannotComputeExpression,
                        span: arg.span,
                    })?;
                    vars.push(var);
                }

                // run the function
                func(self, &vars, expr.span)
            }
            ExprKind::Variable(_) => todo!(),
            ExprKind::Comparison(_, _, _) => todo!(),
            ExprKind::Op(op, lhs, rhs) => match op {
                Op2::Addition => {
                    let lhs = self.compute_expr(env, lhs)?.unwrap().vars;
                    let rhs = self.compute_expr(env, rhs)?.unwrap().vars;

                    if lhs.len() != 1 || rhs.len() != 1 {
                        unreachable!();
                    }

                    Some(self.add(lhs[0], rhs[0], expr.span))
                }
                Op2::Subtraction => todo!(),
                Op2::Multiplication => todo!(),
                Op2::Division => todo!(),
                Op2::Equality => todo!(),
            },
            ExprKind::Negated(_) => todo!(),
            ExprKind::BigInt(b) => {
                let biguint = BigUint::from_str_radix(b, 10).expect("failed to parse number.");
                let f = Field::try_from(biguint).map_err(|_| Error {
                    kind: ErrorKind::CannotConvertToField(b.to_string()),
                    span: expr.span,
                })?;

                Some(self.constant(f, expr.span))
            }
            ExprKind::Identifier(name) => {
                let cvar = env.get_cvar(&name).unwrap();
                Some(cvar)
            }
            ExprKind::ArrayAccess(path, expr) => {
                // retrieve the CircuitVar at the path
                let array: CircuitVar = if path.len() == 1 {
                    // var present in the scope
                    let name = &path.path[0].value;
                    env.variables
                        .get(name)
                        .ok_or(Error {
                            kind: ErrorKind::UndefinedVariable,
                            span: path.span,
                        })
                        .cloned()?
                } else if path.len() == 2 {
                    // check module present in the scope
                    let module = &path.path[0];
                    let name = &path.path[1];
                    let module = env.modules.get(&module.value).ok_or(Error {
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
                let idx_var = self.compute_expr(env, expr)?.ok_or(Error {
                    kind: ErrorKind::CannotComputeExpression,
                    span: expr.span,
                })?;
                if idx_var.len() != 1 {
                    return Err(Error {
                        kind: ErrorKind::ExpectedConstant,
                        span: expr.span,
                    });
                }
                let idx_var = idx_var.var(0).unwrap();

                // the index must be a constant!!
                let idx: Field = self.compute_constant(idx_var, expr.span)?;
                let idx: BigUint = idx.into();
                let idx: usize = idx.try_into().unwrap();

                // index into the CircuitVar
                // (and prevent out of bounds)
                let res = array.var(idx);
                if res.is_none() {
                    return Err(Error {
                        kind: ErrorKind::ArrayIndexOutOfBounds(idx, array.len()),
                        span: expr.span,
                    });
                }

                res.map(|var| CircuitVar { vars: vec![var] })
            }
        };

        Ok(cvar)
    }

    pub fn compute_var(&self, env: &WitnessEnv, var: CellVar) -> Result<Field, Error> {
        match &self.witness_vars[&var] {
            Value::Hint(func) => func(self, env),
            Value::Constant(c) => Ok(*c),
            Value::LinearCombination(lc) => {
                let mut res = Field::zero();
                for (coeff, var) in lc {
                    res += self.compute_var(env, *var)? * *coeff;
                }
                Ok(res)
            }
            Value::External(name, idx) => Ok(env.get_external(name)[*idx]),
            Value::PublicOutput(var) => {
                let var = var.ok_or(Error {
                    kind: ErrorKind::MissingPublicOutput,
                    span: (0, 0),
                })?;
                self.compute_var(env, var)
            }
        }
    }

    pub fn compute_constant(&self, var: CellVar, span: Span) -> Result<Field, Error> {
        match &self.witness_vars.get(&var) {
            Some(Value::Constant(c)) => Ok(*c),
            Some(Value::LinearCombination(lc)) => {
                let mut res = Field::zero();
                for (coeff, var) in lc {
                    res += self.compute_constant(*var, span)? * *coeff;
                }
                Ok(res)
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

    fn add(&mut self, lhs: Var, rhs: Var, span: Span) -> Var {
        match (lhs, rhs) {
            (Var::Constant(lhs), Var::Constant(rhs)) => Var::Constant(lhs + rhs),
            (Var::Constant(cst), Var::CircuitVar(var))
            | (Var::CircuitVar(var), Var::Constant(cst)) => {
                if var.len() != 1 {
                    unimplemented!();
                }
                let var = var.var(0).unwrap();

                // create a new variable to store the result
                let res =
                    self.new_internal_var(Value::LinearCombination(vec![(Field::one(), var)], cst));

                // create a gate to store the result
                self.gates(
                    GateKind::DoubleGeneric,
                    vec![Some(var), None, Some(res)],
                    vec![
                        Field::one(),
                        Field::one(),
                        Field::one().neg(),
                        Field::zero(),
                        cst,
                    ],
                    span,
                );

                Var::new_circuit_var(vec![res])
            }
            (Var::CircuitVar(lhs), Var::CircuitVar(rhs)) => {
                if lhs.len() != 1 || rhs.len() != 1 {
                    unimplemented!();
                }
                let lhs = lhs.var(0).unwrap();
                let rhs = rhs.var(0).unwrap();

                // create a new variable to store the result
                let res = self.new_internal_var(Value::LinearCombination(
                    vec![(Field::one(), lhs), (Field::one(), rhs)],
                    Field::zero(),
                ));

                // create a gate to store the result
                self.gates(
                    GateKind::DoubleGeneric,
                    vec![Some(lhs), Some(rhs), Some(res)],
                    vec![Field::one(), Field::one(), Field::one().neg()],
                    span,
                );

                Var::new_circuit_var(vec![res])
            }
        }
    }

    pub fn constant(&mut self, value: Field, span: Span) -> CircuitVar {
        let var = self.new_internal_var(Value::Constant(value));

        let zero = Field::zero();
        self.gates(
            GateKind::DoubleGeneric,
            vec![Some(var)],
            vec![Field::one(), zero, zero, zero, value.neg()],
            span,
        );

        CircuitVar { vars: vec![var] }
    }

    /// creates a new gate, and the associated row in the witness/execution trace.
    // TODO: add_gate instead of gates?
    pub fn gates(
        &mut self,
        typ: GateKind,
        vars: Vec<Option<CellVar>>,
        coeffs: Vec<Field>,
        span: Span,
    ) {
        assert!(coeffs.len() <= COLUMNS);
        assert!(vars.len() <= COLUMNS);
        self.witness_rows.push(vars.clone());
        let row = self.gates.len();
        self.gates.push(Gate { typ, coeffs, span });

        for (col, var) in vars.iter().enumerate() {
            if let Some(var) = var {
                let curr_cell = Cell { row, col };
                self.wiring
                    .entry(*var)
                    .and_modify(|w| match w {
                        Wiring::NotWired(cell) => *w = Wiring::Wired(vec![cell.clone(), curr_cell]),
                        Wiring::Wired(ref mut cells) => cells.push(curr_cell),
                    })
                    .or_insert(Wiring::NotWired(curr_cell));
            }
        }
    }

    pub fn public_inputs(&mut self, name: String, num: usize, span: Span) -> CircuitVar {
        let mut vars = Vec::with_capacity(num);

        for idx in 0..num {
            // create the var
            let var = self.new_internal_var(Value::External(name.clone(), idx));
            vars.push(var.clone());

            // create the associated generic gate
            self.gates(
                GateKind::DoubleGeneric,
                vec![Some(var)],
                vec![Field::one()],
                span,
            );
        }

        self.public_input_size += num;

        CircuitVar { vars }
    }

    pub fn public_outputs(&mut self, num: usize, span: Span) {
        assert!(self.public_output.is_none());

        let mut vars = Vec::with_capacity(num);
        for _ in 0..num {
            // create the var
            let var = self.new_internal_var(Value::PublicOutput(None));
            vars.push(var);

            // create the associated generic gate
            self.gates(
                GateKind::DoubleGeneric,
                vec![Some(var)],
                vec![Field::one()],
                span,
            );
        }
        self.public_input_size += num;

        // store it
        let cvar = CircuitVar { vars };
        self.public_output = Some(cvar);
    }

    pub fn private_inputs(&mut self, name: String, num: usize) -> CircuitVar {
        let mut vars = Vec::with_capacity(num);

        for idx in 0..num {
            // create the var
            let var = self.new_internal_var(Value::External(name.clone(), idx));
            vars.push(var);
        }

        // TODO: do we really need this?
        self.private_input_size += num;

        CircuitVar { vars }
    }
}

// TODO: right now there's only one scope, but if we want to deal with multiple scopes then we'll need to make sure child scopes have access to parent scope, shadowing, etc.
#[derive(Default, Debug)]
pub struct Environment {
    /// created by the type checker, gives a type to every external variable
    pub var_types: HashMap<String, TyKind>,

    /// used by the private and public inputs, and any other external variables created in the circuit
    pub variables: HashMap<String, Var>,

    /// the functions present in the scope
    /// contains at least the set of builtin functions (like assert_eq)
    pub functions: HashMap<String, FuncInScope>,

    /// stores the imported modules
    pub modules: HashMap<String, ImportedModule>,
    pub types: Vec<String>,
}

impl Environment {
    pub fn store_type(&mut self, ident: String, ty: TyKind) {
        self.var_types.insert(ident, ty);
    }

    pub fn get_type(&self, ident: &str) -> Option<&TyKind> {
        self.var_types.get(ident)
    }

    pub fn get_cvar(&self, ident: &str) -> Option<Var> {
        self.variables.get(ident).cloned()
    }
}

pub type FuncType = fn(&mut Compiler, &[CircuitVar], Span) -> Option<CircuitVar>;

pub enum FuncInScope {
    /// signature of the function
    BuiltIn(FunctionSig, FuncType),
    /// path, and signature of the function
    Library(Vec<String>, FunctionSig),
}

impl fmt::Debug for FuncInScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BuiltIn(arg0, _arg1) => f.debug_tuple("BuiltIn").field(arg0).field(&"_").finish(),
            Self::Library(arg0, arg1) => f.debug_tuple("Library").field(arg0).field(arg1).finish(),
        }
    }
}

//
// Witness stuff
//

#[derive(Debug, Default)]
pub struct WitnessEnv {
    pub var_values: HashMap<String, CircuitValue>,
}

impl WitnessEnv {
    pub fn add_value(&mut self, name: String, val: CircuitValue) {
        assert!(self.var_values.insert(name, val).is_none());
    }

    pub fn get_external(&self, name: &str) -> Vec<Field> {
        // TODO: return an error instead of crashing
        self.var_values.get(name).unwrap().clone().values.clone()
    }
}
