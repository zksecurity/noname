use std::{collections::HashMap, ops::Neg, vec};

use ark_ff::{One, PrimeField, Zero};
use itertools::Itertools as _;
use num_bigint::BigUint;
use num_traits::Num as _;

use crate::{
    asm,
    constants::{Field, Span, COLUMNS},
    error::{Error, ErrorTy},
    parser::{Expr, ExprKind, Function, FunctionSig, Op2, RootKind, Stmt, TyKind, AST},
    stdlib::utils_functions,
};

//
// Data structures
//

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

    pub fn scramble(&mut self) {
        self.0[3][0] = Field::from(8u64);
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

#[derive(Debug, Clone, Copy)]
pub struct Cell {
    pub row: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub enum Wiring {
    NotWired(Cell),
    Wired(Vec<Cell>),
}

#[derive(Default, Debug)]
pub struct Compiler {
    pub source: String,

    /// Once this is set, you can generate a witness (and can't modify the circuit?)
    // TODO: is this useful?
    pub finalized: bool,

    ///
    pub next_variable: usize,

    /// This is how you compute the value of each variable, for witness generation.
    pub witness_vars: HashMap<Var, Value>,

    pub wiring: HashMap<Var, Wiring>,

    /// This can be used to compute the witness.
    witness_rows: Vec<Vec<Option<Var>>>,

    /// the arguments expected by main
    pub main_args: HashMap<String, TyKind>,

    /// The gates created by the circuit
    // TODO: replace by enum and merge with finalized?
    gates: Vec<Gate>,

    /// Size of the public input.
    pub public_input_size: usize,

    /// Size of the private input.
    // TODO: bit weird isn't it?
    pub private_input_size: usize,
    // Wiring here? or inside gate?
    // pub wiring: ()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Var(usize);

/// A variable's actual value in the witness can be computed in different ways.
pub enum Value {
    /// Either it's a hint and can be computed from the outside.
    Hint(Box<dyn Fn() -> Field>),

    /// Or it's a constant.
    Constant(Field),

    /// Or it's a linear combination of internal circuit variables.
    LinearCombination(Vec<(Field, Var)>),

    /// A public or private input to the function
    External(String),
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[value]")
    }
}

impl Compiler {
    // TODO: perhaps don't return Self, but return a new type that only contains what's necessary to create the witness?
    pub fn analyze_and_compile(mut ast: AST, code: &str) -> Result<(String, Self), Error> {
        let mut compiler = Compiler::default();
        compiler.source = code.to_string();

        let env = &mut Environment::default();

        // inject some utility functions in the scope
        // TODO: should we really import them by default?
        {
            let t = utils_functions();
            for (sig, func) in t {
                env.functions
                    .insert(sig.name.clone(), FuncInScope::BuiltIn(sig, func));
            }
        }

        compiler.type_check(env, &mut ast)?;

        compiler.compile(env, &ast)
    }

    pub fn compiled_gates(&self) -> &[Gate] {
        if !self.finalized {
            panic!("Circuit not finalized yet!");
        }
        &self.gates
    }

    fn compile(mut self, env: &mut Environment, ast: &AST) -> Result<(String, Self), Error> {
        for root in &ast.0 {
            match &root.kind {
                // `use crypto::poseidon;`
                RootKind::Use(_path) => {
                    unimplemented!();
                }

                // `fn main() { ... }`
                RootKind::Function(function) => {
                    // create public and private inputs
                    for (public, name, typ) in &function.arguments {
                        // create the variable in the circuit
                        let var = if *public {
                            self.public_input(name.clone(), typ.span)
                        } else {
                            self.private_input(name.clone())
                        };

                        // store it in the env
                        env.variables.insert(name.clone(), var);
                    }

                    // compile function
                    self.compile_function(env, &function)?;
                }

                // ignore comments
                RootKind::Comment(_comment) => (),
            }
        }

        self.finalized = true;

        Ok((self.asm(), self))
    }

    fn type_check(&mut self, env: &mut Environment, ast: &mut AST) -> Result<(), Error> {
        let mut main_function_observed = false;
        //
        // Semantic analysis includes:
        // - type checking
        // - ?
        //

        for root in &mut ast.0 {
            match &root.kind {
                // `use crypto::poseidon;`
                RootKind::Use(path) => {
                    unimplemented!();
                    let path = &mut path.0.into_iter();
                    let root_module = path.next().expect("empty imports can't be parsed");

                    /*
                    let (functions, types) = if root_module == "std" {
                        stdlib::parse_std_import(path)?
                    } else {
                        unimplemented!()
                    };

                    scope.functions.extend(functions);
                    scope.types.extend(types);
                    */
                }

                // `fn main() { ... }`
                RootKind::Function(function) => {
                    // TODO: support other functions
                    if function.name != "main" {
                        unimplemented!();
                    }

                    main_function_observed = true;

                    // create public and private inputs
                    for (public, name, typ) in &function.arguments {
                        if !matches!(typ.kind, TyKind::Field) {
                            unimplemented!();
                        }

                        // store it in the env
                        env.var_types.insert(name.clone(), typ.kind.clone());

                        //
                        self.main_args.insert(name.clone(), typ.kind.clone());
                    }

                    // type system pass!!!
                    self.type_check_fn(env, &function.body)?;
                }

                // ignore comments
                RootKind::Comment(_comment) => (),
            }
        }

        // enforce that there's a main function
        assert!(main_function_observed);

        Ok(())
    }

    fn type_check_fn(&mut self, env: &mut Environment, stmts: &[Stmt]) -> Result<(), Error> {
        // only expressions need type info?
        for stmt in stmts {
            match &stmt.kind {
                crate::parser::StmtKind::Assign { lhs, rhs } => {
                    // inferance can be easy: we can do it the Golang way and just use the type that rhs has (in `let` assignments)

                    // but first we need to compute the type of the rhs expression
                    let typ = rhs.compute_type(env)?.unwrap();

                    // store the type of lhs in the env
                    env.store_type(lhs.clone(), typ);
                }
                crate::parser::StmtKind::FnCall { name, args } => {
                    // compute the arguments
                    let mut typs = Vec::with_capacity(args.len());
                    for arg in args {
                        if let Some(typ) = arg.compute_type(env)? {
                            typs.push((typ.clone(), arg.span));
                        } else {
                            return Err(Error {
                                error: ErrorTy::CannotComputeExpression,
                                span: arg.span,
                            });
                        }
                    }

                    // check if it's the env
                    match env.functions.get(name) {
                        None => {
                            // TODO: type checking already checked that
                            return Err(Error {
                                error: ErrorTy::UnknownFunction(name.clone()),
                                span: stmt.span,
                            });
                        }
                        Some(FuncInScope::BuiltIn(sig, _func)) => {
                            // argument length
                            if sig.arguments.len() != typs.len() {
                                return Err(Error {
                                    error: ErrorTy::WrongNumberOfArguments {
                                        fn_name: name.clone(),
                                        expected_args: sig.arguments.len(),
                                        observed_args: typs.len(),
                                    },
                                    span: stmt.span,
                                });
                            }

                            // compare argument types with the function signature
                            for ((_, _, typ1), (typ2, span)) in sig.arguments.iter().zip(typs) {
                                if typ1.kind != typ2 {
                                    // it's ok if a bigint is supposed to be a field no?
                                    // TODO: replace bigint -> constant?
                                    if matches!(
                                        (&typ1.kind, &typ2),
                                        (TyKind::Field, TyKind::BigInt)
                                            | (TyKind::BigInt, TyKind::Field)
                                    ) {
                                        continue;
                                    }

                                    return Err(Error {
                                        error: ErrorTy::ArgumentTypeMismatch(
                                            typ1.kind.clone(),
                                            typ2,
                                        ),
                                        span,
                                    });
                                }
                            }

                            // make sure the function does not return any type
                            // (it's a statement, it should only work via side effect)
                            if sig.return_type.is_some() {
                                return Err(Error {
                                    error: ErrorTy::FunctionReturnsType(name.clone()),
                                    span: stmt.span,
                                });
                            }
                        }
                        Some(FuncInScope::Library(_, _)) => todo!(),
                    }
                }
                crate::parser::StmtKind::Return(_) => {
                    // TODO: warn if there's code after the return?

                    // infer the return type and check if it's the same as the function return type?
                    unimplemented!();
                }
                crate::parser::StmtKind::Comment(_) => (),
            }
        }

        Ok(())
    }

    fn compile_function(
        &mut self,
        env: &mut Environment,
        function: &Function,
    ) -> Result<(), Error> {
        let in_main = function.name == "main";

        for stmt in &function.body {
            match &stmt.kind {
                crate::parser::StmtKind::Assign { lhs, rhs } => {
                    // compute the rhs
                    let var = self.compute_expr(env, rhs)?.unwrap();

                    // store the new variable
                    // TODO: do we really need to store that in the scope? That's not an actual var in the scope that's an internal var...
                    env.variables.insert(lhs.clone(), var);
                }
                /*
                crate::parser::StmtKind::Assert(expr) => {
                    let lhs = self.compute_expr(scope, expr).unwrap();
                    let one = self.constant(F::one());
                    self.assert_eq(lhs, one);
                }
                */
                crate::parser::StmtKind::FnCall { name, args } => {
                    // compute the arguments
                    let mut vars = Vec::with_capacity(args.len());
                    for arg in args {
                        let var = self.compute_expr(env, arg)?.ok_or(Error {
                            error: ErrorTy::CannotComputeExpression,
                            span: arg.span,
                        })?;
                        vars.push(var);
                    }

                    // check if it's the scope
                    match env.functions.get(name) {
                        None => {
                            return Err(Error {
                                error: ErrorTy::UnknownFunction(name.clone()),
                                span: stmt.span,
                            })
                        }
                        Some(FuncInScope::BuiltIn(sig, func)) => {
                            // run function
                            func(self, &vars, stmt.span);
                        }
                        Some(FuncInScope::Library(_, _)) => todo!(),
                    }
                }
                crate::parser::StmtKind::Return(_) => {
                    if in_main {
                        return Err(Error {
                            error: ErrorTy::ReturnInMain,
                            span: stmt.span,
                        });
                    }

                    todo!();
                }
                crate::parser::StmtKind::Comment(_) => todo!(),
            }
        }

        Ok(())
    }

    //     pub fn constrain(compiler: &mut Compiler)

    // TODO: how to pass arguments?
    pub fn generate_witness(&self, args: HashMap<&str, Field>) -> Result<Witness, Error> {
        let mut witness = vec![];
        let mut env = WitnessEnv::default();

        // create the argument's variables?
        for (name, typ) in &self.main_args {
            // TODO: support more types
            assert_eq!(typ, &TyKind::Field);

            let val = args.get(name.as_str()).ok_or(Error {
                error: ErrorTy::MissingArg(name.clone()),
                span: (0, 0),
            })?;

            env.add_value(name.clone(), *val);
        }

        for row in &self.witness_rows {
            // create the witness row
            let mut res = [Field::zero(); COLUMNS];
            for (col, var) in row.iter().enumerate() {
                let val = if let Some(var) = var {
                    self.compute_var(&env, *var)
                } else {
                    Field::zero()
                };
                res[col] = val;
            }

            //
            witness.push(res);
        }

        Ok(Witness(witness))
    }

    pub fn asm(&self) -> String {
        asm::generate_asm(&self.source, &self.gates)
    }

    fn new_internal_var(&mut self, val: Value) -> Var {
        // create new var
        let var = Var(self.next_variable);
        self.next_variable += 1;

        // store it in the compiler
        self.witness_vars.insert(var, val);

        var
    }

    fn compute_expr(&mut self, env: &Environment, expr: &Expr) -> Result<Option<Var>, Error> {
        // TODO: why would we return a Var, when types could be represented by several vars?
        // I guess for the moment we're only dealing with Field...
        let var = match &expr.kind {
            ExprKind::FnCall {
                function_name,
                args,
            } => todo!(),
            ExprKind::Variable(_) => todo!(),
            ExprKind::Comparison(_, _, _) => todo!(),
            ExprKind::Op(op, lhs, rhs) => match op {
                Op2::Addition => {
                    let lhs = self.compute_expr(env, lhs)?.unwrap();
                    let rhs = self.compute_expr(env, rhs)?.unwrap();
                    Some(self.add(lhs, rhs, expr.span))
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
                    error: ErrorTy::CannotConvertToField(b.to_string()),
                    span: expr.span,
                })?;

                Some(self.constant(f, expr.span))
            }
            ExprKind::Identifier(name) => {
                let var = env.get_var(&name).unwrap();
                Some(var)
            }
            ExprKind::ArrayAccess(_, _) => todo!(),
        };

        Ok(var)
    }

    pub fn compute_var(&self, env: &WitnessEnv, var: Var) -> Field {
        match &self.witness_vars[&var] {
            Value::Hint(func) => func(),
            Value::Constant(c) => *c,
            Value::LinearCombination(lc) => {
                let mut res = Field::zero();
                for (coeff, var) in lc {
                    res += self.compute_var(env, *var) * *coeff;
                }
                res
            }
            Value::External(name) => env.get_external(name),
        }
    }

    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }

    fn add(&mut self, lhs: Var, rhs: Var, span: Span) -> Var {
        // create a new variable to store the result
        let res = self.new_internal_var(Value::LinearCombination(vec![
            (Field::one(), lhs),
            (Field::one(), rhs),
        ]));

        // wire the lhs and rhs to where they're really from
        /*
        let res = match (&self.witness_vars[&lhs], &self.witness_vars[&rhs]) {
            (Value::Hint(_), _) => todo!(),
            (Value::Constant(a), Value::Constant(b)) => self.constant(*a + *b, span),
            (Value::Constant(_), _) | (_, Value::Constant(_)) => {
                self.new_internal_var(Value::LinearCombination(vec![
                    (Field::one(), lhs),
                    (Field::one(), rhs),
                ]))
            }
            (Value::LinearCombination(_), _) => todo!(),
            (Value::External(_), _) => todo!(),
        };
        */

        // create a gate to store the result
        self.gates(
            GateKind::DoubleGeneric,
            vec![Some(lhs), Some(rhs), Some(res)],
            vec![Field::one(), Field::one(), Field::one().neg()],
            span,
        );

        res
    }

    pub fn constant(&mut self, value: Field, span: Span) -> Var {
        let var = self.new_internal_var(Value::Constant(value));

        let zero = Field::zero();
        self.gates(
            GateKind::DoubleGeneric,
            vec![Some(var)],
            vec![Field::one(), zero, zero, zero, value.neg()],
            span,
        );

        var
    }

    /// creates a new gate, and the associated row in the witness/execution trace.
    // TODO: add_gate instead of gates?
    pub fn gates(&mut self, typ: GateKind, vars: Vec<Option<Var>>, coeffs: Vec<Field>, span: Span) {
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

    pub fn public_input(&mut self, name: String, span: Span) -> Var {
        // create the var
        let var = self.new_internal_var(Value::External(name));
        self.public_input_size += 1;

        // create the associated generic gate
        self.gates(
            GateKind::DoubleGeneric,
            vec![Some(var)],
            vec![Field::one()],
            span,
        );

        var
    }

    pub fn private_input(&mut self, name: String) -> Var {
        // create the var
        let var = self.new_internal_var(Value::External(name));

        // TODO: do we really need this?
        self.private_input_size += 1;

        var
    }
}

// TODO: right now there's only one scope, but if we want to deal with multiple scopes then we'll need to make sure child scopes have access to parent scope, shadowing, etc.
#[derive(Default, Debug)]
pub struct Environment {
    pub var_types: HashMap<String, TyKind>,
    pub variables: HashMap<String, Var>,
    pub functions: HashMap<String, FuncInScope>,
    pub types: Vec<String>,
}

impl Environment {
    pub fn store_type(&mut self, ident: String, ty: TyKind) {
        self.var_types.insert(ident, ty);
    }

    pub fn get_type(&self, ident: &str) -> Option<&TyKind> {
        self.var_types.get(ident)
    }

    pub fn get_var(&self, ident: &str) -> Option<Var> {
        self.variables.get(ident).cloned()
    }
}

pub type FuncType = fn(&mut Compiler, &[Var], Span);

pub enum FuncInScope {
    /// signature of the function
    BuiltIn(FunctionSig, FuncType),
    /// path, and signature of the function
    Library(Vec<String>, FunctionSig),
}

impl std::fmt::Debug for FuncInScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BuiltIn(arg0, _arg1) => f.debug_tuple("BuiltIn").field(arg0).field(&"_").finish(),
            Self::Library(arg0, arg1) => f.debug_tuple("Library").field(arg0).field(arg1).finish(),
        }
    }
}

#[derive(Debug, Default)]
pub struct WitnessEnv {
    pub var_values: HashMap<String, Field>,
}

impl WitnessEnv {
    pub fn add_value(&mut self, name: String, value: Field) {
        self.var_values.insert(name, value);
    }

    pub fn get_external(&self, name: &str) -> Field {
        self.var_values.get(name).unwrap().clone()
    }
}
