use std::collections::HashMap;

use crate::{
    parser::{Expr, ExprKind, FunctionSig, Op2, Root, Stmt, AST},
    stdlib,
};

//
// Constants
//

pub const COLUMNS: usize = 15;

//
// Mocking the field for now
//

/// We'll probably want to hardcode the field no?
#[derive(Debug)]
pub struct F(i64);

impl F {
    pub fn zero() -> Self {
        F(0)
    }

    pub fn one() -> Self {
        F(1)
    }

    pub fn neg(self) -> Self {
        Self(-self.0)
    }
}

//
// Data structures
//

#[derive(Debug)]
pub enum GateKind {
    DoubleGeneric,
    Poseidon,
}

#[derive(Debug)]
pub struct Gate {
    /// Type of gate
    typ: GateKind,

    /// col -> (row, col)
    // TODO: do we want to do an external wiring instead?
    //    wiring: HashMap<u8, (u64, u8)>,

    /// Coefficients
    coeffs: Vec<F>,
}

#[derive(Default, Debug)]
pub struct Compiler {
    /// Once this is set, you can generate a witness (and can't modify the circuit?)
    // TODO: is this useful?
    pub finalized: bool,

    ///
    pub next_variable: usize,

    /// This is how you compute the value of each variable, for witness generation.
    pub witness_vars: HashMap<Var, Value>,

    /// This can be used to compute the witness.
    witness_rows: Vec<Vec<Option<Var>>>,

    /// The gates created by the circuit
    // TODO: replace by enum and merge with finalized?
    gates: Vec<Gate>,

    /// Size of the public input.
    pub public_size_input: usize,
    // Wiring here? or inside gate?
    // pub wiring: ()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Var(usize);

/// A variable's actual value in the witness can be computed in different ways.
pub enum Value {
    /// Either it's a hint and can be computed from the outside.
    Hint(Box<dyn Fn() -> F>),

    /// Or it's a constant.
    Constant(F),

    /// Or it's a linear combination of other circuit variables.
    LinearCombination(Vec<(F, Var)>),
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[value]")
    }
}

impl Compiler {
    pub fn compile(ast: AST) -> Result<(), &'static str> {
        let mut compiler = Compiler::default();
        let scope = &mut Scope::default();

        let mut main_function_observed = false;

        for root in ast.0 {
            match root {
                // `use crypto::poseidon;`
                Root::Use(path) => {
                    let path = &mut path.0.into_iter();
                    let root_module = path.next().expect("empty imports can't be parsed");

                    let (functions, types) = if root_module == "std" {
                        stdlib::parse_std_import(path)?
                    } else {
                        unimplemented!()
                    };

                    scope.functions.extend(functions);
                    scope.types.extend(types);
                }

                // `fn main() { ... }`
                Root::Function(function) => {
                    // TODO: support other functions
                    if function.name != "main" {
                        unimplemented!();
                    }

                    main_function_observed = true;

                    compiler.analyze_function(scope, function.body);
                }

                // ignore comments
                Root::Comment(_comment) => (),
            }
        }

        // enforce that there's a main function
        assert!(main_function_observed);

        println!("asm: {:#?}", compiler);

        Ok(())
    }

    fn analyze_function(&mut self, scope: &mut Scope, stmts: Vec<Stmt>) {
        for stmt in stmts {
            match stmt.typ {
                crate::parser::StmtKind::Assign { lhs, rhs } => {
                    // compute the rhs
                    let var = self.compute_expr(scope, rhs).unwrap();

                    // store the new variable
                    scope.variables.insert(lhs.clone(), var);
                }
                crate::parser::StmtKind::Assert(_) => todo!(),
                crate::parser::StmtKind::Return(_) => todo!(),
                crate::parser::StmtKind::Comment(_) => todo!(),
            }
        }
    }

    fn new_internal_var(&mut self, val: Value) -> Var {
        // create new var
        let var = Var(self.next_variable);
        self.next_variable += 1;

        // store it in the compiler
        self.witness_vars.insert(var, val);

        var
    }

    fn compute_expr(&mut self, scope: &mut Scope, expr: Expr) -> Option<Var> {
        // HOW TO DO THAT XD??
        match expr.typ {
            ExprKind::FnCall {
                function_name,
                args,
            } => todo!(),
            ExprKind::Variable(_) => todo!(),
            ExprKind::Comparison(_, _, _) => todo!(),
            ExprKind::Op(op, lhs, rhs) => match op {
                Op2::Addition => {
                    let lhs = self.compute_expr(scope, *lhs).unwrap();
                    let rhs = self.compute_expr(scope, *rhs).unwrap();
                    Some(self.add(scope, lhs, rhs))
                }
                Op2::Subtraction => todo!(),
                Op2::Multiplication => todo!(),
                Op2::Division => todo!(),
                Op2::Equality => todo!(),
            },
            ExprKind::Negated(_) => todo!(),
            ExprKind::BigInt(_) => todo!(),
            ExprKind::Identifier(_) => todo!(),
            ExprKind::ArrayAccess(_, _) => todo!(),
        }
    }

    fn add(&mut self, scope: &mut Scope, lhs: Var, rhs: Var) -> Var {
        // create a new variable to store the result
        let res = self.new_internal_var(Value::LinearCombination(vec![
            (F::one(), lhs),
            (F::one(), rhs),
        ]));

        self.gates(
            GateKind::DoubleGeneric,
            vec![F::one(), F::one(), F::one().neg()],
        );

        self.witness_rows(vec![Some(lhs), Some(rhs), Some(res)]);

        res
    }

    pub fn witness_rows(&mut self, vars: Vec<Option<Var>>) {
        assert!(vars.len() <= COLUMNS);
        self.witness_rows.push(vars);
    }

    pub fn gates(&mut self, typ: GateKind, coeffs: Vec<F>) {
        assert!(coeffs.len() <= COLUMNS);
        self.gates.push(Gate { typ, coeffs })
    }
}

#[derive(Default)]
struct Scope {
    pub variables: HashMap<String, Var>,
    pub functions: Vec<FunctionSig>,
    pub types: Vec<String>,
}

impl Scope {}
