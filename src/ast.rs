use std::collections::HashMap;

use itertools::Itertools;

use crate::{
    error::{Error, ErrorTy},
    parser::{Expr, ExprKind, FunctionSig, Op2, Root, Stmt, TyKind, AST},
    stdlib::{self, utils_functions},
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

impl TryFrom<String> for F {
    type Error = Error;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        let v: i64 = value.parse().unwrap();
        Ok(Self(v))
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
    Hint(Box<dyn Fn() -> F>),

    /// Or it's a constant.
    Constant(F),

    /// Or it's a linear combination of other circuit variables.
    LinearCombination(Vec<(F, Var)>),

    /// A public input
    Public(usize),

    /// A private input
    Private(usize),
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[value]")
    }
}

impl Compiler {
    pub fn compile(ast: AST) -> Result<(), Error> {
        let mut compiler = Compiler::default();
        let scope = &mut Scope::default();

        let mut main_function_observed = false;

        // inject some utility functions in the scope
        // TODO: should we really import them by default?
        {
            let t = utils_functions();
            for (name, sig) in t {
                scope.functions.insert(name, FuncInScope::BuiltIn(sig));
            }
        }

        for root in ast.0 {
            match root {
                // `use crypto::poseidon;`
                Root::Use(path) => {
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
                Root::Function(mut function) => {
                    // TODO: support other functions
                    if function.name != "main" {
                        unimplemented!();
                    }

                    main_function_observed = true;

                    // create public and private inputs
                    for (public, name, typ) in function.arguments {
                        if !matches!(typ.kind, TyKind::Field) {
                            unimplemented!();
                        }

                        // create the variable in the circuit
                        let var = if public {
                            compiler.public_input()
                        } else {
                            compiler.private_input()
                        };

                        // store it in the scope
                        scope.variables.insert(name, var);
                    }

                    // type system pass!!!
                    compiler.fillout_type_info(scope, &mut function.body)?;

                    // compile function
                    compiler.compile_function(scope, function.body)?;
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

    fn fillout_type_info(&mut self, scope: &mut Scope, stmts: &mut [Stmt]) -> Result<(), Error> {
        // only expressions need type info?
        for stmt in stmts {
            match &mut stmt.kind {
                crate::parser::StmtKind::Assign { lhs, ref mut rhs } => {
                    // inferance can be easy: we can do it the Golang way and just use the type that rhs has (in `let` assignments)

                    // but first we need to compute the type of the rhs expression
                    rhs.compute_type(scope);
                    unimplemented!();

                    // ooch... lhs here doesn't even have a type... we can't give it a type o_O
                    // either we replace this String with a Ty
                    // or we keep a map of variables and their types in scope? (ugly, we can't shadow)
                    // so...
                    // but a Ty what? it's not a type, it's like a variable
                    // so Assign { lhs: Variable, rhs: Expr }
                    // with Variable { typ: Option<TyKind> }
                }
                crate::parser::StmtKind::FnCall { name, args } => todo!(),
                crate::parser::StmtKind::Return(_) => {
                    // infer the return type and check if it's the same as the function return type?
                    unimplemented!();
                }
                crate::parser::StmtKind::Comment(_) => (),
            }
        }

        Ok(())
    }

    fn compile_function(&mut self, scope: &mut Scope, stmts: Vec<Stmt>) -> Result<(), Error> {
        for stmt in stmts {
            match stmt.kind {
                crate::parser::StmtKind::Assign { lhs, rhs } => {
                    // compute the rhs
                    let var = self.compute_expr(scope, *rhs)?.unwrap();

                    // store the new variable
                    scope.variables.insert(lhs.clone(), var);
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
                    for arg in &args {
                        let arg = &**arg;
                        let var = self.compute_expr(scope, arg.clone())?.ok_or(Error {
                            error: ErrorTy::CannotComputeExpression,
                            span: arg.span,
                        })?;
                        vars.push(var);
                    }

                    // check if it's the scope
                    match scope.functions.get(&name) {
                        None => {
                            return Err(Error {
                                error: ErrorTy::UnknownFunction(name),
                                span: stmt.span,
                            })
                        }
                        Some(FuncInScope::BuiltIn(sig)) => {
                            // compute the expressions

                            // compare sig
                            if sig.arguments.len() != args.len() {
                                return Err(Error {
                                    error: ErrorTy::WrongNumberOfArguments {
                                        name,
                                        expected_args: sig.arguments.len(),
                                        observed_args: args.len(),
                                    },
                                    span: stmt.span,
                                });
                            }

                            for ((_pub, arg_name, typ), arg) in sig.arguments.iter().zip_eq(args) {
                                let arg_typ = arg.typ.unwrap();
                                if typ.kind != arg_typ {
                                    return Err(Error {
                                        error: ErrorTy::WrongArgumentType {
                                            fn_name: name,
                                            arg_name: arg_name.clone(),
                                            expected_ty: typ.kind.to_string(),
                                            observed_ty: arg_typ.to_string(),
                                        },
                                        span: stmt.span,
                                    });
                                }
                            }

                            // run function
                        }
                        Some(FuncInScope::Library(_, _)) => todo!(),
                    }
                }
                crate::parser::StmtKind::Return(_) => todo!(),
                crate::parser::StmtKind::Comment(_) => todo!(),
            }
        }

        Ok(())
    }

    fn new_internal_var(&mut self, val: Value) -> Var {
        // create new var
        let var = Var(self.next_variable);
        self.next_variable += 1;

        // store it in the compiler
        self.witness_vars.insert(var, val);

        var
    }

    fn compute_expr(&mut self, scope: &mut Scope, expr: Expr) -> Result<Option<Var>, Error> {
        // HOW TO DO THAT XD??
        let var = match expr.kind {
            ExprKind::FnCall {
                function_name,
                args,
            } => todo!(),
            ExprKind::Variable(_) => todo!(),
            ExprKind::Comparison(_, _, _) => todo!(),
            ExprKind::Op(op, lhs, rhs) => match op {
                Op2::Addition => {
                    let lhs = self.compute_expr(scope, *lhs)?.unwrap();
                    let rhs = self.compute_expr(scope, *rhs)?.unwrap();
                    Some(self.add(scope, lhs, rhs))
                }
                Op2::Subtraction => todo!(),
                Op2::Multiplication => todo!(),
                Op2::Division => todo!(),
                Op2::Equality => todo!(),
            },
            ExprKind::Negated(_) => todo!(),
            ExprKind::BigInt(b) => {
                let f = F::try_from(b)?;
                Some(self.new_internal_var(Value::Constant(f)))
            }
            ExprKind::Identifier(name) => {
                let var = scope.variables.get(&name).unwrap();
                Some(*var)
            }
            ExprKind::ArrayAccess(_, _) => todo!(),
        };

        Ok(var)
    }

    fn add(&mut self, scope: &mut Scope, lhs: Var, rhs: Var) -> Var {
        // create a new variable to store the result
        let res = self.new_internal_var(Value::LinearCombination(vec![
            (F::one(), lhs),
            (F::one(), rhs),
        ]));

        self.gates(
            GateKind::DoubleGeneric,
            vec![Some(lhs), Some(rhs), Some(res)],
            vec![F::one(), F::one(), F::one().neg()],
        );

        res
    }

    fn assert_eq(&mut self, lhs: Var, rhs: Var) {
        self.gates(
            GateKind::DoubleGeneric,
            vec![Some(lhs), Some(rhs)],
            vec![F::one(), F::one().neg()],
        );
    }

    pub fn constant(&mut self, value: F) -> Var {
        self.new_internal_var(Value::Constant(value))
    }

    /// creates a new gate, and the associated row in the witness/execution trace.
    pub fn gates(&mut self, typ: GateKind, vars: Vec<Option<Var>>, coeffs: Vec<F>) {
        assert!(coeffs.len() <= COLUMNS);
        assert!(vars.len() <= COLUMNS);
        self.witness_rows.push(vars);
        self.gates.push(Gate { typ, coeffs })
    }

    pub fn public_input(&mut self) -> Var {
        // create the var
        let var = self.new_internal_var(Value::Public(self.public_input_size));
        self.public_input_size += 1;

        // create the associated generic gate
        self.gates(GateKind::DoubleGeneric, vec![Some(var)], vec![F::one()]);

        var
    }

    pub fn private_input(&mut self) -> Var {
        // create the var
        let var = self.new_internal_var(Value::Private(self.private_input_size));
        self.private_input_size += 1;

        var
    }
}

// TODO: right now there's only one scope, but if we want to deal with multiple scopes then we'll need to make sure child scopes have access to parent scope, shadowing, etc.
#[derive(Default)]
pub struct Scope {
    pub variables: HashMap<String, Var>,
    pub functions: HashMap<String, FuncInScope>,
    pub types: Vec<String>,
}

impl Scope {
    pub fn get_type(&self, ident: &str) -> Option<&TyKind> {
        self.variables.get(name)
    }
}

pub enum FuncInScope {
    /// signature of the function
    BuiltIn(FunctionSig),
    /// path, and signature of the function
    Library(Vec<String>, FunctionSig),
}
