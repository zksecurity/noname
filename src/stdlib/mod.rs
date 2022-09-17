use std::{collections::HashMap, ops::Neg as _};

use ark_ff::{One as _, Zero};

use crate::{
    circuit_writer::{CircuitWriter, GateKind},
    constants::{Field, Span},
    error::{Error, ErrorKind, Result},
    imports::{FnHandle, FnKind},
    lexer::Token,
    parser::{FnSig, Ident, ParserCtx, Path},
    type_checker::FnInfo,
    var::{ConstOrCell, Var},
};

use self::crypto::CRYPTO_FNS;

pub mod crypto;

#[derive(Clone)]
pub struct ImportedModule {
    pub name: String,
    pub functions: HashMap<String, FnInfo>,
    pub span: Span,
}

impl std::fmt::Debug for ImportedModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ImportedModule {{ name: {:?}, functions: {:?}, span: {:?} }}",
            self.name,
            self.functions.keys(),
            self.span
        )
    }
}

/// Parses the rest of a `use std::` statement. Returns a list of functions to import in the scope.
pub fn parse_std_import<'a>(
    path: &'a Path,
    path_iter: &mut impl Iterator<Item = &'a Ident>,
) -> Result<ImportedModule> {
    let module = path_iter.next().ok_or(Error {
        kind: ErrorKind::StdImport("no module name found"),
        span: path.span,
    })?;

    let mut res = ImportedModule {
        name: module.value.clone(),
        functions: HashMap::new(),
        span: module.span,
    };

    // TODO: make sure we're not importing the same module twice no?
    match module.value.as_ref() {
        "crypto" => {
            let crypto_functions = parse_fn_sigs(&CRYPTO_FNS);
            for func in crypto_functions {
                res.functions.insert(func.sig().name.value.clone(), func);
            }
        }
        _ => {
            return Err(Error {
                kind: ErrorKind::StdImport("unknown module"),
                span: module.span,
            })
        }
    }

    Ok(res)
}

/// Takes a list of function signatures (as strings) and their associated function pointer,
/// returns the same list but with the parsed functions (as [FunctionSig]).
pub fn parse_fn_sigs(fn_sigs: &[(&str, FnHandle)]) -> Vec<FnInfo> {
    let mut functions: Vec<FnInfo> = vec![];
    let ctx = &mut ParserCtx::default();

    for (sig, fn_ptr) in fn_sigs {
        let mut tokens = Token::parse(sig).unwrap();

        let sig = FnSig::parse(ctx, &mut tokens).unwrap();

        functions.push(FnInfo {
            kind: FnKind::BuiltIn(sig, *fn_ptr),
            span: Span::default(),
        });
    }

    functions
}

//
// Builtins or utils (imported by default)
// TODO: give a name that's useful for the user,
//       not something descriptive internally like "builtins"

const ASSERT_FN: &str = "assert(condition: Bool)";
const ASSERT_EQ_FN: &str = "assert_eq(a: Field, b: Field)";

pub const BUILTIN_FNS: [(&str, FnHandle); 2] = [(ASSERT_EQ_FN, assert_eq), (ASSERT_FN, assert)];

/// Asserts that two vars are equal.
fn assert_eq(compiler: &mut CircuitWriter, vars: &[Var], span: Span) -> Result<Option<Var>> {
    // double check (on top of type checker)
    assert_eq!(vars.len(), 2);

    let lhs = vars[0]
        .const_or_cell()
        .expect("assert_eq: lhs is not a constant or cell");
    let rhs = vars[1]
        .const_or_cell()
        .expect("assert_eq: rhs is not a constant or cell");

    match (lhs, rhs) {
        // two constants
        (ConstOrCell::Const(a), ConstOrCell::Const(b)) => {
            if a != b {
                return Err(Error {
                    kind: ErrorKind::AssertionFailed,
                    span,
                });
            }
        }

        // a const and a var
        (ConstOrCell::Const(cst), ConstOrCell::Cell(cvar))
        | (ConstOrCell::Cell(cvar), ConstOrCell::Const(cst)) => {
            compiler.add_gate(
                "constrain var - cst = 0 to check equality",
                GateKind::DoubleGeneric,
                vec![Some(*cvar)],
                vec![
                    Field::one(),
                    Field::zero(),
                    Field::zero(),
                    Field::zero(),
                    cst.neg(),
                ],
                span,
            );
        }
        (ConstOrCell::Cell(lhs), ConstOrCell::Cell(rhs)) => {
            // TODO: use permutation to check that
            compiler.add_gate(
                "constrain lhs - rhs = 0 to assert that they are equal",
                GateKind::DoubleGeneric,
                vec![Some(*lhs), Some(*rhs)],
                vec![Field::one(), Field::one().neg()],
                span,
            );
        }
    }

    Ok(None)
}

/// Asserts that a condition is true.
fn assert(compiler: &mut CircuitWriter, vars: &[Var], span: Span) -> Result<Option<Var>> {
    // double check (on top of type checker)
    assert_eq!(vars.len(), 1);
    let cond = vars[0]
        .const_or_cell()
        .expect("assert: condition is not a constant or cell");

    match cond {
        ConstOrCell::Const(cst) => {
            assert!(cst.is_one());
        }
        ConstOrCell::Cell(cvar) => {
            // TODO: use permutation to check that
            let zero = Field::zero();
            let one = Field::one();
            compiler.add_gate(
                "constrain 1 - X = 0 to assert that X is true",
                GateKind::DoubleGeneric,
                vec![None, Some(*cvar)],
                // use the constant to constrain 1 - X = 0
                vec![zero, one.neg(), zero, zero, one],
                span,
            );
        }
    }

    Ok(None)
}
