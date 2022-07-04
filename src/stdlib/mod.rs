use std::ops::Neg as _;

use ark_ff::One as _;

use crate::{
    ast::{Compiler, Field, FuncType, GateKind, Var},
    lexer::Token,
    parser::{FunctionSig, ParserCtx},
};

pub mod crypto;

pub fn parse_std_import(
    path: &mut impl Iterator<Item = String>,
) -> Result<(Vec<FunctionSig>, Vec<String>), &'static str> {
    let mut functions = vec![];
    let mut types = vec![];

    let module = path.next().ok_or("no module to read")?;

    match module.as_ref() {
        "crypto" => {
            let thing = crypto::parse_crypto_import(path)?;
            // TODO: make sure we're not importing colliding names
            functions.extend(thing.0);
            types.extend(thing.1);
        }
        _ => return Err("unknown module"),
    }

    Ok((functions, types))
}

const ASSERT_FN: &str = "assert(condition: Field)";
const ASSERT_EQ_FN: &str = "assert_eq(a: Field, b: Field)";

pub fn utils_functions() -> Vec<(FunctionSig, FuncType)> {
    let to_parse = [ASSERT_FN, ASSERT_EQ_FN];
    let mut functions: Vec<(FunctionSig, FuncType)> = vec![];
    let ctx = &mut ParserCtx::default();

    for function in to_parse {
        let mut tokens = Token::parse(function).unwrap();

        let sig = FunctionSig::parse(ctx, &mut tokens).unwrap();

        functions.push((sig, assert_eq));
    }

    functions
}

fn assert_eq(compiler: &mut Compiler, vars: &[Var], span: (usize, usize)) {
    let lhs = vars[0];
    let rhs = vars[1];

    // TODO: use permutation to check that
    compiler.gates(
        GateKind::DoubleGeneric,
        vec![Some(lhs), Some(rhs)],
        vec![Field::one(), Field::one().neg()],
        span,
    );
}
