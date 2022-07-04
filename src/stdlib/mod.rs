use crate::{
    ast::{Compiler, FuncType, GateKind, Value, Var, F},
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

    // TODO: handle one being a constant or not
    // the problem here: I want to avoid being able to create a gate if the value needs to be constrained further
    // I need Var to tell me what it is (not just its index)
    /*
    let lhs = match compiler.witness_vars[&lhs] {
        Value::Hint(_) => todo!(),
        // constant: need to constraint it? Or already constrained?
        Value::Constant(_) => todo!(),
        // linear combination of other variables: already constrained?
        Value::LinearCombination(_) => lhs,
        // external: no need to constrain
        Value::External(_) => lhs,
    };

    let rhs = match compiler.witness_vars[&rhs] {
        Value::Hint(_) => todo!(),
        Value::Constant(_) => todo!(),
        Value::LinearCombination(_) => lhs,
        // external: no need to constrain
        Value::External(_) => rhs,
    };
    */

    compiler.gates(
        GateKind::DoubleGeneric,
        vec![Some(lhs), Some(rhs)],
        vec![F::one(), F::one().neg()],
        span,
    );
}
