use crate::{
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

pub fn utils_functions() -> Vec<(String, FunctionSig)> {
    let to_parse = [ASSERT_FN, ASSERT_EQ_FN];
    let mut functions = vec![];
    let ctx = &mut ParserCtx::default();

    for function in to_parse {
        let mut tokens = Token::parse(function).unwrap();

        let function = FunctionSig::parse(ctx, &mut tokens).unwrap();

        functions.push((function.name.clone(), function));
    }

    functions
}
