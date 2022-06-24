use crate::parser::FunctionSig;

pub mod crypto;

pub fn parse_std_import(
    path: &mut impl Iterator<Item = String>,
) -> Result<(Vec<FunctionSig>, Vec<String>), ()> {
    let mut functions = vec![];
    let mut types = vec![];

    let module = path.next().ok_or(())?;

    match module.as_ref() {
        "crypto" => {
            let thing = crypto::parse_crypto_import(path)?;
            // TODO: make sure we're not importing colliding names
            functions.extend(thing.0);
            types.extend(thing.1);
        }
        _ => return Err(()),
    }

    Ok((functions, types))
}
