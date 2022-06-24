use crate::parser::{FunctionSig, Ty};

pub fn parse_crypto_import(
    path: &mut impl Iterator<Item = String>,
) -> Result<(Vec<FunctionSig>, Vec<String>), ()> {
    let module = path.next().ok_or(())?;

    match module.as_ref() {
        "poseidon" => {
            let array_of_3_fel = Ty::Array(Box::new(Ty::Struct("Field".to_string())), 3);

            let poseidon = FunctionSig {
                name: "poseidon".to_string(),
                arguments: vec![(true, "input".to_string(), array_of_3_fel.clone())],
                return_type: Some(array_of_3_fel),
            };
            let functions = vec![poseidon];
            let types = vec![];
            Ok((functions, types))
        }
        _ => Err(()),
    }
}
