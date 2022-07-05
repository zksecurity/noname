use crate::parser::{Attribute, FunctionSig, Ident, Ty, TyKind};

pub fn parse_crypto_import(
    path: &mut impl Iterator<Item = String>,
) -> Result<(Vec<FunctionSig>, Vec<String>), &'static str> {
    let module = path.next().ok_or("no module to read")?;

    match module.as_ref() {
        "poseidon" => {
            let array_of_3_fel = Ty {
                kind: TyKind::Array(Box::new(TyKind::Field), 3),
                span: (0, 0),
            };

            let poseidon = FunctionSig {
                name: Ident {
                    value: "poseidon".to_string(),
                    span: (0, 0),
                },
                arguments: vec![(
                    Attribute::Pub,
                    Ident {
                        value: "input".to_string(),
                        span: (0, 0),
                    },
                    array_of_3_fel.clone(),
                )],
                return_type: Some(array_of_3_fel),
            };
            let functions = vec![poseidon];
            let types = vec![];
            Ok((functions, types))
        }
        _ => Err("unknown module"),
    }
}
