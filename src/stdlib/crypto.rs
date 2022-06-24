use crate::parser::{FunctionSig, Ty, TyKind};

pub fn parse_crypto_import(
    path: &mut impl Iterator<Item = String>,
) -> Result<(Vec<FunctionSig>, Vec<String>), &'static str> {
    let module = path.next().ok_or("no module to read")?;

    match module.as_ref() {
        "poseidon" => {
            let array_of_3_fel = Ty {
                typ: TyKind::Array(
                    Box::new(Ty {
                        typ: TyKind::Custom("Field".to_string()),
                        span: (0, 0),
                    }),
                    3,
                ),
                span: (0, 0),
            };

            let poseidon = FunctionSig {
                name: "poseidon".to_string(),
                arguments: vec![(true, "input".to_string(), array_of_3_fel.clone())],
                return_type: Some(array_of_3_fel),
            };
            let functions = vec![poseidon];
            let types = vec![];
            Ok((functions, types))
        }
        _ => Err("unknown module"),
    }
}
