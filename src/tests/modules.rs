use crate::{
    circuit_writer::CircuitWriter,
    cli::packages::{path_to_package, UserRepo},
    compiler::{compile_single, get_tast},
    type_checker::Dependencies,
};

const LIB: &str = r#"struct Thing {
  tt: Field
}

fn Thing.tt(self) -> Field {
  return self.tt;
}

fn new() -> Thing {
  return Thing { tt: 4 };
}

fn add(xx: Field, yy: Field) -> Field {
  return xx + yy;
}"#;

const MAIN: &str = r#"use mimoo::field;

fn main(pub xx: Field, yy: Field) {
  let y2 = field::new();
  let y3 = y2.tt();
  let zz = field::add(y3, 1);
  assert_eq(zz, xx);
}"#;

#[test]
fn test_simple_module() -> miette::Result<()> {
    // parse the lib
    let mut deps_tasts = Dependencies::default();
    let tast = get_tast("lib.no".to_string(), LIB.to_string(), &deps_tasts)?;
    deps_tasts.add_type_checker(UserRepo::new("mimoo/field"), "lib.no".to_string(), tast);

    // parse the main
    let tast = get_tast("main.no".to_string(), MAIN.to_string(), &deps_tasts)?;

    // compile
    CircuitWriter::generate_circuit(tast, deps_tasts)?;

    Ok(())
}
