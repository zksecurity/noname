use crate::{
    cli::packages::UserRepo,
    compiler::{compile, get_tast, Sources},
    type_checker::Dependencies,
};

//
// MAIN -> LIB -> LIBLIB
//

const LIBLIB: &str = "
// test a transitive dependency type
struct Lol {
    aa: Field,
}

fn Lol.match(self, bb: Field) {
    assert_eq(self.aa, bb);
}

fn Lol.new() -> Lol {
    return Lol {
        aa: 1,
    };
}
";

const LIB: &str = r#"
use mimoo::liblib;

// test a library's type that links to its own type
struct Inner {
    inner: Field,
}

struct Lib {
  tt: Inner
}

// a normal function
fn add(xx: Field, yy: Field) -> Field {
    return xx + yy;
}

fn Lib.tt(self) -> Field {
    return self.tt.inner;
}

fn new() -> Lib {
    let inner = Inner { inner: 5 };
    return Lib { tt: inner };
}

// a transitive dependency
fn new_liblib() -> Liblib::Lol {}
    Liblib::Lol.new();
}

fn test_liblib(ff: Field, lol: Liblib::Lol) {
    lol.match(ff);
}
"#;

const MAIN: &str = r#"
use mimoo::lib;

fn main(pub xx: Field, yy: Field) {
    // use a library's function
    assert_eq(lib::add(xx, yy), 2);

    // use a library's type
    let y2 = lib::new();
    let y3 = y2.tt();
    let zz = lib::add(y3, 1);
    assert_eq(zz, xx);

    // use a transitive dependency
    let lol = lib::new_liblib();
    lib::test_liblib(1, lol);
}
"#;

#[test]
fn test_simple_module() -> miette::Result<()> {
    let mut sources = Sources::new();
    let mut deps_tasts = Dependencies::default();

    // parse the transitive dependency
    let tast = get_tast(
        &mut sources,
        "liblib.no".to_string(),
        LIBLIB.to_string(),
        &deps_tasts,
    )?;
    deps_tasts.add_type_checker(UserRepo::new("mimoo/liblib"), "liblib.no".to_string(), tast);

    // parse the lib
    let tast = get_tast(
        &mut sources,
        "lib.no".to_string(),
        LIB.to_string(),
        &deps_tasts,
    )?;
    deps_tasts.add_type_checker(UserRepo::new("mimoo/lib"), "lib.no".to_string(), tast);

    // parse the main
    let tast = get_tast(
        &mut sources,
        "main.no".to_string(),
        MAIN.to_string(),
        &deps_tasts,
    )?;

    // compile
    compile(&sources, tast, deps_tasts)?;

    Ok(())
}
