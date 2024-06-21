use crate::{
    backends::kimchi::KimchiVesta,
    cli::packages::UserRepo,
    compiler::{compile, typecheck_next_file, Sources},
    type_checker::TypeChecker,
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

const LIB: &str = r"
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
fn new_liblib() -> liblib::Lol {
    return liblib::Lol.new();
}

fn test_liblib(ff: Field, lol: liblib::Lol) {
    lol.match(ff);
}
";

const MAIN: &str = r"
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
";

#[test]
fn test_simple_module() -> miette::Result<()> {
    let mut sources = Sources::new();

    // parse the transitive dependency
    let mut tast = TypeChecker::<KimchiVesta>::new();
    let mut node_id = 0;
    node_id = typecheck_next_file(
        &mut tast,
        Some(UserRepo::new("mimoo/liblib")),
        &mut sources,
        "liblib.no".to_string(),
        LIBLIB.to_string(),
        node_id,
    )?;

    // parse the lib
    node_id = typecheck_next_file(
        &mut tast,
        Some(UserRepo::new("mimoo/lib")),
        &mut sources,
        "lib.no".to_string(),
        LIB.to_string(),
        node_id,
    )?;

    // parse the main
    typecheck_next_file(
        &mut tast,
        None,
        &mut sources,
        "main.no".to_string(),
        MAIN.to_string(),
        node_id,
    )?;

    // backend
    let kimchi_vesta = KimchiVesta::new(false);

    // compile
    compile(&sources, tast, kimchi_vesta)?;

    Ok(())
}
