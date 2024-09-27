use std::path::Path;

use crate::{
    backends::Backend,
    cli::packages::UserRepo,
    compiler::{typecheck_next_file, Sources},
    type_checker::TypeChecker,
};

mod examples;
mod modules;
mod stdlib;

fn init_stdlib_dep<B: Backend>(sources: &mut Sources, tast: &mut TypeChecker<B>) {
    let libs = vec!["int", "comparator", "bigint"];

    // read stdlib files from src/stdlib/native/
    for lib in libs {
        let module = UserRepo::new(&format!("std/{}", lib));
        let prefix_stdlib = Path::new("src/stdlib/native/");
        let code = std::fs::read_to_string(prefix_stdlib.join(format!("{lib}.no"))).unwrap();
        let _node_id =
            typecheck_next_file(tast, Some(module), sources, lib.to_string(), code, 0).unwrap();
    }
}
