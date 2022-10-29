use std::{path::PathBuf, process};

use miette::{Context, IntoDiagnostic, NamedSource, Result};

use crate::{
    circuit_writer::CircuitWriter,
    cli::packages::{get_dep_code, path_to_package},
    compiler::get_tast,
    type_checker::{Dependencies, TypeChecker},
};

use super::{
    manifest::Manifest,
    packages::{
        get_deps_of_package, is_lib, validate_package_and_get_manifest, DependencyGraph, UserRepo,
    },
    NONAME_DIRECTORY, PACKAGE_DIRECTORY,
};

#[derive(clap::Parser)]
pub struct CmdBuild {
    /// path to the directory to create
    #[clap(short, long, value_parser)]
    path: Option<PathBuf>,
}

pub fn cmd_build(args: CmdBuild) -> miette::Result<()> {
    let curr_dir = args
        .path
        .unwrap_or_else(|| std::env::current_dir().unwrap());

    // produce all TASTs
    let (code, tast, deps_tasts) = produce_all_asts(&curr_dir)?;

    // produce indexes
    let compiled_circuit = CircuitWriter::generate_circuit(tast, deps_tasts, &code)
        .into_diagnostic()
        .map_err(|e| e.with_source_code(NamedSource::new(curr_dir.to_str().unwrap(), code)))?;

    compiled_circuit.asm(false);

    // store/cache artifacts

    //
    Ok(())
}

#[derive(clap::Parser)]
pub struct CmdCheck {
    /// path to the directory to create
    #[clap(short, long, value_parser)]
    path: Option<PathBuf>,
}

pub fn cmd_check(args: CmdCheck) -> Result<()> {
    let curr_dir = args
        .path
        .unwrap_or_else(|| std::env::current_dir().unwrap());

    // produce all TASTs and stop here
    produce_all_asts(&curr_dir)?;

    println!("all good!");
    Ok(())
}

fn produce_all_asts(path: &PathBuf) -> Result<(String, TypeChecker, Dependencies)> {
    // find manifest
    let manifest = validate_package_and_get_manifest(&path, false)?;

    // get all dependencies
    get_deps_of_package(&manifest);

    // produce dependency graph
    let is_lib = is_lib(&path);

    let this = if is_lib {
        Some(UserRepo::new(&manifest.package.name))
    } else {
        None
    };

    let dep_graph = DependencyGraph::new_from_manifest(this, &manifest)?;

    // produce artifacts for each dependency, starting from leaf dependencies
    let mut deps_tasts = Dependencies::default();
    for dep in dep_graph.from_leaves_to_roots() {
        let path = path_to_package(&dep);

        let lib_file = path.join("src").join("lib.no");
        let code = std::fs::read_to_string(&lib_file)
            .into_diagnostic()
            .wrap_err_with(|| format!("could not read file `{}`", path.display()))?;

        let lib_file_str = lib_file.to_str().unwrap();
        let tast = get_tast(&code, &deps_tasts)
            .map_err(|e| e.with_source_code(NamedSource::new(lib_file_str, code.clone())))?;
        deps_tasts
            .deps
            .insert(dep, (tast, lib_file_str.to_string(), code));
    }

    // produce artifact for this one
    let src_dir = path.join("src");

    let lib_file = src_dir.join("lib.no");
    let main_file = src_dir.join("main.no");

    // assuming that we can't have both lib.no and main.no
    let file_path = if lib_file.exists() {
        lib_file
    } else {
        main_file
    };

    let code = std::fs::read_to_string(&file_path)
        .into_diagnostic()
        .wrap_err_with(|| format!("could not read file `{}`", file_path.display()))?;

    let tast = get_tast(&code, &deps_tasts).map_err(|e| {
        e.with_source_code(NamedSource::new(file_path.to_str().unwrap(), code.clone()))
    })?;

    Ok((code, tast, deps_tasts))
}
