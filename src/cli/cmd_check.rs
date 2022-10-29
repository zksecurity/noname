use std::{path::PathBuf, process};

use miette::{Context, IntoDiagnostic, NamedSource, Result};

use crate::{cli::packages::get_dep_code, compiler::get_tast};

use super::{
    manifest::Manifest,
    packages::{
        get_deps_of_package, is_lib, path_to_package, validate_package_and_get_manifest,
        DependencyGraph, UserRepo,
    },
    NONAME_DIRECTORY, PACKAGE_DIRECTORY,
};

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

    // find manifest
    let manifest = validate_package_and_get_manifest(&curr_dir, false)?;

    // get all dependencies
    get_deps_of_package(&manifest);

    // produce dependency graph
    let is_lib = is_lib(&curr_dir);

    let this = if is_lib {
        Some(UserRepo::new(&manifest.package.name))
    } else {
        None
    };

    let dep_graph = DependencyGraph::new_from_manifest(this, &manifest)?;

    // produce artifacts for each dependency, starting from leaf dependencies
    for dep in dep_graph.from_leaves_to_roots() {
        let path = path_to_package(&dep);

        let lib_file = path.join("src").join("lib.no");
        let code = std::fs::read_to_string(&lib_file)
            .into_diagnostic()
            .wrap_err_with(|| format!("could not read file `{}`", path.display()))?;

        let _tast = get_tast(&code)
            .map_err(|e| e.with_source_code(NamedSource::new(lib_file.to_str().unwrap(), code)))?;
    }

    // produce artifact for this one
    let src_dir = curr_dir.join("src");

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

    get_tast(&code)
        .map_err(|e| e.with_source_code(NamedSource::new(file_path.to_str().unwrap(), code)))?;

    println!("all good!");
    Ok(())
}
