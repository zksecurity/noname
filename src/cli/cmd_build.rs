use std::{path::PathBuf, process};

use miette::{Context, IntoDiagnostic, Result};

use super::{
    manifest::Manifest,
    packages::{
        get_deps_of_package, is_lib, validate_package_and_get_manifest, DependencyGraph, Library,
    },
    NONAME_DIRECTORY, PACKAGE_DIRECTORY,
};

#[derive(clap::Parser)]
pub struct CmdBuild {
    /// path to the directory to create
    #[clap(short, long, value_parser)]
    path: Option<PathBuf>,
}

pub fn cmd_build(args: CmdBuild) -> Result<()> {
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
        Some(Library::new(&manifest.package.name))
    } else {
        None
    };

    let dep_graph = DependencyGraph::new_from_manifest(this, &manifest)?;

    // find local `lib.no` or `main.no` file

    //
    Ok(())
}
