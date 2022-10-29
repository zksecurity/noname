use std::{path::PathBuf, process};

use miette::{Context, IntoDiagnostic, Result};

use super::{
    manifest::Manifest,
    packages::{get_deps_of_package, validate_package_and_get_manifest},
    NONAME_DIRECTORY, PACKAGE_DIRECTORY,
};

#[derive(clap::Parser)]
pub struct CmdBuild {
    /// path to the directory to create
    #[clap(short, long, value_parser)]
    path: PathBuf,
}

pub fn cmd_build(args: CmdBuild) -> Result<()> {
    // find manifest
    let manifest = validate_package_and_get_manifest(&args.path, false)?;

    // get all dependencies
    get_deps_of_package(&manifest);

    // produce dependency graph

    // find local `lib.no` or `main.no` file

    //
    Ok(())
}
