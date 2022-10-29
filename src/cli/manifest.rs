use std::path::PathBuf;

use miette::{Context, IntoDiagnostic, Result};

#[derive(Clone, serde::Deserialize)]
pub struct Manifest {
    pub package: Package,
    // no versioning at the moment
    pub dependencies: Vec<String>,
}

#[derive(Clone, serde::Deserialize)]
pub struct Package {
    pub name: String,
    // does not matter atm
    pub version: String,
    pub description: Option<String>,
}

/// This retrieves a dependency listed in the manifest file.
/// It downloads it from github, and stores it under the `deps` directory.
/// A dependency is expected go be given as "user/repo".
pub fn read_manifest(path: &PathBuf) -> Result<Manifest> {
    if !path.exists() {
        miette::bail!("path `{path}` doesn't exists. Use `noname new` to create a new package in an non-existing directory");
    }

    // read the package manifest file
    let manifest_file = path.join("Noname.toml");
    let content = std::fs::read_to_string(&manifest_file)
        .into_diagnostic()
        .wrap_err_with(|| format!("could not read file `{}`", path.display()))?;

    let manifest: Manifest = toml::from_str(&content)
        .into_diagnostic()
        .wrap_err_with(|| format!("could not parse file `{}`", path.display()))?;

    Ok(manifest)
}
