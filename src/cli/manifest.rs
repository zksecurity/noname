use camino::Utf8PathBuf as PathBuf;
use miette::{Context, IntoDiagnostic, Result};
use regex::Regex;

#[derive(Clone, serde::Deserialize)]
pub struct Manifest {
    pub package: Package,
}

#[derive(Clone, serde::Deserialize)]
pub struct Package {
    pub name: String,
    // does not matter atm
    pub version: String,
    pub description: Option<String>,
    // no versioning at the moment
    pub dependencies: Option<Vec<String>>,
}

impl Manifest {
    pub(crate) fn dependencies(&self) -> Vec<String> {
        self.package.dependencies.clone().unwrap_or_default()
    }
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
        .wrap_err_with(|| format!("could not find manifest file `{manifest_file}`"))?;

    let manifest: Manifest = toml::from_str(&content)
        .into_diagnostic()
        .wrap_err_with(|| format!("could not parse file `{manifest_file}`"))?;

    // ensure the package name is correctly formatted
    let re = Regex::new(r"^[a-z0-9_-]+/[a-z0-9_-]+$").unwrap();
    if !re.is_match(&manifest.package.name) {
        miette::bail!(format!(
            "invalid package name `{}`. Package names must be in the format `user/repo`",
            manifest.package.name
        ));
    }

    // the package cannot have repo "std"
    if manifest.package.name.starts_with("std") {
        miette::bail!("package name `std` is reserved");
    }

    for dep in manifest.dependencies() {
        // none of the deps can have repo "std"
        if dep.starts_with("std") {
            miette::bail!("package `std/..` cannot be a dependency");
        }

        if !re.is_match(&dep) {
            miette::bail!(format!(
                "invalid package name `{}`. Package names must be in the format `user/repo`",
                dep
            ));
        }
    }

    Ok(manifest)
}
