use miette::{IntoDiagnostic, Result, WrapErr};
use std::path::PathBuf;

#[derive(clap::Parser)]
pub struct CmdNew {
    /// path to the directory to create
    #[clap(short, long, value_parser)]
    path: PathBuf,

    /// is the package a library or a binary?
    #[clap(short, long)]
    lib: bool,
}

#[derive(clap::Parser)]
pub struct CmdInit {
    /// path to the directory to create
    #[clap(short, long, value_parser)]
    path: PathBuf,

    /// is the package a library or a binary?
    #[clap(short, long)]
    lib: bool,
}

pub fn cmd_new(args: CmdNew) -> Result<()> {
    let path = args.path;

    // for now, the package name is the same as the path
    let package_name = path
        .as_os_str()
        .to_str()
        .expect("couldn't parse the path as a string")
        .to_owned();

    if path.exists() {
        miette::bail!("path `{path}` already exists. Use `noname init` to create a new package in an existing directory");
    }

    std::fs::create_dir(&path)
        .into_diagnostic()
        .wrap_err("couldn't create directory at given path")?;

    mk(path, &package_name, args.lib)
}

pub fn cmd_init(args: CmdInit) -> Result<()> {
    let path = args.path;

    // for now, the package name is the same as the path
    let package_name = path
        .as_os_str()
        .to_str()
        .expect("couldn't parse the path as a string")
        .to_owned();

    if !path.exists() {
        miette::bail!("path `{path}` doesn't exists. Use `noname new` to create a new package in an non-existing directory");
    }

    mk(path, &package_name, args.lib)
}

fn mk(path: PathBuf, package_name: &str, is_lib: bool) -> Result<()> {
    let content = format!(
        r#"[package]
name = "{package_name}"
version = "0.1.0"
# see documentation at TODO for more information on how to edit this file

[dependencies]
"#
    );

    std::fs::write(path.join("Noname.toml"), content)
        .into_diagnostic()
        .wrap_err("cannot create Noname.toml file at given path")?;

    let src_path = path.join("src");

    std::fs::create_dir(&src_path)
        .into_diagnostic()
        .wrap_err("cannot create src directory at given path")?;

    if is_lib {
        let content = r#"fn add(xx: Field, yy: Field) -> Field {
    return xx + yy;
}
"#;

        std::fs::write(src_path.join("lib.no"), content)
            .into_diagnostic()
            .wrap_err("cannot create src/lib.no file at given path")?;
    } else {
        let content = r#"fn main(pub xx: Field, yy: Field) {
    let zz = yy + 1;
    assert_eq(zz, xx);
}
"#;
        std::fs::write(src_path.join("main.no"), content)
            .into_diagnostic()
            .wrap_err("cannot create src/main.no file at given path")?;
    }

    Ok(())
}
