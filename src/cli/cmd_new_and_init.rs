use camino::Utf8PathBuf as PathBuf;
use miette::{IntoDiagnostic, Result, WrapErr};

const MAIN_CONTENT: &str = r"fn main(pub xx: Field, yy: Field) {
    let zz = yy + 1;
    assert_eq(zz, xx);
}
";

const LIB_CONTENT: &str = r"fn add(xx: Field, yy: Field) -> Field {
    return xx + yy;
}
";

#[derive(clap::Parser)]
pub struct CmdNew {
    /// path to the directory to create
    #[clap(short, long, value_parser)]
    path: PathBuf,

    /// is the package a library or a binary?
    #[clap(long)]
    lib: bool,
}

#[derive(clap::Parser)]
pub struct CmdInit {
    /// path to the directory to create
    #[clap(short, long, value_parser)]
    path: Option<PathBuf>,

    /// is the package a library or a binary?
    #[clap(long)]
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
    let path = args
        .path
        .unwrap_or(std::env::current_dir().unwrap().try_into().unwrap());

    // for now, the package name is the same as the path
    let package_name = path
        .file_name()
        .ok_or(miette::miette!("invalid path given in argument to CLI"))?
        .to_string();

    if !path.exists() {
        miette::bail!("path `{path}` doesn't exists. Use `noname new` to create a new package in an non-existing directory");
    }

    if !path.is_dir() {
        miette::bail!("path `{path}` is not a directory");
    }

    mk(path, &package_name, args.lib)
}

fn mk(path: PathBuf, package_name: &str, is_lib: bool) -> Result<()> {
    let user = get_git_user();

    let content = format!(
        r#"[package]
name = "{user}/{package_name}"
version = "0.1.0"
# see documentation at TODO for more information on how to edit this file

dependencies = []
"#
    );

    let manifest_file = path.join("Noname.toml");

    if manifest_file.exists() {
        miette::bail!("manifest file already exists at `{manifest_file}`");
    }

    std::fs::write(&manifest_file, content)
        .into_diagnostic()
        .wrap_err(format!(
            "cannot create Noname.toml file at given path: `{manifest_file}`"
        ))?;

    let src_path = path.join("src");

    if src_path.exists() {
        miette::bail!("src directory already exists at `{src_path}`");
    }

    std::fs::create_dir(&src_path)
        .into_diagnostic()
        .wrap_err(format!(
            "cannot create src directory at given path: `{src_path}`"
        ))?;

    let (content, file_name) = if is_lib {
        (LIB_CONTENT, "lib.no")
    } else {
        (MAIN_CONTENT, "main.no")
    };

    let file_path = src_path.join(file_name);

    if file_path.exists() {
        miette::bail!("file already exists at `{file_path}`");
    }

    std::fs::write(&file_path, content)
        .into_diagnostic()
        .wrap_err(format!("cannot create file at given path: `{file_path}`"))?;

    // success msg
    println!("created new package at `{path}`");

    Ok(())
}

fn get_git_user() -> String {
    let output = std::process::Command::new("git")
        .arg("config")
        .arg("user.name")
        .output()
        .expect("failed to execute git command");

    assert!(output.status.success(), "failed to get git user name");

    let output = String::from_utf8(output.stdout).expect("couldn't parse git output as utf8");

    let username = output.trim().to_owned();

    username.replace(' ', "_").to_lowercase()
}
