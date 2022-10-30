pub mod cmd_build_and_check;
pub mod cmd_new_and_init;
pub mod manifest;
pub mod packages;
pub mod cmd_prove_and_verify;

pub use cmd_build_and_check::{cmd_build, cmd_check, CmdBuild, CmdCheck};
pub use cmd_new_and_init::{cmd_init, cmd_new, CmdInit, CmdNew};
pub use cmd_prove_and_verify::{cmd_prove, cmd_verify, CmdProve, CmdVerify};

/// The directory under the user home directory containing all noname-related files.
pub const NONAME_DIRECTORY: &str = ".noname";

/// The directory under [NONAME_DIRECTORY] containing all package-related files.
pub const PACKAGE_DIRECTORY: &str = "packages";
