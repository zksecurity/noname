pub mod cmd_build;
pub mod cmd_check;
pub mod cmd_new_and_init;
pub mod cmd_test;
pub mod manifest;
pub mod packages;

pub use cmd_build::{cmd_build, CmdBuild};
pub use cmd_check::{cmd_check, CmdCheck};
pub use cmd_new_and_init::{cmd_init, cmd_new, CmdInit, CmdNew};
pub use cmd_test::{cmd_test, CmdTest};

/// The directory under the user home directory containing all noname-related files.
pub const NONAME_DIRECTORY: &str = ".noname";

/// The directory under [NONAME_DIRECTORY] containing all package-related files.
pub const PACKAGE_DIRECTORY: &str = "packages";
