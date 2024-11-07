use std::{env, path::PathBuf};

use fs_extra::dir::{self, CopyOptions};

fn main() {
    let home_dir: PathBuf =
        dirs::home_dir().expect("could not find home directory of current user");

    let noname_dir = home_dir.join(".noname");
    let release_dir = noname_dir.join("release");

    // If it exists, then remove it
    if release_dir.exists() {
        fs_extra::remove_items(&[&release_dir]).expect("could not remove release directory");
    }

    let current_dir = env::current_dir().expect("could not get current directory");

    // Set up copy options
    let mut options = CopyOptions::new();
    options.overwrite = true; // Overwrite existing files
    options.copy_inside = true; // Copy contents inside the source directory

    // Copy the current folder to the release directory
    dir::copy(current_dir, &release_dir, &options)
        .expect("could not copy current directory to release directory");
}
