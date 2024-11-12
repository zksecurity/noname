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
    let folders = ["assets", "src/stdlib"];

    let mut options = CopyOptions::new();
    options.overwrite = true; // Overwrite existing files
    options.copy_inside = true; // Copy contents inside the source directory

    for folder in &folders {
        let src = current_dir.join(folder);
        dir::copy(&src, &release_dir.join(folder), &options)
            .expect("could not copy assets and stdlib folders to release directory");
    }
}
