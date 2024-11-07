use std::{
    env, fs,
    path::{Path, PathBuf},
};

// Copy a folder recursively
fn copy_recursively(src: &Path, dst: &Path) -> std::io::Result<()> {
    if src.is_dir() {
        fs::create_dir_all(dst)?;
        for entry in fs::read_dir(src)? {
            let entry = entry?;
            let entry_path = entry.path();
            let dest_path = dst.join(entry.file_name());
            copy_recursively(&entry_path, &dest_path)?;
        }
    } else {
        fs::copy(src, dst)?;
    }
    Ok(())
}

fn main() {
    let home_dir: PathBuf =
        dirs::home_dir().expect("could not find home directory of current user");

    let noname_dir = home_dir.join(".noname");
    let release_dir = noname_dir.join("release");

    // if it exists, then revmove it
    if release_dir.exists() {
        fs::remove_dir_all(&release_dir).expect("could not remove release directory");
    }

    // copy the current folder to the release directory
    fs::create_dir_all(&release_dir).expect("could not create release directory");

    let current_dir = env::current_dir().expect("could not get current directory");
    copy_recursively(&current_dir, &release_dir)
        .expect("could not copy current directory to release directory");
}
