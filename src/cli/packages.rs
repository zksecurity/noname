use std::{
    collections::{HashMap, HashSet},
    process,
};

use camino::Utf8PathBuf as PathBuf;
use miette::{Context, IntoDiagnostic, Result};
use serde::{Deserialize, Serialize};

use super::{
    manifest::{read_manifest, Manifest},
    NONAME_DIRECTORY, PACKAGE_DIRECTORY,
};

/// A dependency is a Github `user/repo` pair.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UserRepo {
    pub user: String,
    pub repo: String,
}

impl UserRepo {
    pub(crate) fn new(arg: &str) -> Self {
        let mut args = arg.split('/');
        let user = args.next().unwrap().to_string();
        let repo = args.next().unwrap().to_string();
        assert!(args.next().is_none());
        Self { user, repo }
    }
}

impl std::fmt::Display for UserRepo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.user, self.repo)
    }
}

#[derive(Debug)]
pub struct DependencyGraph {
    /// Name of this package.
    /// Useful to make sure the package doesn't depend on itself.
    this: Option<UserRepo>,
    root: Vec<DependencyNode>,
    cached_manifests: HashMap<UserRepo, Vec<UserRepo>>,
}

impl DependencyGraph {
    pub(crate) fn new(this: Option<UserRepo>) -> Self {
        Self {
            this,
            root: vec![],
            cached_manifests: HashMap::new(),
        }
    }

    pub(crate) fn new_from_manifest(this: Option<UserRepo>, manifest: &Manifest) -> Result<Self> {
        let mut dep_graph = Self::new(this);
        let deps = get_deps_of_package(manifest);
        dep_graph.add_deps(deps)?;
        Ok(dep_graph)
    }

    fn add_deps(&mut self, deps: Vec<UserRepo>) -> Result<()> {
        for dep in deps {
            self.add_dep(dep)?;
        }

        Ok(())
    }

    fn add_dep(&mut self, dep: UserRepo) -> Result<()> {
        let mut parents = HashSet::new();
        if let Some(this) = &self.this {
            if this == &dep {
                miette::bail!("this library (`{}`) cannot depend on itself", dep);
            }

            parents.insert(this.clone());
        }

        let node = self.init_package(&dep, parents.clone())?;
        self.root.push(node);

        Ok(())
    }

    pub fn init_package(
        &mut self,
        package: &UserRepo,
        mut parents: HashSet<UserRepo>,
    ) -> Result<DependencyNode> {
        // add package to parent
        parents.insert(package.clone());

        let deps = if let Some(deps) = self.cached_manifests.get(package) {
            deps.clone()
        } else {
            // download dependency (if not already downloaded)
            let path = path_to_package(package);
            if !path.exists() {
                download_from_github(package)?;
            }

            // get manifest
            // TODO: if it's garbage, we actually don't delete the repo we just cloned (and this in other places as well)
            let manifest = validate_package_and_get_manifest(&path, true)
                .wrap_err(format!("the dependency {package} is invalid."))?;

            // make sure it matches the package `user/repo` format
            if manifest.package.name != package.to_string() {
                miette::bail!(
                    "package `{}` has a different name in its manifest: `{}`",
                    package,
                    manifest.package.name
                );
            }

            // extract dependencies
            get_deps_of_package(&manifest)
        };

        // recursively do the same
        let mut deps_nodes = vec![];
        for dep in deps {
            if parents.contains(&dep) {
                miette::bail!(format!(
                    "circular dependency detected: {dep} is already a parent of {package}"
                ));
            }

            let dep_node = self.init_package(&dep, parents.clone())?; // recursion :(
            deps_nodes.push(dep_node);
        }

        // return the created node
        let node = DependencyNode::new(package.clone(), deps_nodes);
        Ok(node)
    }

    // TODO: Fix usage of self in `from_*` fn.
    // Either this should be renamed, or `self` shouldn't be used.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn from_leaves_to_roots(&self) -> Vec<UserRepo> {
        let mut res = vec![];

        for root in &self.root {
            let mut stack = vec![root];
            while let Some(node) = stack.pop() {
                // we already added this package (and its dependencies)
                if res.contains(&node.dep) {
                    continue;
                }

                // we can add this package only if all its deps are already in
                let can_add = node.deps.iter().all(|dep| res.contains(&dep.dep));

                if can_add {
                    res.push(node.dep.clone());
                } else {
                    stack.push(node); //re-add

                    for dep in &node.deps {
                        stack.push(dep);
                    }
                }
            }
        }

        res
    }
}

#[derive(Clone, Debug)]
pub struct DependencyNode {
    dep: UserRepo,
    deps: Vec<DependencyNode>,
}

impl DependencyNode {
    fn new(dep: UserRepo, deps: Vec<DependencyNode>) -> Self {
        Self { dep, deps }
    }
}

/// This retrieves a dependency listed in the manifest file.
/// It downloads it from github, and stores it under the `deps` directory.
/// A dependency is expected go be given as "user/repo".
/// Note that this does not download the dependencies of the dependency.
pub fn get_dep(dep: &UserRepo) -> Result<Manifest> {
    // download the dependency if we don't already have it
    let path = path_to_package(dep);

    if !path.exists() {
        download_from_github(dep)?;
    }

    // validate and get manifest file
    let must_be_lib = true;
    let manifest = validate_package_and_get_manifest(&path, must_be_lib)?;

    //
    Ok(manifest)
}

/// Returns the dependencies of a package (given it's manifest).
#[must_use]
pub fn get_deps_of_package(manifest: &Manifest) -> Vec<UserRepo> {
    manifest
        .dependencies()
        .iter()
        .map(|dep| {
            let mut split = dep.split('/');
            let user = split.next().unwrap().to_owned();
            let repo = split.next().unwrap().to_owned();
            assert!(
                split.next().is_none(),
                "invalid dependency name (expected: user/repo (TODO: better error)"
            );
            UserRepo { user, repo }
        })
        .collect()
}

// read the package's `lib.no` file
pub fn get_dep_code(dep: &UserRepo) -> Result<String> {
    let path = path_to_package(dep);

    let lib_file = path.join("src").join("lib.no");
    let lib_content = std::fs::read_to_string(lib_file)
        .into_diagnostic()
        .wrap_err_with(|| format!("could not read file `{path}`"))?;

    Ok(lib_content)
}

/// Obtain local path to a package.
pub(crate) fn path_to_package(dep: &UserRepo) -> PathBuf {
    let home_dir: PathBuf = dirs::home_dir()
        .expect("could not find home directory of current user")
        .try_into()
        .expect("invalid UTF8 path");
    let noname_dir = home_dir.join(NONAME_DIRECTORY);
    let package_dir = noname_dir.join(PACKAGE_DIRECTORY);

    package_dir.join(&dep.user).join(&dep.repo)
}

/// download package from github
pub fn download_from_github(dep: &UserRepo) -> Result<()> {
    let url = format!(
        "https://github.com/{user}/{repo}.git",
        user = &dep.user,
        repo = &dep.repo
    );
    let path = path_to_package(dep);

    let output = process::Command::new("git")
        .arg("clone")
        .arg(url)
        .arg(path)
        .output()
        .expect("failed to execute git command");

    if !output.status.success() {
        miette::bail!(format!("could not download package `{dep}`. Are you sure that https://www.github.com/{dep} is a valid package?"));
    }

    Ok(())
}

#[must_use]
pub fn is_lib(path: &PathBuf) -> bool {
    path.join("src").join("lib.no").exists()
}

/// A valid package must have a valid `Noname.toml` as well as a `lib.no` file.
pub fn validate_package_and_get_manifest(path: &PathBuf, must_be_lib: bool) -> Result<Manifest> {
    // check if folder exists
    if !path.exists() {
        miette::bail!(format!("path `{path}` doesn't exists. Use `noname new` to create a new package in an non-existing directory"));
    }

    // parse `NoName.toml`
    let manifest: Manifest = read_manifest(path)?;

    // check if `lib.no` exists
    let src_path = path.join("src");

    let main_path = src_path.join("main.no");
    let lib_path = src_path.join("lib.no");

    match (lib_path.exists(), main_path.exists()) {
        (true, true) => miette::bail!(
            "package has both a `lib.no` and a `main.no` file. Only one of them is allowed"
        ),
        (false, false) => miette::bail!(
            "package has neither a `lib.no` nor a `main.no` file. At least one of them is required"
        ),
        (false, true) if must_be_lib => miette::bail!("package is missing a `lib.no` file"),
        _ => (),
    }

    //
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    const USER: &str = "mimoo";
    const THIS: &str = "this";

    fn dep(ii: &str) -> UserRepo {
        UserRepo::new(&format!("{USER}/dep{ii}"))
    }

    fn new_dep_graph() -> DependencyGraph {
        // create the main package
        let this = dep(THIS);
        DependencyGraph::new(Some(this))
    }

    fn add_relations(dep_graph: &mut DependencyGraph, relations: &[&str]) {
        // create all the relations
        for relation in relations {
            let mut split = relation.split("->");
            let parent_str = split.next().unwrap().trim();
            let childs_str = split.next().unwrap();
            assert!(
                split.next().is_none(),
                "invalid relation (expected: parent -> [child1, child2, ..])"
            );

            let parent = dep(parent_str);

            let mut childs = vec![];
            for child_str in childs_str.split(',') {
                let child = dep(child_str.trim());
                childs.push(child.clone());

                // make sure that each child has their own presence in the cache
                dep_graph.cached_manifests.entry(child).or_default();
            }

            dep_graph.cached_manifests.insert(parent, childs);
        }
    }

    fn add_deps(dep_graph: &mut DependencyGraph, deps_str: &[&str]) -> Result<()> {
        // make sure that each dep exists in the cached manifest
        let mut libs = vec![];
        for dep_str in deps_str {
            let dep = dep(dep_str);
            dep_graph.cached_manifests.entry(dep.clone()).or_default();
            libs.push(dep);
        }

        // now add each dep
        dep_graph.add_deps(libs)?;

        Ok(())
    }

    fn check_from_leaves_to_roots(expected: &str, obtained: Vec<UserRepo>) {
        let expected: Vec<UserRepo> = expected
            .split(',')
            .map(|dep_str| dep(dep_str.trim()))
            .collect();

        assert_eq!(expected, obtained);
    }

    #[test]
    fn test_simple_dep_graph() {
        let mut dep_graph = new_dep_graph();
        add_relations(&mut dep_graph, &["0 -> 1"]);
        add_deps(&mut dep_graph, &["0", "1"]).unwrap();

        assert_eq!(dep_graph.root.len(), 2);

        assert_eq!(dep_graph.root[0].dep, dep("0"));
        assert_eq!(dep_graph.root[0].deps.len(), 1);
        assert_eq!(dep_graph.root[0].deps[0].dep, dep("1"));

        assert_eq!(dep_graph.root[1].dep, dep("1"));
        assert_eq!(dep_graph.root[1].deps.len(), 0);

        check_from_leaves_to_roots("1,0", dep_graph.from_leaves_to_roots());
    }

    #[test]
    fn test_simple_cycle() {
        let mut dep_graph = new_dep_graph();
        add_relations(&mut dep_graph, &["0 -> 1", "1 -> 0"]);
        assert!(add_deps(&mut dep_graph, &["0"]).is_err());
        assert!(add_deps(&mut dep_graph, &["1"]).is_err());
        add_deps(&mut dep_graph, &["2"]).unwrap();

        check_from_leaves_to_roots("2", dep_graph.from_leaves_to_roots());
    }

    #[test]
    fn test_more_complicated_cycle() {
        let mut dep_graph = new_dep_graph();
        add_relations(&mut dep_graph, &["0 -> 1", "1 -> 2", "2 -> 3", "3 -> 1"]);
        assert!(add_deps(&mut dep_graph, &["0"]).is_err());
        assert!(add_deps(&mut dep_graph, &["1"]).is_err());
        assert!(add_deps(&mut dep_graph, &["2"]).is_err());
        assert!(add_deps(&mut dep_graph, &["3"]).is_err());
    }

    #[test]
    fn test_diamond() {
        let mut dep_graph = new_dep_graph();
        add_relations(&mut dep_graph, &["0 -> 1", "1 -> 2", "3 -> 4", "4 -> 2"]);
        add_deps(&mut dep_graph, &["0", "3"]).unwrap();

        check_from_leaves_to_roots("2,1,0,4,3", dep_graph.from_leaves_to_roots());
    }

    #[test]
    fn test_direct_issue() {
        let mut dep_graph = new_dep_graph();
        add_relations(&mut dep_graph, &["0 -> 1", "1 -> 2", "2 -> 3", "3 -> 1"]);
        assert!(add_deps(&mut dep_graph, &["0", "3"]).is_err());
    }

    #[test]
    fn test_recursive_main_lib() {
        let mut dep_graph = new_dep_graph();
        assert!(add_deps(&mut dep_graph, &[THIS]).is_err());
    }
}
