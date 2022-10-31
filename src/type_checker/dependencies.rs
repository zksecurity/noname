use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    cli::packages::UserRepo,
    error::{Error, ErrorKind, Result},
    parser::types::{Ident, UsePath},
    stdlib::get_std_fn,
};

use super::{FnInfo, StructInfo, TypeChecker};

/// Contains metadata from other dependencies that might be use in this module.
#[derive(Default, Debug, Serialize, Deserialize)]
pub struct Dependencies {
    /// Maps each `user/repo` to their filename and type checker state.
    deps: HashMap<UserRepo, (String, TypeChecker)>,
}

impl Dependencies {
    pub fn get_file(&self, user_repo: &UserRepo) -> Option<&String> {
        self.deps.get(user_repo).map(|(file, _)| file)
    }

    pub fn get_type_checker(&self, user_repo: &UserRepo) -> Option<&TypeChecker> {
        self.deps.get(user_repo).map(|(_, tc)| tc)
    }

    pub fn add_type_checker(&mut self, user_repo: UserRepo, file: String, tc: TypeChecker) {
        assert!(self.deps.insert(user_repo, (file, tc)).is_none());
    }

    /// Expects the real use_path
    pub fn get_fn(&self, use_path: &UsePath, fn_name: &Ident) -> Result<FnInfo> {
        let user_repo: UserRepo = use_path.into();

        // hijack builtins (only std for now)
        if user_repo.user == "std" {
            return get_std_fn(&user_repo.repo, &fn_name.value, use_path.span);
        }

        // then check in imported dependencies
        let tast = self.get_type_checker(&user_repo).ok_or_else(|| {
            Error::new(
                ErrorKind::UnknownDependency(user_repo.to_string()),
                use_path.span,
            )
        })?;

        // we found the module, now let's find the function
        let fn_info = tast.fn_info(&fn_name.value).ok_or_else(|| {
            Error::new(
                ErrorKind::UnknownExternalFn(user_repo.to_string(), fn_name.value.to_string()),
                fn_name.span,
            )
        })?;

        Ok(fn_info.clone())
    }

    /// Expects the real use_path
    pub fn get_struct(&self, use_path: &UsePath, struct_name: &Ident) -> Result<StructInfo> {
        let user_repo: UserRepo = use_path.into();

        // first check in std
        if user_repo.user == "std" {
            todo!();
        }

        // then check in imported dependencies
        let tast = self.get_type_checker(&user_repo).ok_or_else(|| {
            Error::new(
                ErrorKind::UnknownDependency(user_repo.to_string()),
                use_path.span,
            )
        })?;

        // we found the module, now let's find the function
        tast.struct_info(&struct_name.value)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::UnknownExternalStruct(
                        user_repo.to_string(),
                        struct_name.value.to_string(),
                    ),
                    struct_name.span,
                )
            })
            .cloned()
    }
}
