use std::{collections::HashMap, fmt};

use crate::{
    circuit_writer::CircuitWriter,
    constants::Span,
    error::{Error, ErrorKind, Result},
    parser::{FuncArg, FunctionSig, Path},
    stdlib::{self, parse_fn_sigs, ImportedModule, BUILTIN_FNS},
    var::Var,
};

/// This seems to be used by both the type checker and the AST
// TODO: right now there's only one scope, but if we want to deal with multiple scopes then we'll need to make sure child scopes have access to parent scope, shadowing, etc.
#[derive(Default, Debug)]
pub struct GlobalEnv {
    /// the functions present in the scope
    /// contains at least the set of builtin functions (like assert_eq)
    pub functions: HashMap<String, FuncInScope>,

    /// stores the imported modules
    pub modules: HashMap<String, ImportedModule>,

    /// the arguments expected by main
    pub main_args: (HashMap<String, FuncArg>, Span),
}

pub type FuncType = fn(&mut CircuitWriter, &[Var], Span) -> Result<Option<Var>>;

pub enum FuncInScope {
    /// signature of the function
    BuiltIn(FunctionSig, FuncType),
    /// path, and signature of the function
    Library(Vec<String>, FunctionSig),
}

impl fmt::Debug for FuncInScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BuiltIn(arg0, _arg1) => f.debug_tuple("BuiltIn").field(arg0).field(&"_").finish(),
            Self::Library(arg0, arg1) => f.debug_tuple("Library").field(arg0).field(arg1).finish(),
        }
    }
}

impl GlobalEnv {
    pub fn resolve_global_imports(&mut self) -> Result<()> {
        let builtin_functions = parse_fn_sigs(&BUILTIN_FNS);
        for (sig, func) in builtin_functions {
            if self
                .functions
                .insert(sig.name.value.clone(), FuncInScope::BuiltIn(sig, func))
                .is_some()
            {
                panic!("global imports conflict");
            }
        }

        Ok(())
    }

    pub fn resolve_imports(&mut self, path: &Path) -> Result<()> {
        let path_iter = &mut path.path.iter();
        let root_module = path_iter.next().expect("empty imports can't be parsed");

        if root_module.value == "std" {
            let module = stdlib::parse_std_import(path, path_iter)?;
            if self
                .modules
                .insert(module.name.clone(), module.clone())
                .is_some()
            {
                return Err(Error {
                    kind: ErrorKind::DuplicateModule(module.name.clone()),
                    span: module.span,
                });
            }
        } else {
            // we only support std root module for now
            unimplemented!()
        };

        Ok(())
    }
}
