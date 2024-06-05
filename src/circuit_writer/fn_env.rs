use std::collections::HashMap;

use crate::{
    backends::{BackendField, BackendVar},
    parser::types::TyKind,
    var::Var,
};

/// Information about a variable.
#[derive(Debug, Clone)]
pub struct VarInfo<F, C>
where
    F: BackendField,
    C: BackendVar,
{
    /// The variable.
    pub var: Var<F, C>,

    // TODO: we could also store this on the expression node... but I think this is lighter
    pub mutable: bool,

    /// We keep track of the type of variables, eventhough we're not in the typechecker anymore,
    /// because we need to know the type for method calls.
    // TODO: why is this an option?
    pub typ: Option<TyKind>,
}

impl<F: BackendField, C: BackendVar> VarInfo<F, C> {
    #[must_use]
    pub fn new(var: Var<F, C>, mutable: bool, typ: Option<TyKind>) -> Self {
        Self { var, mutable, typ }
    }

    #[must_use]
    pub fn reassign(&self, var: Var<F, C>) -> Self {
        Self {
            var,
            mutable: self.mutable,
            typ: self.typ.clone(),
        }
    }

    #[must_use]
    pub fn reassign_range(&self, var: Var<F, C>, start: usize, len: usize) -> Self {
        // sanity check
        assert_eq!(var.len(), len);

        // create new cvars by modifying a specific range
        let mut cvars = self.var.cvars.clone();
        let cvars_range = &mut cvars[start..start + len];
        cvars_range.clone_from_slice(&var.cvars);

        let var = Var::new(cvars, self.var.span);

        Self {
            var,
            mutable: self.mutable,
            typ: self.typ.clone(),
        }
    }
}

/// Is used to store functions' scoped variables.
/// This include inputs and output of the function,  as well as local variables.
/// You can think of it as a function call stack.
#[derive(Default, Debug, Clone)]
pub struct FnEnv<F, C>
where
    F: BackendField,
    C: BackendVar,
{
    /// The current nesting level.
    /// Starting at 0 (top level), and increasing as we go into a block.
    current_scope: usize,

    /// Used by the private and public inputs,
    /// and any other external variables created in the circuit
    /// This needs to be garbage collected when we exit a scope.
    /// Note: The `usize` is the scope in which the variable was created.
    vars: HashMap<String, (usize, VarInfo<F, C>)>,
}

impl<F: BackendField, C: BackendVar> FnEnv<F, C> {
    /// Creates a new `FnEnv`
    #[must_use]
    pub fn new() -> Self {
        Self {
            current_scope: 0,
            vars: HashMap::new(),
        }
    }

    /// Enters a scoped block.
    pub fn nest(&mut self) {
        self.current_scope += 1;
    }

    /// Exits a scoped block.
    pub fn pop(&mut self) {
        self.current_scope = self.current_scope.checked_sub(1).expect("scope bug");

        // remove variables as we exit the scope
        // (we don't need to keep them around to detect shadowing,
        // as we already did that in type checker)
        let mut to_delete = vec![];
        for (name, (scope, _)) in &self.vars {
            if *scope > self.current_scope {
                to_delete.push(name.clone());
            }
        }
        for d in to_delete {
            self.vars.remove(&d);
        }
    }

    /// Returns true if a scope is a prefix of our scope.
    fn is_in_scope(&self, prefix_scope: usize) -> bool {
        self.current_scope >= prefix_scope
    }

    /// Stores type information about a local variable.
    /// Note that we forbid shadowing at all scopes.
    pub fn add_local_var(&mut self, var_name: String, var_info: VarInfo<F, C>) {
        let scope = self.current_scope;

        if self
            .vars
            .insert(var_name.clone(), (scope, var_info))
            .is_some()
        {
            panic!("type checker error: var `{var_name}` already exists");
        }
    }

    /// Retrieves type information on a variable, given a name.
    /// If the variable is not in scope, return false.
    // TODO: return an error no?
    #[must_use]
    pub fn get_local_var(&self, var_name: &str) -> VarInfo<F, C> {
        let (scope, var_info) = self
            .vars
            .get(var_name)
            .unwrap_or_else(|| panic!("type checking bug: local variable `{var_name}` not found"));
        assert!(
            self.is_in_scope(*scope),
            "type checking bug: local variable `{var_name}` not in scope"
        );

        var_info.clone()
    }

    pub fn reassign_local_var(&mut self, var_name: &str, var: Var<F, C>) {
        // get the scope first, we don't want to modify that
        let (scope, var_info) = self
            .vars
            .get(var_name)
            .expect("type checking bug: local variable for reassigning not found");

        assert!(
            self.is_in_scope(*scope),
            "type checking bug: local variable for reassigning not in scope"
        );

        assert!(
            var_info.mutable,
            "type checking bug: local variable for reassigning is not mutable"
        );

        let var_info = var_info.reassign(var);
        self.vars.insert(var_name.to_string(), (*scope, var_info));
    }

    /// Same as [`Self::reassign_var`], but only reassigns a specific range of the variable.
    pub fn reassign_var_range(&mut self, var_name: &str, var: Var<F, C>, start: usize, len: usize) {
        // get the scope first, we don't want to modify that
        let (scope, var_info) = self
            .vars
            .get(var_name)
            .expect("type checking bug: local variable for reassigning not found");

        assert!(
            self.is_in_scope(*scope),
            "type checking bug: local variable for reassigning not in scope"
        );

        assert!(
            var_info.mutable,
            "type checking bug: local variable for reassigning is not mutable"
        );

        let var_info = var_info.reassign_range(var, start, len);
        self.vars.insert(var_name.to_string(), (*scope, var_info));
    }
}
