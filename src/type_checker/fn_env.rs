//! This module defines the context (or environment) that gets created when type checking a function.

use std::collections::HashMap;

use crate::{
    constants::Span,
    error::{Error, ErrorKind, Result},
    parser::types::TyKind,
};

/// Some type information on local variables that we want to track in the [TypedFnEnv] environment.
#[derive(Debug, Clone)]
pub struct TypeInfo {
    /// If the variable can be mutated or not.
    pub mutable: bool,

    /// Some type information.
    pub typ: TyKind,

    /// The span of the variable declaration.
    pub span: Span,
}

impl TypeInfo {
    pub fn new(typ: TyKind, span: Span) -> Self {
        Self {
            mutable: false,
            typ,
            span,
        }
    }

    pub fn new_mut(typ: TyKind, span: Span) -> Self {
        Self {
            mutable: true,
            ..Self::new(typ, span)
        }
    }
}

/// The environment we use to type check functions.
#[derive(Default, Debug, Clone)]
pub struct TypedFnEnv {
    /// The current nesting level.
    /// Starting at 0 (top level), and increasing as we go into a block.
    current_scope: usize,

    /// Vars local to their scope.
    /// This needs to be garbage collected when we exit a scope.
    // TODO: there's an output_type field that's a reserved keyword?
    vars: HashMap<String, (usize, TypeInfo)>,

    /// The forloop scopes if it is within a for loop.
    forloop_scopes: Vec<usize>,

    /// Determines if forloop variables are allowed to be accessed.
    forbid_forloop_scope: bool,

    /// Indicates if the function is a hint function.
    in_hint_fn: bool,
}

impl TypedFnEnv {
    /// Creates a new TypeEnv
    pub fn new(is_hint: bool) -> Self {
        Self {
            current_scope: 0,
            vars: HashMap::new(),
            forloop_scopes: Vec::new(),
            forbid_forloop_scope: false,
            in_hint_fn: is_hint,
        }
    }

    /// Enters a scoped block.
    pub fn nest(&mut self) {
        self.current_scope += 1;
    }

    /// Exits a scoped block.
    pub fn pop(&mut self) {
        self.current_scope = self.current_scope.checked_sub(1).expect("scope bug");

        //Remove variables as we exit the scope
        let current_scope = self.current_scope;
        self.vars
            .retain(|_name, (scope, _type_info)| *scope <= current_scope);
    }

    pub fn forbid_forloop_scope(&mut self) {
        self.forbid_forloop_scope = true;
    }

    pub fn allow_forloop_scope(&mut self) {
        self.forbid_forloop_scope = false;
    }

    /// Returns whether it is in a for loop.
    pub fn is_in_forloop(&self) -> bool {
        if let Some(scope) = self.forloop_scopes.last() {
            self.current_scope >= *scope
        } else {
            false
        }
    }

    /// Pushes a new for loop scope.
    pub fn start_forloop(&mut self) {
        self.forloop_scopes.push(self.current_scope);
    }

    /// Pop the last loop scope.
    pub fn end_forloop(&mut self) {
        self.forloop_scopes.pop();
    }

    /// Returns whether it is in a hint function.
    pub fn is_in_hint_fn(&self) -> bool {
        self.in_hint_fn
    }

    /// Returns true if a scope is a prefix of our scope.
    pub fn is_in_scope(&self, prefix_scope: usize) -> bool {
        self.current_scope >= prefix_scope
    }

    /// Since currently we don't support unrolling, the generic function calls are assumed to target a same instance.
    /// Each loop iteration should instantiate generic function calls with the same parameters.
    /// This assumption requires a few type checking rules to forbid the cases that needs unrolling.
    /// Forbid rules:
    /// - Access to variables within the for loop scope.
    /// - Access to mutable variables, except if it is an array.
    ///   Because once the array is declared, the size is fixed even if the array is mutable,
    ///   so the generic value resolved from array size will be same for generic function argument.
    pub fn is_forbidden(&self, scope: usize, ty_info: TypeInfo) -> bool {
        let in_forbidden_scope = if let Some(forloop_scope) = self.forloop_scopes.first() {
            scope >= *forloop_scope
        } else {
            false
        };

        let forbidden_mutable = ty_info.mutable
            && !matches!(
                ty_info.typ,
                TyKind::GenericSizedArray(..) | TyKind::Array(..)
            );

        self.forbid_forloop_scope && (in_forbidden_scope || forbidden_mutable)
    }

    /// Stores type information about a local variable.
    /// Note that we forbid shadowing at all scopes.
    pub fn store_type(&mut self, ident: String, type_info: TypeInfo) -> Result<()> {
        match self
            .vars
            .insert(ident.clone(), (self.current_scope, type_info.clone()))
        {
            Some(_) => Err(Error::new(
                "type-checker",
                ErrorKind::DuplicateDefinition(ident),
                type_info.span,
            )),
            None => Ok(()),
        }
    }

    pub fn get_type(&self, ident: &str) -> Result<Option<&TyKind>> {
        Ok(self.get_type_info(ident)?.map(|type_info| &type_info.typ))
    }

    pub fn mutable(&self, ident: &str) -> Result<Option<bool>> {
        Ok(self
            .get_type_info(ident)?
            .map(|type_info| type_info.mutable))
    }

    /// Retrieves type information on a variable, given a name.
    /// If the variable is not in scope, return false.
    // TODO: return an error no?
    pub fn get_type_info(&self, ident: &str) -> Result<Option<&TypeInfo>> {
        if let Some((scope, type_info)) = self.vars.get(ident) {
            if self.is_forbidden(*scope, type_info.clone()) {
                return Err(Error::new(
                    "type-checker",
                    ErrorKind::VarAccessForbiddenInForLoop(ident.to_string()),
                    type_info.span,
                ));
            }

            if self.is_in_scope(*scope) {
                Ok(Some(type_info))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }
}
