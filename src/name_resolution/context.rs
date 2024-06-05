use std::collections::HashMap;

use ark_ff::Field;

use crate::{
    cli::packages::UserRepo,
    constants::Span,
    error::{Error, ErrorKind, Result},
    parser::{
        types::{FnArg, FnSig, FuncOrMethod, ModulePath, Stmt, StmtKind, TyKind},
        ConstDef, CustomType, FunctionDef, StructDef, UsePath,
    },
};

pub struct NameResCtx {
    /// Set only if this module is a third-party library.
    pub this_module: Option<UserRepo>,

    /// maps `module` to its original `use a::module`
    pub modules: HashMap<String, UsePath>,
}

impl NameResCtx {
    pub(crate) fn new(this_module: Option<UserRepo>) -> Self {
        Self {
            this_module,
            modules: HashMap::new(),
        }
    }

    pub(crate) fn error(&self, kind: ErrorKind, span: Span) -> Error {
        Error::new("name resolution", kind, span)
    }

    /// Resolves a single [`ModulePath`].
    /// `force` is set to `true` if it is expected that the [`ModulePath`] is set to [`ModulePath::Local`].
    /// This is usually the case for things like struct or function definitions.
    pub(crate) fn resolve(&self, module: &mut ModulePath, local: bool) -> Result<()> {
        match module {
            // if this is a local module, qualify it with its `user::repo` name
            ModulePath::Local => {
                if let Some(this) = &self.this_module {
                    *module = ModulePath::Absolute(this.clone());
                }
            }

            // we're expecting a local type, this can't be an alias
            ModulePath::Alias(_) if local => unreachable!(),

            // if this is a third-party module, qualify it with its `user::repo` name
            ModulePath::Alias(alias) => {
                if let Some(use_path) = self.modules.get(&alias.value) {
                    *module = ModulePath::Absolute(use_path.into());
                } else {
                    return Err(
                        self.error(ErrorKind::UndefinedModule(alias.value.clone()), alias.span)
                    );
                }
            }

            // fully-qualified modules are not allowed in noname. You need to import the repo first.
            ModulePath::Absolute(_) => unreachable!(),
        };

        Ok(())
    }

    pub(crate) fn resolve_fn_def(&self, fn_def: &mut FunctionDef) -> Result<()> {
        let FunctionDef { sig, body, span: _ } = fn_def;

        //
        // signature
        //

        let FnSig {
            kind,
            name: _,
            arguments,
            return_type,
        } = sig;

        match kind {
            // we set the fully-qualified name of the function
            FuncOrMethod::Function(module) => {
                self.resolve(module, true)?;
            }

            // or the fully-qualified name of the type implementing the method
            FuncOrMethod::Method(custom) => {
                let CustomType {
                    module,
                    name: _,
                    span: _,
                } = custom;

                self.resolve(module, true)?;
            }
        };

        // we resolve the fully-qualified types of the arguments and return value
        for arg in arguments {
            let FnArg {
                name: _,
                typ,
                attribute: _,
                span: _,
            } = arg;
            self.resolve_typ_kind(&mut typ.kind)?;
        }

        if let Some(return_type) = return_type {
            self.resolve_typ_kind(&mut return_type.kind)?;
        }

        //
        // body
        //

        for stmt in body {
            self.resolve_stmt(stmt)?;
        }

        Ok(())
    }

    fn resolve_typ_kind(&self, typ_kind: &mut TyKind) -> Result<()> {
        match typ_kind {
            TyKind::Field => (),
            TyKind::Custom { module, name: _ } => {
                self.resolve(module, false)?;
            }
            TyKind::BigInt => (),
            TyKind::Array(typ_kind, _) => self.resolve_typ_kind(typ_kind)?,
            TyKind::Bool => (),
        };

        Ok(())
    }

    pub(crate) fn resolve_struct_def(&self, struct_def: &mut StructDef) -> Result<()> {
        let StructDef {
            module,
            name: _,
            fields,
            span: _,
        } = struct_def;

        // we set the fully-qualified name of the struct
        self.resolve(module, true)?;

        // we resolve the fully-qualified types of the fields
        for (_field_name, field_typ) in fields {
            self.resolve_typ_kind(&mut field_typ.kind)?;
        }

        Ok(())
    }

    pub(crate) fn resolve_const_def<F: Field>(&self, cst_def: &mut ConstDef<F>) -> Result<()> {
        let ConstDef {
            module,
            name: _,
            value: _,
            span: _,
        } = cst_def;

        self.resolve(module, true)?;

        Ok(())
    }

    fn resolve_stmt(&self, stmt: &mut Stmt) -> Result<()> {
        let Stmt { kind, span: _ } = stmt;

        match kind {
            StmtKind::Assign {
                mutable: _,
                lhs: _,
                rhs,
            } => {
                self.resolve_expr(rhs)?;
            }
            StmtKind::Expr(expr) => self.resolve_expr(expr)?,
            StmtKind::Return(expr) => self.resolve_expr(expr)?,
            StmtKind::Comment(_) => (),
            StmtKind::ForLoop {
                var: _,
                range: _,
                body,
            } => {
                for stmt in body {
                    self.resolve_stmt(stmt)?;
                }
            }
        };

        Ok(())
    }
}
