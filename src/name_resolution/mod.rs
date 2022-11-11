use crate::{
    cli::packages::UserRepo,
    error::{Error, ErrorKind, Result},
    parser::{ConstDef, FunctionDef, RootKind, StructDef, AST},
};

use self::context::NameResCtx;

mod context;
mod expr;

pub struct NAST {
    pub ast: AST,
}

impl NAST {
    fn new(ast: AST) -> Self {
        Self { ast }
    }

    pub fn resolve_modules(this_module: Option<UserRepo>, mut ast: AST) -> Result<NAST> {
        let mut ctx = NameResCtx::new(this_module);

        // create a map of the imported modules (and how they are aliases)
        let mut abort = None;
        for root in &ast.0 {
            match &root.kind {
                // `use user::repo;`
                RootKind::Use(path) => {
                    // important: no struct or function definition can appear before a use declaration
                    if let Some(span) = abort {
                        return Err(Error::new(
                            "type-checker",
                            ErrorKind::OrderOfUseDeclaration,
                            span,
                        ));
                    }

                    // insert and detect duplicates
                    if ctx
                        .modules
                        .insert(path.submodule.value.clone(), path.clone())
                        .is_some()
                    {
                        return Err(ctx.error(
                            ErrorKind::DuplicateModule(path.submodule.value.clone()),
                            path.submodule.span,
                        ));
                    }
                }
                RootKind::FunctionDef(FunctionDef { span, .. })
                | RootKind::StructDef(StructDef { span, .. })
                | RootKind::ConstDef(ConstDef { span, .. }) => abort = Some(*span),
                RootKind::Comment(_) => (),
            }
        }

        // now go through the AST and mutate any module to its fully-qualified path
        for root in &mut ast.0 {
            match &mut root.kind {
                RootKind::FunctionDef(f) => ctx.resolve_fn_def(f)?,
                RootKind::StructDef(s) => ctx.resolve_struct_def(s)?,
                RootKind::ConstDef(c) => ctx.resolve_const_def(c)?,
                RootKind::Use(_) | RootKind::Comment(_) => (),
            }
        }

        Ok(NAST::new(ast))
    }
}
