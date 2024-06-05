use crate::{
    backends::Backend,
    cli::packages::UserRepo,
    error::{Error, ErrorKind, Result},
    parser::{ConstDef, FunctionDef, RootKind, StructDef, AST},
};

use self::context::NameResCtx;

mod context;
mod expr;

pub struct NAST<B>
where
    B: Backend,
{
    pub ast: AST<B>,
}

impl<B: Backend> NAST<B> {
    fn new(ast: AST<B>) -> Self {
        Self { ast }
    }

    pub fn resolve_modules(this_module: Option<UserRepo>, mut ast: AST<B>) -> Result<NAST<B>> {
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

#[cfg(test)]
mod tests {
    use crate::{
        backends::kimchi::KimchiVesta,
        lexer::Token,
        parser::{
            types::{ModulePath, StmtKind},
            ExprKind,
        },
    };

    use super::*;

    const CODE: &str = r"
    use user::repo;

    const some_cst = 0;

    struct Thing {
        bb: Field,
    }

    fn some_func(xx: [Field; 3], yy: Field) -> [Field; 3] {
        let aa = Thing { bb: yy };
        return repo::thing(xx, aa);
    }
";

    #[test]
    fn test_name_res() {
        let tokens = Token::parse(0, CODE).unwrap();
        let (ast, _node_id) = AST::<KimchiVesta>::parse(0, tokens, 0).unwrap();
        let nast = NAST::resolve_modules(None, ast).unwrap();

        // find constant declaration
        let mut roots = nast
            .ast
            .0
            .iter()
            .skip_while(|r| !matches!(r.kind, RootKind::ConstDef(_)));
        let cst = match &roots.next().unwrap().kind {
            RootKind::ConstDef(c) => c,
            _ => panic!("expected const def"),
        };
        assert!(matches!(cst.module, ModulePath::Local));

        // struct definition
        let struct_def = match &roots.next().unwrap().kind {
            RootKind::StructDef(d) => d,
            _ => panic!("expected const def"),
        };
        assert!(matches!(struct_def.module, ModulePath::Local));

        // return statement
        let fn_def = match &roots.next().unwrap().kind {
            RootKind::FunctionDef(d) => d,
            _ => panic!("expected const def"),
        };
        let fn_call = match &fn_def.body[1].kind {
            StmtKind::Return(e) => e,
            _ => panic!("expected assignment"),
        };
        let module = match &fn_call.kind {
            ExprKind::FnCall { module, .. } => module,
            _ => panic!("expected struct"),
        };

        match module {
            ModulePath::Absolute(u) if u == &UserRepo::new("user/repo") => (),
            _ => panic!("expected absolute module path"),
        };
    }

    #[test]
    fn test_name_res_for_library() {
        let user_repo = UserRepo::new("mimoo/example");

        let tokens = Token::parse(0, CODE).unwrap();
        let (ast, _node_id) = AST::<KimchiVesta>::parse(0, tokens, 0).unwrap();
        let nast = NAST::resolve_modules(Some(user_repo.clone()), ast).unwrap();

        // find constant declaration
        let mut roots = nast
            .ast
            .0
            .iter()
            .skip_while(|r| !matches!(r.kind, RootKind::ConstDef(_)));
        let cst = match &roots.next().unwrap().kind {
            RootKind::ConstDef(c) => c,
            _ => panic!("expected const def"),
        };

        match &cst.module {
            ModulePath::Absolute(u) if u == &UserRepo::new("mimoo/example") => (),
            _ => panic!("expected absolute module path"),
        };

        // struct definition
        let struct_def = match &roots.next().unwrap().kind {
            RootKind::StructDef(d) => d,
            _ => panic!("expected const def"),
        };

        match &struct_def.module {
            ModulePath::Absolute(u) if u == &UserRepo::new("mimoo/example") => (),
            _ => panic!("expected absolute module path"),
        };

        // return statement
        let fn_def = match &roots.next().unwrap().kind {
            RootKind::FunctionDef(d) => d,
            _ => panic!("expected const def"),
        };
        let fn_call = match &fn_def.body[1].kind {
            StmtKind::Return(e) => e,
            _ => panic!("expected assignment"),
        };
        let module = match &fn_call.kind {
            ExprKind::FnCall { module, .. } => module,
            _ => panic!("expected struct"),
        };

        match module {
            ModulePath::Absolute(u) if u == &UserRepo::new("user/repo") => (),
            _ => panic!("expected absolute module path"),
        };
    }
}
