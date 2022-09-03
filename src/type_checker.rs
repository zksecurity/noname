use std::collections::HashMap;

use crate::{
    constants::Span,
    error::{Error, ErrorKind, Result},
    imports::{FuncInScope, GlobalEnv},
    parser::{Expr, ExprKind, Function, FunctionSig, Path, RootKind, StmtKind, TyKind, AST},
};

//
// Expr
//

impl Expr {
    pub fn compute_type(&self, env: &GlobalEnv, type_env: &mut TypeEnv) -> Result<Option<TyKind>> {
        match &self.kind {
            ExprKind::FnCall { name, args } => {
                typecheck_fn_call(env, type_env, name, args, self.span)
            }
            ExprKind::Variable(_) => todo!(),
            ExprKind::Assignment { lhs, rhs } => todo!(),
            ExprKind::Comparison(_, _, _) => todo!(),
            ExprKind::Op(_, lhs, rhs) => {
                let lhs_typ = lhs.compute_type(env, type_env)?.unwrap();
                let rhs_typ = rhs.compute_type(env, type_env)?.unwrap();

                if lhs_typ != rhs_typ {
                    // only allow bigint mixed with field
                    match (&lhs_typ, &rhs_typ) {
                        (TyKind::BigInt, TyKind::Field) | (TyKind::Field, TyKind::BigInt) => (),
                        _ => {
                            return Err(Error {
                                kind: ErrorKind::MismatchType(lhs_typ.clone(), rhs_typ.clone()),
                                span: self.span,
                            })
                        }
                    }
                }

                Ok(Some(lhs_typ))
            }
            ExprKind::Negated(inner) => {
                let inner_typ = inner.compute_type(env, type_env)?.unwrap();
                if !matches!(inner_typ, TyKind::Bool) {
                    return Err(Error {
                        kind: ErrorKind::MismatchType(TyKind::Bool, inner_typ),
                        span: self.span,
                    });
                }

                Ok(Some(TyKind::Bool))
            }
            ExprKind::BigInt(_) => Ok(Some(TyKind::BigInt)),
            ExprKind::Bool(_) => Ok(Some(TyKind::Bool)),
            ExprKind::Identifier(ident) => {
                let typ = type_env.get_type(ident).ok_or(Error {
                    kind: ErrorKind::UndefinedVariable,
                    span: self.span,
                })?;

                Ok(Some(typ))
            }
            ExprKind::ArrayAccess(path, expr) => {
                // only support scoped variable for now
                if path.len() != 1 {
                    unimplemented!();
                }

                // figure out if variable is in scope
                let name = &path.path[0].value;
                let typ = type_env.get_type(name).ok_or(Error {
                    kind: ErrorKind::UndefinedVariable,
                    span: self.span,
                })?;

                // check that expression is a bigint
                match expr.compute_type(env, type_env)? {
                    Some(TyKind::BigInt) => (),
                    _ => {
                        return Err(Error {
                            kind: ErrorKind::ExpectedConstant,
                            span: self.span,
                        })
                    }
                };

                //
                match typ {
                    TyKind::Array(typkind, _) => Ok(Some(*typkind)),
                    _ => panic!("not an array"),
                }
            }
        }
    }
}

//
// Type Environment
//

#[derive(Debug, Clone)]
pub struct TypeInfo {
    pub mutable: bool,
    pub typ: TyKind,
    pub span: Span,
}

#[derive(Default, Debug)]
pub struct TypeEnv {
    /// created by the type checker, gives a type to every external variable
    pub var_types: HashMap<String, TypeInfo>,
}

impl TypeEnv {
    pub fn store_type(&mut self, ident: String, type_info: TypeInfo) -> Result<()> {
        match self.var_types.insert(ident.clone(), type_info.clone()) {
            Some(_) => Err(Error {
                kind: ErrorKind::DuplicateDefinition(ident),
                span: type_info.span,
            }),
            None => Ok(()),
        }
    }

    pub fn get_type(&self, ident: &str) -> Option<TyKind> {
        self.var_types.get(ident).map(|t| t.typ.clone())
    }
}

//
// Type checking
//

/// TAST for Typed-AST. Not sure how else to call this,
/// this is to make sure we call this compilation phase before the actual compilation.
pub struct TAST {
    pub ast: AST,
    pub global_env: GlobalEnv,
}

impl TAST {
    /// This takes the AST produced by the parser, and performs two things:
    /// - resolves imports
    /// - type checks
    pub fn analyze(ast: AST) -> Result<TAST> {
        // enforce a main function
        let mut main_function_observed = false;

        //
        // inject some utility builtin functions in the scope
        // TODO: should we really import them by default?

        let mut global_env = GlobalEnv::default();

        global_env.resolve_global_imports()?;

        //
        // Resolve imports
        //

        for root in &ast.0 {
            match &root.kind {
                // `use crypto::poseidon;`
                RootKind::Use(path) => global_env.resolve_imports(path)?,
                RootKind::Function(_) => (),
                RootKind::Comment(_) => (),
            }
        }

        //
        // Semantic analysis includes:
        // - type checking
        // - ?
        //

        for root in &ast.0 {
            match &root.kind {
                RootKind::Use(path) => (),

                // `fn main() { ... }`
                RootKind::Function(function) => {
                    // TODO: support other functions
                    if !function.is_main() {
                        panic!("we do not yet support functions other than main()");
                    }

                    main_function_observed = true;

                    let mut type_env = TypeEnv::default();

                    global_env.main_args.1 = function.span;

                    // store variables and their types in the env
                    for arg in &function.arguments {
                        // public_output is a reserved name,
                        // associated automatically to the public output of the main function
                        if arg.name.value == "public_output" {
                            return Err(Error {
                                kind: ErrorKind::PublicOutputReserved,
                                span: arg.name.span,
                            });
                        }

                        match &arg.typ.kind {
                            TyKind::Field => {
                                type_env.store_type(
                                    arg.name.value.clone(),
                                    TypeInfo {
                                        mutable: false,
                                        typ: arg.typ.kind.clone(),
                                        span: arg.span,
                                    },
                                )?;
                            }

                            TyKind::Array(..) => {
                                type_env.store_type(
                                    arg.name.value.clone(),
                                    TypeInfo {
                                        mutable: false,
                                        typ: arg.typ.kind.clone(),
                                        span: arg.span,
                                    },
                                )?;
                            }

                            TyKind::Bool => {
                                type_env.store_type(
                                    arg.name.value.clone(),
                                    TypeInfo {
                                        mutable: false,
                                        typ: arg.typ.kind.clone(),
                                        span: arg.span,
                                    },
                                )?;
                            }

                            t => panic!("unimplemented type {:?}", t),
                        }

                        //
                        global_env
                            .main_args
                            .0
                            .insert(arg.name.value.clone(), arg.clone());
                    }

                    // the output value returned by the main function is also a main_args with a special name (public_output)
                    if let Some(typ) = &function.return_type {
                        if !matches!(typ.kind, TyKind::Field) {
                            unimplemented!();
                        }

                        let name = "public_output";

                        type_env.store_type(
                            name.to_string(),
                            TypeInfo {
                                mutable: true,
                                typ: typ.kind.clone(),
                                span: typ.span,
                            },
                        )?;
                    }

                    // type system pass!!!
                    Self::type_check_fn_body(&global_env, type_env, function)?;
                }

                // ignore comments
                RootKind::Comment(_comment) => (),
            }
        }

        // enforce that there's a main function
        assert!(main_function_observed);

        Ok(TAST { ast, global_env })
    }

    pub fn type_check_fn_body(
        env: &GlobalEnv,
        mut type_env: TypeEnv,
        function: &Function,
    ) -> Result<()> {
        let mut still_need_to_check_return_type = function.return_type.is_some();

        // only expressions need type info?
        for stmt in &function.body {
            match &stmt.kind {
                StmtKind::Assign { mutable, lhs, rhs } => {
                    // inferance can be easy: we can do it the Golang way and just use the type that rhs has (in `let` assignments)

                    // but first we need to compute the type of the rhs expression
                    let typ = rhs.compute_type(env, &mut type_env)?.unwrap();

                    let type_info = TypeInfo {
                        mutable: *mutable,
                        typ,
                        span: lhs.span,
                    };

                    // store the type of lhs in the env
                    type_env.store_type(lhs.value.clone(), type_info)?;
                }
                StmtKind::For {
                    var,
                    start,
                    end,
                    body,
                } => unimplemented!(),
                StmtKind::Expr(expr) => {
                    // make sure the expression does not return any type
                    // (it's a statement expression, it should only work via side effect)

                    let typ = expr.compute_type(env, &mut type_env)?;
                    if typ.is_some() {
                        return Err(Error {
                            kind: ErrorKind::ExpectedUnitExpr,
                            span: expr.span,
                        });
                    }
                }
                StmtKind::Return(res) => {
                    // TODO: warn if there's code after the return?

                    // infer the return type and check if it's the same as the function return type?
                    if !function.is_main() {
                        unimplemented!();
                    }

                    assert!(still_need_to_check_return_type);

                    let typ = res.compute_type(env, &mut type_env)?.unwrap();

                    let expected = match type_env.get_type("public_output") {
                        Some(t) => t,
                        None => panic!("return statement when function signature doesn't have a return value (TODO: replace by error)"),
                    };

                    if expected != typ {
                        return Err(Error {
                            kind: ErrorKind::ReturnTypeMismatch(expected, typ),
                            span: stmt.span,
                        });
                    }

                    still_need_to_check_return_type = false;
                }
                StmtKind::Comment(_) => (),
            }
        }

        if still_need_to_check_return_type {
            return Err(Error {
                kind: ErrorKind::MissingPublicOutput,
                span: function.span,
            });
        }

        Ok(())
    }
}

pub fn typecheck_fn_call(
    env: &GlobalEnv,
    type_env: &mut TypeEnv,
    name: &Path,
    args: &[Expr],
    span: Span,
) -> Result<Option<TyKind>> {
    // retrieve the function sig in the env
    let path_len = name.path.len();
    let sig: FunctionSig = if path_len == 1 {
        // functions present in the scope
        let fn_name = &name.path[0].value;
        match env.functions.get(fn_name).ok_or(Error {
            kind: ErrorKind::UndefinedFunction(fn_name.clone()),
            span: name.span,
        })? {
            FuncInScope::BuiltIn(sig, _) => sig.clone(),
            FuncInScope::Library(_, _) => todo!(),
        }
    } else if path_len == 2 {
        // check module present in the scope
        let module = &name.path[0];
        let fn_name = &name.path[1];
        let module = env.modules.get(&module.value).ok_or(Error {
            kind: ErrorKind::UndefinedModule(module.value.clone()),
            span: module.span,
        })?;
        let (sig, _) = module.functions.get(&fn_name.value).ok_or(Error {
            kind: ErrorKind::UndefinedFunction(fn_name.value.clone()),
            span: fn_name.span,
        })?;
        sig.clone()
    } else {
        return Err(Error {
            kind: ErrorKind::InvalidFnCall("sub-sub modules unsupported"),
            span: name.span,
        });
    };

    // compute the arguments
    let mut typs = Vec::with_capacity(args.len());
    for arg in args {
        if let Some(typ) = arg.compute_type(env, type_env)? {
            typs.push((typ.clone(), arg.span));
        } else {
            return Err(Error {
                kind: ErrorKind::CannotComputeExpression,
                span: arg.span,
            });
        }
    }

    // argument length
    if sig.arguments.len() != typs.len() {
        return Err(Error {
            kind: ErrorKind::WrongNumberOfArguments {
                expected_args: sig.arguments.len(),
                observed_args: typs.len(),
            },
            span,
        });
    }

    // compare argument types with the function signature
    for (sig_arg, (typ, span)) in sig.arguments.iter().zip(typs) {
        if sig_arg.typ.kind != typ {
            // it's ok if a bigint is supposed to be a field no?
            // TODO: replace bigint -> constant?
            if matches!(
                (&sig_arg.typ.kind, &typ),
                (TyKind::Field, TyKind::BigInt) | (TyKind::BigInt, TyKind::Field)
            ) {
                continue;
            }

            return Err(Error {
                kind: ErrorKind::ArgumentTypeMismatch(sig_arg.typ.kind.clone(), typ),
                span,
            });
        }
    }

    Ok(sig.return_type.as_ref().map(|ty| ty.kind.clone()))
}
