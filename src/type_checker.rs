use crate::{
    ast::{Compiler, Environment},
    constants::Span,
    error::{Error, ErrorKind, Result},
    parser::{Expr, ExprKind, Function, FunctionSig, Path, RootKind, StmtKind, TyKind, AST},
    stdlib,
};

impl Expr {
    pub fn compute_type(&self, env: &Environment) -> Result<Option<TyKind>> {
        match &self.kind {
            ExprKind::FnCall { name, args } => typecheck_fn_call(env, name, args, self.span),
            ExprKind::Variable(_) => todo!(),
            ExprKind::Comparison(_, _, _) => todo!(),
            ExprKind::Op(_, lhs, rhs) => {
                let lhs_typ = lhs.compute_type(env)?.unwrap();
                let rhs_typ = rhs.compute_type(env)?.unwrap();

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
                let inner_typ = inner.compute_type(env)?.unwrap();
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
                let typ = env.get_type(ident).ok_or(Error {
                    kind: ErrorKind::UndefinedVariable,
                    span: self.span,
                })?;

                Ok(Some(typ.clone()))
            }
            ExprKind::ArrayAccess(path, expr) => {
                // only support scoped variable for now
                if path.len() != 1 {
                    unimplemented!();
                }

                // figure out if variable is in scope
                let name = &path.path[0].value;
                let typ = env.get_type(name).ok_or(Error {
                    kind: ErrorKind::UndefinedVariable,
                    span: self.span,
                })?;

                // check that expression is a bigint
                match expr.compute_type(env)? {
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
                    TyKind::Array(typkind, _) => Ok(Some(*typkind.clone())),
                    _ => panic!("not an array"),
                }
            }
        }
    }
}

impl Compiler {
    pub fn type_check(&mut self, env: &mut Environment, ast: &mut AST) -> Result<()> {
        let mut main_function_observed = false;
        //
        // Semantic analysis includes:
        // - type checking
        // - ?
        //

        for root in &mut ast.0 {
            match &root.kind {
                // `use crypto::poseidon;`
                RootKind::Use(path) => {
                    let path_iter = &mut path.path.iter();
                    let root_module = path_iter.next().expect("empty imports can't be parsed");

                    if root_module.value == "std" {
                        let module = stdlib::parse_std_import(path, path_iter)?;
                        if env
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
                }

                // `fn main() { ... }`
                RootKind::Function(function) => {
                    // TODO: support other functions
                    if !function.is_main() {
                        panic!("we do not yet support functions other than main()");
                    }

                    main_function_observed = true;

                    self.main_args.1 = function.span;

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
                                env.var_types
                                    .insert(arg.name.value.clone(), arg.typ.kind.clone());
                            }

                            TyKind::Array(..) => {
                                env.var_types
                                    .insert(arg.name.value.clone(), arg.typ.kind.clone());
                            }

                            TyKind::Bool => {
                                env.var_types
                                    .insert(arg.name.value.clone(), arg.typ.kind.clone());
                            }

                            t => panic!("unimplemented type {:?}", t),
                        }

                        //
                        self.main_args.0.insert(arg.name.value.clone(), arg.clone());
                    }

                    // the output value returned by the main function is also a main_args with a special name (public_output)
                    if let Some(typ) = &function.return_type {
                        if !matches!(typ.kind, TyKind::Field) {
                            unimplemented!();
                        }

                        let name = "public_output";

                        env.var_types.insert(name.to_string(), typ.kind.clone());
                    }

                    // type system pass!!!
                    self.type_check_fn_body(env, function)?;
                }

                // ignore comments
                RootKind::Comment(_comment) => (),
            }
        }

        // enforce that there's a main function
        assert!(main_function_observed);

        Ok(())
    }

    pub fn type_check_fn_body(&mut self, env: &mut Environment, function: &Function) -> Result<()> {
        let mut still_need_to_check_return_type = function.return_type.is_some();

        // only expressions need type info?
        for stmt in &function.body {
            match &stmt.kind {
                StmtKind::Assign { lhs, rhs } => {
                    // inferance can be easy: we can do it the Golang way and just use the type that rhs has (in `let` assignments)

                    // but first we need to compute the type of the rhs expression
                    let typ = rhs.compute_type(env)?.unwrap();

                    // store the type of lhs in the env
                    env.store_type(lhs.value.clone(), typ, lhs.span)?;
                }
                StmtKind::Expr(expr) => {
                    // make sure the expression does not return any type
                    // (it's a statement expression, it should only work via side effect)

                    let typ = expr.compute_type(env)?;
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

                    let typ = res.compute_type(env)?.unwrap();

                    if env.var_types["public_output"] != typ {
                        return Err(Error {
                            kind: ErrorKind::ReturnTypeMismatch(
                                env.var_types["public_output"].clone(),
                                typ,
                            ),
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
    env: &Environment,
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
            crate::ast::FuncInScope::BuiltIn(sig, _) => sig.clone(),
            crate::ast::FuncInScope::Library(_, _) => todo!(),
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
        if let Some(typ) = arg.compute_type(env)? {
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
