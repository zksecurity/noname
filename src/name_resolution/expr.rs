use crate::{
    cli::packages::UserRepo,
    error::Result,
    imports::BUILTIN_FNS,
    parser::{types::ModulePath, CustomType, Expr, ExprKind},
    stdlib::QUALIFIED_BUILTINS,
};

use super::context::NameResCtx;

impl NameResCtx {
    pub(crate) fn resolve_expr(&self, expr: &mut Expr) -> Result<()> {
        let Expr {
            node_id: _,
            kind,
            span: _,
        } = expr;

        match kind {
            ExprKind::FnCall {
                module,
                fn_name,
                args,
            } => {
                if matches!(module, ModulePath::Local) && BUILTIN_FNS.get(&fn_name.value).is_some()
                {
                    // if it's a builtin, use `std::builtin`
                    *module = ModulePath::Absolute(UserRepo::new(QUALIFIED_BUILTINS));
                } else {
                    self.resolve(module, false)?;
                }

                for arg in args {
                    self.resolve_expr(arg)?;
                }
            }
            ExprKind::MethodCall {
                lhs,
                method_name: _,
                args,
            } => {
                self.resolve_expr(lhs)?;
                for arg in args {
                    self.resolve_expr(arg)?;
                }
            }
            ExprKind::Assignment { lhs, rhs } => {
                self.resolve_expr(lhs)?;
                self.resolve_expr(rhs)?;
            }
            ExprKind::FieldAccess { lhs, rhs: _ } => {
                self.resolve_expr(lhs)?;
            }
            ExprKind::BinaryOp {
                op: _,
                lhs,
                rhs,
                protected: _,
            } => {
                self.resolve_expr(lhs)?;
                self.resolve_expr(rhs)?;
            }
            ExprKind::Negated(expr) => {
                self.resolve_expr(expr)?;
            }
            ExprKind::Not(expr) => {
                self.resolve_expr(expr)?;
            }
            ExprKind::BigInt(_) => {}
            ExprKind::Variable { module, name: _ } => {
                self.resolve(module, false)?;
            }
            ExprKind::ArrayAccess { array, idx } => {
                self.resolve_expr(array)?;
                self.resolve_expr(idx)?;
            }
            ExprKind::ArrayDeclaration(items) => {
                for expr in items {
                    self.resolve_expr(expr)?;
                }
            }
            ExprKind::CustomTypeDeclaration {
                custom: struct_name,
                fields,
            } => {
                let CustomType {
                    module,
                    name: _,
                    span: _,
                } = struct_name;
                self.resolve(module, true)?;
                for (_field_name, field_value) in fields {
                    self.resolve_expr(field_value)?;
                }
            }
            ExprKind::Bool(_) => {}
            ExprKind::IfElse { cond, then_, else_ } => {
                self.resolve_expr(cond)?;
                self.resolve_expr(then_)?;
                self.resolve_expr(else_)?;
            }
        };

        Ok(())
    }
}
