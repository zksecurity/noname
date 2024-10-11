use crate::{
    cli::packages::UserRepo,
    error::Result,
    parser::{types::ModulePath, CustomType, Expr, ExprKind},
    stdlib::builtins::{BUILTIN_FN_NAMES, QUALIFIED_BUILTINS},
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
                unsafe_attr: _,
            } => {
                if matches!(module, ModulePath::Local)
                    && BUILTIN_FN_NAMES.contains(&fn_name.value.as_str())
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
            ExprKind::BigUInt(_) => {}
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
            ExprKind::RepeatedArrayInit { item, size } => {
                self.resolve_expr(item)?;
                self.resolve_expr(size)?;
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
