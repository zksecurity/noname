use crate::parser::{Expr, ExprKind};

use super::MastCtx;

impl Expr {
    /// Convert an expression to another expression, with the same span and a regenerated node id.
    pub fn to_mast(&self, ctx: &mut MastCtx, kind: &ExprKind) -> Expr {
        Expr {
            node_id: ctx.next_node_id(),
            kind: kind.clone(),
            ..self.clone()
        }
    }
}
