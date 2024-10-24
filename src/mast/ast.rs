use crate::{
    backends::Backend,
    parser::{Expr, ExprKind},
};

use super::MastCtx;

impl Expr {
    /// Convert an expression to another expression, with the same span and a regenerated node id.
    pub fn to_mast<B: Backend>(&self, ctx: &mut MastCtx<B>, kind: &ExprKind) -> Expr {
        Expr {
            node_id: ctx.next_node_id(),
            kind: kind.clone(),
            ..self.clone()
        }
    }
}
