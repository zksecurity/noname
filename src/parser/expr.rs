use crate::syntax::is_type;
use crate::{
    constants::Span,
    error::{ErrorKind, Result},
    lexer::{Keyword, Token, TokenKind, Tokens},
};

use num_bigint::BigUint;
use serde::{Deserialize, Serialize};

use super::{
    types::{parse_fn_call_args, parse_type_declaration, Ident, ModulePath},
    CustomType, ParserCtx,
};

//~
//~ ## Expression
//~
//~ Backus–Naur Form (BNF) grammar:
//~
//~ expr ::=
//~     | expr { bin_op expr }
//~     | "-" expr
//~     | "(" expr ")"
//~     | numeric
//~     | ident
//~     | fn_call
//~     | array_access
//~ bin_op ::= "+" | "-" | "/" | "*" | "=="
//~ numeric ::= /[0-9]+/
//~ ident ::= /[A-Za-z_][A-Za-z_0-9]*/
//~ fn_call ::= ident "(" expr { "," expr } ")"
//~ array_access ::= ident "[" expr "]"
//~

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expr {
    pub node_id: usize,
    pub kind: ExprKind,
    pub span: Span,
}

impl Expr {
    pub fn new(ctx: &mut ParserCtx, kind: ExprKind, span: Span) -> Self {
        let node_id = ctx.next_node_id();
        Self {
            node_id,
            kind,
            span,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExprKind {
    /// `lhs(args)`
    FnCall {
        module: ModulePath,
        fn_name: Ident,
        args: Vec<Expr>,
    },

    /// `lhs.method_name(args)`
    MethodCall {
        lhs: Box<Expr>,
        method_name: Ident,
        args: Vec<Expr>,
    },

    /// `let lhs = rhs`
    Assignment { lhs: Box<Expr>, rhs: Box<Expr> },

    /// `lhs.rhs`
    FieldAccess { lhs: Box<Expr>, rhs: Ident },

    /// `lhs <op> rhs`
    BinaryOp {
        op: Op2,
        lhs: Box<Expr>,
        rhs: Box<Expr>,

        /// is it surrounded by parenthesis?
        protected: bool,
    },

    /// `-expr`
    Negated(Box<Expr>),

    /// `!bool_expr`
    Not(Box<Expr>),

    /// any numbers
    BigUInt(BigUint),

    /// a variable or a type. For example, `mod::A`, `x`, `y`, etc.
    // TODO: change to `identifier` or `path`?
    Variable { module: ModulePath, name: Ident },

    /// An array access, for example:
    /// `lhs[idx]`
    ArrayAccess { array: Box<Expr>, idx: Box<Expr> },

    /// `[ ... ]`
    ArrayDeclaration(Vec<Expr>),

    /// `name { fields }`
    CustomTypeDeclaration {
        custom: CustomType,
        fields: Vec<(Ident, Expr)>,
    },

    /// `true` or `false`
    Bool(bool),

    /// `if cond { then_ } else { else_ }`
    IfElse {
        cond: Box<Expr>,
        then_: Box<Expr>,
        else_: Box<Expr>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Op2 {
    Addition,
    Subtraction,
    Multiplication,
    Division,
    Equality,
    BoolAnd,
    BoolOr,
}

impl Expr {
    /// Parses until it finds something it doesn't know, then returns without consuming the token it doesn't know (the caller will have to make sense of it)
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self> {
        let token = tokens.bump_err(ctx, ErrorKind::MissingExpression)?;
        let span = token.span;

        let lhs = match token.kind {
            // numeric
            TokenKind::BigUInt(b) => Expr::new(ctx, ExprKind::BigUInt(b), span),

            // identifier
            TokenKind::Identifier(value) => {
                let maybe_module = Ident::new(value, span);

                // is it a qualified identifier?
                // name::other_name
                //     ^^
                match tokens.peek() {
                    Some(Token {
                        kind: TokenKind::DoubleColon,
                        ..
                    }) => {
                        tokens.bump(ctx); // ::

                        // mod::expr
                        //      ^^^^
                        let name = match tokens.bump(ctx) {
                            Some(Token {
                                kind: TokenKind::Identifier(value),
                                span,
                            }) => Ident::new(value, span),
                            _ => panic!("cannot qualify a non-identifier"),
                        };

                        Expr::new(
                            ctx,
                            ExprKind::Variable {
                                module: ModulePath::Alias(maybe_module),
                                name,
                            },
                            span,
                        )
                    }

                    // just an identifier
                    _ => Expr::new(
                        ctx,
                        ExprKind::Variable {
                            module: ModulePath::Local,
                            name: maybe_module,
                        },
                        span,
                    ),
                }
            }

            // negated expr
            TokenKind::Minus => {
                let expr = Expr::parse(ctx, tokens)?;

                Expr::new(ctx, ExprKind::Negated(Box::new(expr)), span)
            }

            // parenthesis
            TokenKind::LeftParen => {
                let mut expr = Expr::parse(ctx, tokens)?;
                tokens.bump_expected(ctx, TokenKind::RightParen)?;

                if let ExprKind::BinaryOp { protected, .. } = &mut expr.kind {
                    *protected = true;
                }

                expr
            }

            // true or false
            TokenKind::Keyword(Keyword::True) | TokenKind::Keyword(Keyword::False) => {
                let is_true = matches!(token.kind, TokenKind::Keyword(Keyword::True));

                Expr::new(ctx, ExprKind::Bool(is_true), span)
            }

            // `if cond { expr1 } else { expr2 }`
            TokenKind::Keyword(Keyword::If) => {
                // if cond { expr1 } else { expr2 }
                //    ^^^^
                let cond = Box::new(Expr::parse(ctx, tokens)?);

                // if cond { expr1 } else { expr2 }
                //         ^
                tokens.bump_expected(ctx, TokenKind::LeftCurlyBracket)?;

                // if cond { expr1 } else { expr2 }
                //           ^^^^^
                let then_ = Box::new(Expr::parse(ctx, tokens)?);

                if !matches!(
                    &then_.kind,
                    ExprKind::Variable { .. }
                        | ExprKind::Bool { .. }
                        | ExprKind::BigUInt { .. }
                        | ExprKind::FieldAccess { .. }
                        | ExprKind::ArrayAccess { .. }
                ) {
                    panic!("_then_ branch of ternary operator cannot be more than a variable")
                }

                // if cond { expr1 } else { expr2 }
                //                 ^
                tokens.bump_expected(ctx, TokenKind::RightCurlyBracket)?;

                // if cond { expr1 } else { expr2 }
                //                   ^^^^
                tokens.bump_expected(ctx, TokenKind::Keyword(Keyword::Else))?;

                // if cond { expr1 } else { expr2 }
                //                        ^
                tokens.bump_expected(ctx, TokenKind::LeftCurlyBracket)?;

                // if cond { expr1 } else { expr2 }
                //                          ^^^^^
                let else_ = Box::new(Expr::parse(ctx, tokens)?);

                if !matches!(
                    &else_.kind,
                    ExprKind::Variable { .. }
                        | ExprKind::Bool { .. }
                        | ExprKind::BigUInt { .. }
                        | ExprKind::FieldAccess { .. }
                        | ExprKind::ArrayAccess { .. }
                ) {
                    panic!("_else_ branch of ternary operator cannot be more than a variable")
                }

                // if cond { expr1 } else { expr2 }
                //                                ^
                let end = tokens.bump_expected(ctx, TokenKind::RightCurlyBracket)?;

                let span = span.merge_with(end.span);

                Expr::new(ctx, ExprKind::IfElse { cond, then_, else_ }, span)
            }

            // negation (logical NOT)
            TokenKind::Exclamation => {
                let expr = Expr::parse(ctx, tokens)?;

                Expr::new(ctx, ExprKind::Not(Box::new(expr)), span)
            }

            // array declaration
            TokenKind::LeftBracket => {
                let mut items = vec![];
                let last_span;

                // [1, 2];
                //  ^^^^
                loop {
                    let token = tokens.peek();

                    // [1, 2];
                    //      ^
                    if let Some(Token {
                        kind: TokenKind::RightBracket,
                        span,
                    }) = token
                    {
                        last_span = span;
                        tokens.bump(ctx);
                        break;
                    };

                    // [1, 2];
                    //  ^
                    let item = Expr::parse(ctx, tokens)?;
                    items.push(item);

                    // [1, 2];
                    //   ^  ^
                    let token = tokens.bump_err(ctx, ErrorKind::InvalidEndOfLine)?;
                    match &token.kind {
                        TokenKind::RightBracket => {
                            last_span = token.span;
                            break;
                        }
                        TokenKind::Comma => (),
                        _ => return Err(ctx.error(ErrorKind::InvalidEndOfLine, token.span)),
                    };
                }

                if items.is_empty() {
                    return Err(
                        ctx.error(ErrorKind::UnexpectedError("empty array declaration"), span)
                    );
                }

                Expr::new(
                    ctx,
                    ExprKind::ArrayDeclaration(items),
                    span.merge_with(last_span),
                )
            }

            // unrecognized pattern
            _ => {
                return Err(ctx.error(ErrorKind::InvalidExpression, token.span));
            }
        };

        // continue parsing. Potentially there's more
        lhs.parse_rhs(ctx, tokens)
    }

    /// an expression is sometimes unfinished when we parse it with [Self::parse],
    /// we use this function to see if the expression we just parsed (`self`) is actually part of a bigger expression
    fn parse_rhs(self, ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Expr> {
        // we peek into what's next to see if there's an expression that uses
        // the expression `self` we just parsed.
        // warning: ALL of the rules here should make use of `self`.
        let lhs = match tokens.peek() {
            // assignment
            Some(Token {
                kind: TokenKind::Equal,
                span,
            }) => {
                tokens.bump(ctx); // =

                // sanitize
                if !matches!(
                    &self.kind,
                    ExprKind::Variable { .. }
                        | ExprKind::ArrayAccess { .. }
                        | ExprKind::FieldAccess { .. },
                ) {
                    return Err(ctx.error(
                        ErrorKind::InvalidAssignmentExpression,
                        self.span.merge_with(span),
                    ));
                }

                let rhs = Expr::parse(ctx, tokens)?;
                let span = self.span.merge_with(rhs.span);

                Expr::new(
                    ctx,
                    ExprKind::Assignment {
                        lhs: Box::new(self),
                        rhs: Box::new(rhs),
                    },
                    span,
                )
            }

            // binary operation
            Some(Token {
                kind:
                    TokenKind::Plus
                    | TokenKind::Minus
                    | TokenKind::Star
                    | TokenKind::Slash
                    | TokenKind::DoubleEqual
                    | TokenKind::DoubleAmpersand
                    | TokenKind::DoublePipe
                    | TokenKind::Exclamation,
                ..
            }) => {
                // lhs + rhs
                //     ^
                let op = match tokens.bump(ctx).unwrap().kind {
                    TokenKind::Plus => Op2::Addition,
                    TokenKind::Minus => Op2::Subtraction,
                    TokenKind::Star => Op2::Multiplication,
                    TokenKind::Slash => Op2::Division,
                    TokenKind::DoubleEqual => Op2::Equality,
                    TokenKind::DoubleAmpersand => Op2::BoolAnd,
                    TokenKind::DoublePipe => Op2::BoolOr,
                    _ => unreachable!(),
                };

                // lhs + rhs
                //       ^^^
                let rhs = Expr::parse(ctx, tokens)?;

                let span = self.span.merge_with(rhs.span);

                // make sure that arithmetic operations are not chained without parenthesis
                if let ExprKind::BinaryOp { protected, .. } = &rhs.kind {
                    if !protected {
                        return Err(ctx.error(ErrorKind::MissingParenthesis, span));
                    }
                }

                //
                Expr::new(
                    ctx,
                    ExprKind::BinaryOp {
                        op,
                        lhs: Box::new(self),
                        rhs: Box::new(rhs),
                        protected: false,
                    },
                    span,
                )
            }

            // type declaration or if condition
            Some(Token {
                kind: TokenKind::LeftCurlyBracket,
                ..
            }) => {
                let ident = match &self.kind {
                    ExprKind::Variable { module, name } => {
                        // probably an if condition
                        if !is_type(&name.value) {
                            return Ok(self);
                        }

                        if !matches!(module, ModulePath::Local) {
                            return Err(ctx.error(
                                ErrorKind::UnexpectedError(
                                    "a type declaration cannot be qualified",
                                ),
                                name.span,
                            ));
                        }

                        name.clone()
                    }

                    // probably an if condition
                    _ => return Ok(self),
                };

                parse_type_declaration(ctx, tokens, ident)?
            }

            // array access
            Some(Token {
                kind: TokenKind::LeftBracket,
                ..
            }) => {
                tokens.bump(ctx); // [

                // sanity check
                if !matches!(
                    self.kind,
                    ExprKind::Variable { .. } | ExprKind::FieldAccess { .. }
                ) {
                    panic!("an array access can only follow a variable");
                }

                // array[idx]
                //       ^^^
                let idx = Expr::parse(ctx, tokens)?;

                // array[idx]
                //          ^
                tokens.bump_expected(ctx, TokenKind::RightBracket)?;

                let span = self.span.merge_with(idx.span);

                Expr::new(
                    ctx,
                    ExprKind::ArrayAccess {
                        array: Box::new(self),
                        idx: Box::new(idx),
                    },
                    span,
                )
            }

            // fn call
            Some(Token {
                kind: TokenKind::LeftParen,
                ..
            }) => {
                // sanitize
                let (module, fn_name) = match self.kind {
                    ExprKind::Variable { module, name } => (module, name),
                    _ => panic!("invalid fn name"),
                };

                // parse the arguments
                let (args, span) = parse_fn_call_args(ctx, tokens)?;

                let span = self.span.merge_with(span);

                Expr::new(
                    ctx,
                    ExprKind::FnCall {
                        module,
                        fn_name,
                        args,
                    },
                    span,
                )
            }

            // field access or method call
            // thing.thing2
            //      ^
            Some(Token {
                kind: TokenKind::Dot,
                ..
            }) => {
                let period = tokens.bump(ctx).unwrap(); // .

                // sanitize
                if !matches!(
                    &self.kind,
                    ExprKind::FieldAccess { .. }
                        | ExprKind::Variable { .. }
                        | ExprKind::ArrayAccess { .. }
                ) {
                    let span = self.span.merge_with(period.span);
                    return Err(ctx.error(ErrorKind::InvalidFieldAccessExpression, span));
                }

                // lhs.field
                //     ^^^^^
                let rhs = Ident::parse(ctx, tokens)?;
                let span = self.span.merge_with(rhs.span);

                // lhs.field or lhs.method_name()
                //     ^^^^^        ^^^^^^^^^^^^^
                match tokens.peek() {
                    // method call:
                    // lhs.method_name(...)
                    //     ^^^^^^^^^^^^^^^^
                    Some(Token {
                        kind: TokenKind::LeftParen,
                        ..
                    }) => {
                        // lhs.method_name(args)
                        //                 ^^^^
                        let (args, end_span) = parse_fn_call_args(ctx, tokens)?;

                        let span = span.merge_with(end_span);

                        Expr::new(
                            ctx,
                            ExprKind::MethodCall {
                                lhs: Box::new(self),
                                method_name: rhs,
                                args,
                            },
                            span,
                        )
                    }

                    // field access
                    // lhs.field
                    //     ^^^^^
                    _ => Expr::new(
                        ctx,
                        ExprKind::FieldAccess {
                            lhs: Box::new(self),
                            rhs,
                        },
                        span,
                    ),
                }
            }

            // it looks like the lhs is a valid expression in itself
            _ => return Ok(self),
        };

        lhs.parse_rhs(ctx, tokens)
    }
}
