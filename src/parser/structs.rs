use serde::{Deserialize, Serialize};

use crate::{
    constants::Span,
    error::{ErrorKind, Result},
    lexer::{Token, TokenKind, Tokens},
    syntax::is_type,
};

use super::{
    types::{Ident, ModulePath, Ty, TyKind},
    ParserCtx,
};

#[derive(Debug)]
pub struct StructDef {
    //pub attribute: Attribute,
    pub module: ModulePath, // name resolution
    pub name: CustomType,
    pub fields: Vec<(Ident, Ty)>,
    pub span: Span,
}

impl StructDef {
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self> {
        // ghetto way of getting the span of the function: get the span of the first token (name), then try to get the span of the last token
        let mut span = tokens
            .peek()
            .ok_or_else(|| {
                ctx.error(
                    ErrorKind::InvalidFunctionSignature("expected function name"),
                    ctx.last_span(),
                )
            })?
            .span;

        // struct Foo { a: Field, b: Field }
        //        ^^^

        let name = CustomType::parse(ctx, tokens)?;

        // struct Foo { a: Field, b: Field }
        //            ^
        tokens.bump_expected(ctx, TokenKind::LeftCurlyBracket)?;

        let mut fields = vec![];
        loop {
            // struct Foo { a: Field, b: Field }
            //                                 ^
            if let Some(Token {
                kind: TokenKind::RightCurlyBracket,
                ..
            }) = tokens.peek()
            {
                tokens.bump(ctx);
                break;
            }
            // struct Foo { a: Field, b: Field }
            //              ^
            let field_name = Ident::parse(ctx, tokens)?;

            // struct Foo { a: Field, b: Field }
            //               ^
            tokens.bump_expected(ctx, TokenKind::Colon)?;

            // struct Foo { a: Field, b: Field }
            //                 ^^^^^
            let field_ty = Ty::parse(ctx, tokens)?;
            span = span.merge_with(field_ty.span);
            fields.push((field_name, field_ty));

            // struct Foo { a: Field, b: Field }
            //                      ^          ^
            match tokens.peek() {
                Some(Token {
                    kind: TokenKind::Comma,
                    ..
                }) => {
                    tokens.bump(ctx);
                }
                Some(Token {
                    kind: TokenKind::RightCurlyBracket,
                    ..
                }) => {
                    tokens.bump(ctx);
                    break;
                }
                _ => {
                    return Err(
                        ctx.error(ErrorKind::ExpectedToken(TokenKind::Comma), ctx.last_span())
                    )
                }
            }
        }

        //
        Ok(StructDef {
            module: ModulePath::Local,
            name,
            fields,
            span,
        })
    }
}

// TODO: why is Default implemented here?
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct CustomType {
    pub module: ModulePath, // name resolution
    pub name: String,
    pub span: Span,
}

impl CustomType {
    pub fn parse(ctx: &mut ParserCtx, tokens: &mut Tokens) -> Result<Self> {
        let ty_name = tokens.bump_ident(ctx, ErrorKind::InvalidType)?;

        assert!(
            is_type(&ty_name.value),
            "type name should start with uppercase letter (TODO: better error"
        );

        // make sure that this type is allowed
        if !matches!(
            Ty::reserved_types(ModulePath::Local, ty_name.clone()),
            TyKind::Custom { .. }
        ) {
            return Err(ctx.error(ErrorKind::ReservedType(ty_name.value), ty_name.span));
        }

        Ok(Self {
            module: ModulePath::Local,
            name: ty_name.value,
            span: ty_name.span,
        })
    }
}
