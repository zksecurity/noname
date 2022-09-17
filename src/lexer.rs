use std::fmt::Display;

use crate::{
    constants::Span,
    error::{Error, ErrorKind, Result},
    syntax::{is_const, is_hexadecimal, is_identifier, is_numeric, is_type},
    tokens::Tokens,
};

#[derive(Default, Debug)]
pub struct LexerCtx {
    /// the offset of what we've read so far in the file
    offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keyword {
    /// Importing a library
    Use,
    /// A function
    Fn,
    /// New variable
    Let,
    /// Public input
    Pub,
    /// Return from a function
    Return,
    /// The boolean value `true`
    True,
    /// The boolean value `false`
    False,
    /// The `mut` keyword for mutable variables
    Mut,
    /// The `if` keyword
    If,
    /// The `else` keyword
    Else,
    /// The `for` keyword
    For,
    /// The `in` keyword for iterating
    In,
    /// Allows custom structs to be defined
    Struct,
    /// Allows constants to be defined
    Const,
}

impl Keyword {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "use" => Some(Self::Use),
            "fn" => Some(Self::Fn),
            "let" => Some(Self::Let),
            "pub" => Some(Self::Pub),
            "return" => Some(Self::Return),
            "true" => Some(Self::True),
            "false" => Some(Self::False),
            "mut" => Some(Self::Mut),
            "if" => Some(Self::If),
            "else" => Some(Self::Else),
            "for" => Some(Self::For),
            "in" => Some(Self::In),
            "struct" => Some(Self::Struct),
            "const" => Some(Self::Const),
            _ => None,
        }
    }
}

impl Display for Keyword {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let desc = match self {
            Self::Use => "use",
            Self::Fn => "fn",
            Self::Let => "let",
            Self::Pub => "pub",
            Self::Return => "return",
            Self::True => "true",
            Self::False => "false",
            Self::Mut => "mut",
            Self::If => "if",
            Self::Else => "else",
            Self::For => "for",
            Self::In => "in",
            Self::Struct => "struct",
            Self::Const => "const",
        };

        write!(f, "{}", desc)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    Keyword(Keyword),   // reserved keywords
    Identifier(String), // [a-z_](a-z0-9_)*
    Const(String),      // [A-Z_](A-Z0-9_)*
    Type(String),       // [A-Z](a-zA-Z0-9)*
    BigInt(String),     // (0-9)*
    Hex(String),        // 0x[0-9a-fA-F]+
    Period,             // .
    DoublePeriod,       // ..
    Comma,              // ,
    Colon,              // :
    DoubleColon,        // ::
    LeftParen,          // (
    RightParen,         // )
    LeftBracket,        // [
    RightBracket,       // ]
    LeftCurlyBracket,   // {
    RightCurlyBracket,  // }
    SemiColon,          // ;
    Slash,              // /
    Comment(String),    // // comment
    Greater,            // >
    Less,               // <
    Equal,              // =
    DoubleEqual,        // ==
    Plus,               // +
    Minus,              // -
    RightArrow,         // ->
    Star,               // *
    Ampersand,          // &
    Pipe,               // |
    Exclamation,        // !
                        //    Literal,               // "thing"
}

impl Display for TokenKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use TokenKind::*;
        let desc = match self {
            Keyword(_) => "keyword (use, let, etc.)",
            Identifier(_) => {
                "a lowercase alphanumeric (including underscore) string starting with a letter"
            }
            Const(_) => {
                "an uppercase alphanumeric (including underscore) string starting with a letter"
            }
            Type(_) => "an alphanumeric string starting with an uppercase letter",
            BigInt(_) => "a number",
            Hex(_) => "a hexadecimal string, starting with 0x",
            Period => ".",
            DoublePeriod => "..",
            Comma => "`,`",
            Colon => "`:`",
            DoubleColon => "`::`",
            LeftParen => "`(`",
            RightParen => "`)`",
            LeftBracket => "`[`",
            RightBracket => "`]`",
            LeftCurlyBracket => "`{`",
            RightCurlyBracket => "`}`",
            SemiColon => "`;`",
            Slash => "`/`",
            Comment(_) => "`//`",
            Greater => "`>`",
            Less => "`<`",
            Equal => "`=`",
            DoubleEqual => "`==`",
            Plus => "`+`",
            Minus => "`-`",
            RightArrow => "`->`",
            Star => "`*`",
            Ampersand => "`&`",
            Pipe => "`|`",
            Exclamation => "`!`",
            //            TokenType::Literal => "`\"something\"",
        };

        write!(f, "{}", desc)
    }
}

impl TokenKind {
    pub fn new_token(self, ctx: &mut LexerCtx, len: usize) -> Token {
        let token = Token {
            kind: self,
            span: Span(ctx.offset, len),
        };

        ctx.offset += len;

        token
    }
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    fn parse_line(ctx: &mut LexerCtx, line: &str) -> Result<Vec<Self>> {
        let mut tokens = vec![];

        // keep track of variables
        let mut ident_or_number: Option<String> = None;

        let add_thing =
            |ctx: &mut LexerCtx, tokens: &mut Vec<_>, ident_or_number: String| -> Result<()> {
                let len = ident_or_number.len();
                if let Some(keyword) = Keyword::parse(&ident_or_number) {
                    tokens.push(TokenKind::Keyword(keyword).new_token(ctx, len));
                } else {
                    let token_type = if is_numeric(&ident_or_number) {
                        TokenKind::BigInt(ident_or_number)
                    } else if is_hexadecimal(&ident_or_number) {
                        TokenKind::Hex(ident_or_number)
                    } else if is_identifier(&ident_or_number) {
                        TokenKind::Identifier(ident_or_number)
                    } else if is_type(&ident_or_number) {
                        TokenKind::Type(ident_or_number)
                    } else if is_const(&ident_or_number) {
                        TokenKind::Const(ident_or_number)
                    } else {
                        return Err(Error {
                            kind: ErrorKind::InvalidIdentifier(ident_or_number),
                            span: Span(ctx.offset, 1),
                        });
                    };
                    tokens.push(token_type.new_token(ctx, len));
                }
                Ok(())
            };

        // go through line char by char
        let mut chars = line.chars().peekable();

        loop {
            // get next char
            let c = if let Some(c) = chars.next() {
                c
            } else {
                // if no next char, don't forget to add the last ident_or_number we saw

                if let Some(ident_or_number) = ident_or_number {
                    add_thing(ctx, &mut tokens, ident_or_number)?;
                }
                break;
            };

            // where we in the middle of parsing an ident or number?
            if !c.is_alphanumeric() && c != '_' {
                if let Some(ident_or_number) = ident_or_number.take() {
                    add_thing(ctx, &mut tokens, ident_or_number)?;
                }
            }

            // other type of token
            match c {
                c if c.is_alphanumeric() || c == '_' => {
                    if let Some(ref mut ident_or_number) = &mut ident_or_number {
                        ident_or_number.push(c);
                    } else {
                        ident_or_number = Some(c.to_string());
                    }
                }
                '.' => {
                    let next_c = chars.peek();
                    if matches!(next_c, Some(&'.')) {
                        tokens.push(TokenKind::DoublePeriod.new_token(ctx, 2));
                        chars.next();
                    } else {
                        tokens.push(TokenKind::Period.new_token(ctx, 1));
                    }
                }
                ',' => {
                    tokens.push(TokenKind::Comma.new_token(ctx, 1));
                }
                ':' => {
                    // TODO: replace `peek` with `next_if_eq`?
                    let next_c = chars.peek();
                    if matches!(next_c, Some(&':')) {
                        tokens.push(TokenKind::DoubleColon.new_token(ctx, 2));
                        chars.next();
                    } else {
                        tokens.push(TokenKind::Colon.new_token(ctx, 1));
                    }
                }
                '(' => {
                    tokens.push(TokenKind::LeftParen.new_token(ctx, 1));
                }
                ')' => {
                    tokens.push(TokenKind::RightParen.new_token(ctx, 1));
                }
                '[' => {
                    tokens.push(TokenKind::LeftBracket.new_token(ctx, 1));
                }
                ']' => {
                    tokens.push(TokenKind::RightBracket.new_token(ctx, 1));
                }
                '{' => {
                    tokens.push(TokenKind::LeftCurlyBracket.new_token(ctx, 1));
                }
                '}' => {
                    tokens.push(TokenKind::RightCurlyBracket.new_token(ctx, 1));
                }
                ';' => {
                    tokens.push(TokenKind::SemiColon.new_token(ctx, 1));
                }
                '/' => {
                    let next_c = chars.peek();
                    if matches!(next_c, Some(&'/')) {
                        chars.next(); // ignore the second /

                        // TODO: why can't I call chars.as_str().to_string()
                        let comment = chars.collect::<String>();
                        let len = comment.len();
                        tokens.push(TokenKind::Comment(comment).new_token(ctx, 2 + len));
                        break;
                    } else {
                        tokens.push(TokenKind::Slash.new_token(ctx, 1));
                    }
                }
                '>' => {
                    tokens.push(TokenKind::Greater.new_token(ctx, 1));
                }
                '<' => {
                    tokens.push(TokenKind::Less.new_token(ctx, 1));
                }
                '=' => {
                    let next_c = chars.peek();
                    if matches!(next_c, Some(&'=')) {
                        tokens.push(TokenKind::DoubleEqual.new_token(ctx, 2));
                        chars.next();
                    } else {
                        tokens.push(TokenKind::Equal.new_token(ctx, 1));
                    }
                }
                '+' => {
                    tokens.push(TokenKind::Plus.new_token(ctx, 1));
                }
                '-' => {
                    let next_c = chars.peek();
                    if matches!(next_c, Some(&'>')) {
                        tokens.push(TokenKind::RightArrow.new_token(ctx, 2));
                        chars.next();
                    } else {
                        tokens.push(TokenKind::Minus.new_token(ctx, 1));
                    }
                }
                '*' => {
                    tokens.push(TokenKind::Star.new_token(ctx, 1));
                }
                '&' => {
                    tokens.push(TokenKind::Ampersand.new_token(ctx, 1));
                }
                '|' => {
                    tokens.push(TokenKind::Pipe.new_token(ctx, 1));
                }
                '!' => {
                    tokens.push(TokenKind::Exclamation.new_token(ctx, 1));
                }
                ' ' => ctx.offset += 1,
                _ => {
                    return Err(Error {
                        kind: ErrorKind::InvalidToken,
                        span: Span(ctx.offset, 1),
                    });
                }
            }
        }

        Ok(tokens)
    }

    pub fn parse(code: &str) -> Result<Tokens> {
        let mut ctx = LexerCtx::default();
        let mut tokens = vec![];

        for line in code.lines() {
            let line_tokens = Token::parse_line(&mut ctx, line)?;
            ctx.offset += 1; // newline
            tokens.extend(line_tokens);
        }

        Ok(Tokens::new(tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CODE: &str = r#"use crypto::poseidon;

fn main(public_input: [fel; 3], private_input: [fel; 3]) -> [fel; 8] {
    let digest = poseidon(private_input);
    assert(digest == public_input);
}
"#;

    #[test]
    fn test_lexer() {
        match Token::parse(CODE) {
            Ok(root) => {
                println!("{:#?}", root);
            }
            Err(e) => {
                println!("{:?}", e);
            }
        }
    }

    #[test]
    fn test_ident() {
        assert!(is_identifier("cond2"));
        assert!(is_type("Cond2"));
    }
}
