use std::fmt::Display;

use crate::error::{Error, ErrorTy};

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
    Return,
    Assert,
}

impl Keyword {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "use" => Some(Self::Use),
            "fn" => Some(Self::Fn),
            "let" => Some(Self::Let),
            "pub" => Some(Self::Pub),
            "return" => Some(Self::Return),
            "assert" => Some(Self::Assert),
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
            Self::Assert => "assert",
        };

        write!(f, "{}", desc)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenType {
    Keyword(Keyword),   // reserved keywords
    Identifier(String), // [a-z_](a-z0-9_)*
    BigInt(String),     // (0-9)*
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
    Assign,             // =
    Equal,              // ==
    Plus,               // +
    Minus,              // -
    RightArrow,         // ->
    Star,               // *
                        //    Literal,               // "thing"
}

impl Display for TokenType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let desc = match self {
            TokenType::Keyword(_) => "keyword (use, let, etc.)",
            TokenType::Identifier(_) => {
                "a lowercase alphanumeric (including underscore) string starting with a letter"
            }
            TokenType::BigInt(_) => "a number",
            TokenType::Comma => "`,`",
            TokenType::Colon => "`:`",
            TokenType::DoubleColon => "`::`",
            TokenType::LeftParen => "`(`",
            TokenType::RightParen => "`)`",
            TokenType::LeftBracket => "`[`",
            TokenType::RightBracket => "`]`",
            TokenType::LeftCurlyBracket => "`{`",
            TokenType::RightCurlyBracket => "`}`",
            TokenType::SemiColon => "`;`",
            TokenType::Slash => "`/`",
            TokenType::Comment(_) => "`//`",
            TokenType::Greater => "`>`",
            TokenType::Less => "`<`",
            TokenType::Assign => "`=`",
            TokenType::Equal => "`==`",
            TokenType::Plus => "`+`",
            TokenType::Minus => "`-`",
            TokenType::RightArrow => "`->`",
            TokenType::Star => "`*`",
            //            TokenType::Literal => "`\"something\"",
        };

        write!(f, "{}", desc)
    }
}

impl TokenType {
    pub fn new_token(self, ctx: &mut LexerCtx, len: usize) -> Token {
        let token = Token {
            typ: self,
            span: (ctx.offset, len),
        };

        ctx.offset += len;

        token
    }
}

#[derive(Debug, Clone)]
pub struct Token {
    pub typ: TokenType,
    pub span: (usize, usize),
}

impl Token {
    fn parse_line(ctx: &mut LexerCtx, line: &str) -> Result<Vec<Self>, Error> {
        let mut tokens = vec![];

        // keep track of variables
        let mut ident_or_number: Option<String> = None;

        let add_thing = |ctx: &mut LexerCtx,
                         tokens: &mut Vec<_>,
                         ident_or_number: String|
         -> Result<(), Error> {
            let len = ident_or_number.len();
            if let Some(keyword) = Keyword::parse(&ident_or_number) {
                tokens.push(TokenType::Keyword(keyword).new_token(ctx, len));
            } else {
                // integer?
                if ident_or_number.chars().all(|c| c.is_digit(10)) {
                    tokens.push(TokenType::BigInt(ident_or_number).new_token(ctx, len));
                } else {
                    // valid identifier?
                    if !ident_or_number.chars().next().unwrap().is_alphabetic()
                        || !ident_or_number
                            .chars()
                            .all(|c| (c.is_alphanumeric() && c.is_lowercase()) || c == '_')
                    {
                        return Err(Error {
                            error: ErrorTy::InvalidIdentifier,
                            span: (ctx.offset, 1),
                        });
                    }

                    tokens.push(TokenType::Identifier(ident_or_number).new_token(ctx, len));
                }
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
                ',' => {
                    tokens.push(TokenType::Comma.new_token(ctx, 1));
                }
                ':' => {
                    // TODO: replace `peek` with `next_if_eq`?
                    let next_c = chars.peek();
                    if matches!(next_c, Some(&':')) {
                        tokens.push(TokenType::DoubleColon.new_token(ctx, 2));
                        chars.next();
                    } else {
                        tokens.push(TokenType::Colon.new_token(ctx, 1));
                    }
                }
                '(' => {
                    tokens.push(TokenType::LeftParen.new_token(ctx, 1));
                }
                ')' => {
                    tokens.push(TokenType::RightParen.new_token(ctx, 1));
                }
                '[' => {
                    tokens.push(TokenType::LeftBracket.new_token(ctx, 1));
                }
                ']' => {
                    tokens.push(TokenType::RightBracket.new_token(ctx, 1));
                }
                '{' => {
                    tokens.push(TokenType::LeftCurlyBracket.new_token(ctx, 1));
                }
                '}' => {
                    tokens.push(TokenType::RightCurlyBracket.new_token(ctx, 1));
                }
                ';' => {
                    tokens.push(TokenType::SemiColon.new_token(ctx, 1));
                }
                '/' => {
                    let next_c = chars.peek();
                    if matches!(next_c, Some(&'/')) {
                        // TODO: why can't I call chars.as_str().to_string()
                        let comment = chars.collect::<String>();
                        let len = comment.len();
                        tokens.push(TokenType::Comment(comment).new_token(ctx, 2 + len));
                        break;
                    } else {
                        tokens.push(TokenType::Slash.new_token(ctx, 1));
                    }
                }
                '>' => {
                    tokens.push(TokenType::Greater.new_token(ctx, 1));
                }
                '<' => {
                    tokens.push(TokenType::Less.new_token(ctx, 1));
                }
                '=' => {
                    let next_c = chars.peek();
                    if matches!(next_c, Some(&'=')) {
                        tokens.push(TokenType::Equal.new_token(ctx, 2));
                        chars.next();
                    } else {
                        tokens.push(TokenType::Assign.new_token(ctx, 1));
                    }
                }
                '+' => {
                    tokens.push(TokenType::Plus.new_token(ctx, 1));
                }
                '-' => {
                    let next_c = chars.peek();
                    if matches!(next_c, Some(&'>')) {
                        tokens.push(TokenType::RightArrow.new_token(ctx, 2));
                        chars.next();
                    } else {
                        tokens.push(TokenType::Minus.new_token(ctx, 1));
                    }
                }
                '*' => {
                    tokens.push(TokenType::Star.new_token(ctx, 1));
                }
                ' ' => ctx.offset += 1,
                _ => {
                    return Err(Error {
                        error: ErrorTy::InvalidToken,
                        span: (ctx.offset, 1),
                    });
                }
            }
        }

        Ok(tokens)
    }

    pub fn parse(code: &str) -> Result<Vec<Self>, Error> {
        let mut ctx = LexerCtx::default();
        let mut tokens = vec![];

        for line in code.lines() {
            let line_tokens = Token::parse_line(&mut ctx, line)?;
            ctx.offset += 1; // newline
            tokens.extend(line_tokens);
        }

        Ok(tokens)
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
}
