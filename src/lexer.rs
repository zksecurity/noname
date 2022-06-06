use miette::{Diagnostic, Result, SourceSpan};

use crate::error::{Error, ErrorTy};

#[derive(Debug)]
pub enum Keyword {
    Use,
    Fn,
}

impl Keyword {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "use" => Some(Self::Use),
            "fn" => Some(Self::Fn),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum TokenType {
    Whitespace(usize),     // required to count offset
    Keyword(Keyword),      // reserved keywords
    AlphaNumeric_(String), // (a-zA_Z0-9_)*
    Comma,                 // ,
    Colon,                 // :
    DoubleColon,           // ::
    OpenParen,             // (
    CloseParen,            // )
    OpenBracket,           // [
    CloseBracket,          // ]
    OpenCurlyBracket,      // {
    CloseCurlyBracket,     // }
    SemiColon,             // ;
    Division,              // /
    Comment(String),       // //
    Greater,               // >
    Less,                  // <
    Assign,                // =
    Equal,                 // ==
    Plus,                  // +
    Minus,                 // -
    Mul,                   // *
}

pub struct Token {
    pub typ: TokenType,
    pub span: (usize, usize),
}

impl Token {
    /// No whitespace expected from the argument. This function is called by [TokenType::parse] internally.
    fn parse_(ctx: &mut Context, s: &str) -> Result<Vec<Self>, Error> {
        let mut tokens = vec![];

        let mut thing: Option<String> = None;
        let add_thing = |tokens: &mut Vec<_>, thing: String| {
            if let Some(keyword) = Keyword::parse(&thing) {
                tokens.push(TokenType::Keyword(keyword));
            } else {
                tokens.push(TokenType::AlphaNumeric_(thing));
            }
        };

        // go through line char by char
        let mut chars = s.chars().peekable();

        loop {
            // get next char
            let c = if let Some(c) = chars.next() {
                c
            } else {
                // if no next char, don't forget to add the last thing we saw

                if let Some(thing) = thing {
                    add_thing(&mut tokens, thing);
                }
                break;
            };

            let is_alphanumeric = c.is_alphanumeric() || c == '_';
            match (is_alphanumeric, &mut thing) {
                (true, None) => {
                    thing = Some(c.to_string());
                    continue;
                }
                (true, Some(ref mut thing)) => {
                    thing.push(c);
                    continue;
                }
                (false, Some(_)) => {
                    let thing = thing.take().unwrap();
                    add_thing(&mut tokens, thing);
                }
                (false, None) => (),
            }

            match c {
                ',' => tokens.push(TokenType::Comma),
                ':' => {
                    let next_c = chars.peek();
                    if matches!(next_c, Some(&':')) {
                        tokens.push(TokenType::DoubleColon);
                        chars.next();
                    } else {
                        tokens.push(TokenType::Colon)
                    }
                }
                '(' => tokens.push(TokenType::OpenParen),
                ')' => tokens.push(TokenType::CloseParen),
                '[' => tokens.push(TokenType::OpenBracket),
                ']' => tokens.push(TokenType::CloseBracket),
                '{' => tokens.push(TokenType::OpenCurlyBracket),
                '}' => tokens.push(TokenType::CloseCurlyBracket),
                ';' => tokens.push(TokenType::SemiColon),
                '/' => {
                    let next_c = chars.peek();
                    if matches!(next_c, Some(&'/')) {
                        // TODO: why can't I call chars.as_str().to_string()
                        let comment = chars.collect::<Vec<_>>().join("");
                        tokens.push(TokenType::Comment);
                        chars.next();
                    } else {
                        tokens.push(TokenType::Division);
                    }
                }
                '>' => tokens.push(TokenType::Greater),
                '<' => tokens.push(TokenType::Less),
                '=' => {
                    let next_c = chars.peek();
                    if matches!(next_c, Some(&'=')) {
                        tokens.push(TokenType::Equal);
                        chars.next();
                    } else {
                        tokens.push(TokenType::Assign);
                    }
                }
                '+' => tokens.push(TokenType::Plus),
                '-' => tokens.push(TokenType::Minus),
                '*' => tokens.push(TokenType::Mul),
                _ => {
                    return Err(Error {
                        error: ErrorTy::InvalidToken,
                        span: (ctx.offset + ctx.inline_offset, 1),
                    });
                }
            }
        }

        Ok(tokens)
    }

    pub fn parse_line(ctx: &mut Context, line: &str) -> Result<Vec<Self>, Error> {
        let line = line.trim();
        let blocks = line.split_whitespace();

        let mut tokens = vec![];
        for block in blocks {
            let more = Self::parse_(ctx, block)?;
            tokens.extend(more);
        }

        Ok(tokens)
    }
}
