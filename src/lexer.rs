use crate::error::{Error, ErrorTy};

#[derive(Default, Debug)]
pub struct LexerCtx {
    /// the offset of what we've read so far in the file
    offset: usize,
}

#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Clone)]
pub enum TokenType {
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

impl TokenType {
    pub fn new_token(self, ctx: &LexerCtx, len: usize) -> Token {
        Token {
            typ: self,
            span: (ctx.offset, len),
        }
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
        let mut thing: Option<String> = None;
        let add_thing = |ctx: &mut LexerCtx, tokens: &mut Vec<_>, thing: String| {
            ctx.offset += thing.len();
            if let Some(keyword) = Keyword::parse(&thing) {
                tokens.push(TokenType::Keyword(keyword).new_token(ctx, 1));
            } else {
                tokens.push(TokenType::AlphaNumeric_(thing).new_token(ctx, 1));
            }
        };

        // go through line char by char
        let mut chars = line.chars().peekable();

        loop {
            // get next char
            let c = if let Some(c) = chars.next() {
                c
            } else {
                // if no next char, don't forget to add the last thing we saw

                if let Some(thing) = thing {
                    add_thing(ctx, &mut tokens, thing);
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
                    add_thing(ctx, &mut tokens, thing);
                }
                (false, None) => (),
            }

            match c {
                ',' => {
                    tokens.push(TokenType::Comma.new_token(ctx, 1));
                    ctx.offset += 1;
                }
                ':' => {
                    let next_c = chars.peek();
                    if matches!(next_c, Some(&':')) {
                        tokens.push(TokenType::DoubleColon.new_token(ctx, 2));
                        chars.next();
                        ctx.offset += 2;
                    } else {
                        tokens.push(TokenType::Colon.new_token(ctx, 1));
                        ctx.offset += 1;
                    }
                }
                '(' => {
                    tokens.push(TokenType::OpenParen.new_token(ctx, 1));
                    ctx.offset += 1;
                }
                ')' => {
                    tokens.push(TokenType::CloseParen.new_token(ctx, 1));
                    ctx.offset += 1;
                }
                '[' => {
                    tokens.push(TokenType::OpenBracket.new_token(ctx, 1));
                    ctx.offset += 1;
                }
                ']' => {
                    tokens.push(TokenType::CloseBracket.new_token(ctx, 1));
                    ctx.offset += 1;
                }
                '{' => {
                    tokens.push(TokenType::OpenCurlyBracket.new_token(ctx, 1));
                    ctx.offset += 1;
                }
                '}' => {
                    tokens.push(TokenType::CloseCurlyBracket.new_token(ctx, 1));
                    ctx.offset += 1;
                }
                ';' => {
                    tokens.push(TokenType::SemiColon.new_token(ctx, 1));
                    ctx.offset += 1;
                }
                '/' => {
                    let next_c = chars.peek();
                    if matches!(next_c, Some(&'/')) {
                        // TODO: why can't I call chars.as_str().to_string()
                        let comment = chars.collect::<String>();
                        ctx.offset += comment.len();
                        tokens.push(TokenType::Comment(comment).new_token(ctx, 2));
                        break;
                    } else {
                        tokens.push(TokenType::Division.new_token(ctx, 1));
                        ctx.offset += 1;
                    }
                }
                '>' => {
                    tokens.push(TokenType::Greater.new_token(ctx, 1));
                    ctx.offset += 1;
                }
                '<' => {
                    tokens.push(TokenType::Less.new_token(ctx, 1));
                    ctx.offset += 1;
                }
                '=' => {
                    let next_c = chars.peek();
                    if matches!(next_c, Some(&'=')) {
                        tokens.push(TokenType::Equal.new_token(ctx, 1));
                        chars.next();
                    } else {
                        tokens.push(TokenType::Assign.new_token(ctx, 1));
                    }
                }
                '+' => {
                    tokens.push(TokenType::Plus.new_token(ctx, 1));
                    ctx.offset += 1;
                }
                '-' => {
                    tokens.push(TokenType::Minus.new_token(ctx, 1));
                    ctx.offset += 1
                }
                '*' => {
                    tokens.push(TokenType::Mul.new_token(ctx, 1));
                    ctx.offset += 1;
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
