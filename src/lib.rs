//! noname project

use miette::{Diagnostic, Result, SourceSpan};
use thiserror::Error;

use error::{Error, ErrorTy};

pub mod error;

#[derive(Default, Debug)]
pub struct Context {
    offset: usize,
    inline_offset: usize,
}

#[derive(Debug)]
pub struct Path(Vec<String>);

impl Path {
    /// Parses a path from a list of tokens.
    pub fn parse_path(
        ctx: &Context,
        tokens: &mut impl Iterator<Item = Token>,
    ) -> Result<Self, Error> {
        let mut path = vec![];

        loop {
            let module_or_leaf = tokens.next();

            match module_or_leaf {
                // a chunk of the path
                Some(Token::AlphaNumeric_(chunk)) => {
                    if !is_valid_module(&chunk) {
                        return Err(Error {
                            error: ErrorTy::InvalidModule,
                            span: (ctx.offset + ctx.inline_offset, 1),
                        });
                    }

                    path.push(chunk.to_string());

                    // next, we expect a `::` to continue the path,
                    // or a separator, for example, `;` or `)`
                    match tokens.next() {
                        None => {
                            return Err(Error {
                                error: ErrorTy::InvalidEndOfLine,
                                span: (ctx.offset + ctx.inline_offset, 1),
                            })
                        }
                        Some(Token::DoubleColon) => continue,
                        Some(Token::SemiColon) => break,
                        _ => {
                            return Err(Error {
                                error: ErrorTy::InvalidToken,
                                span: (ctx.offset + ctx.inline_offset, 1),
                            })
                        }
                    }
                }
                _ => {
                    return Err(Error {
                        error: ErrorTy::InvalidPath,
                        span: (ctx.offset + ctx.inline_offset, 1),
                    });
                }
            }
        }

        Ok(Self(path))
    }
}

#[derive(Debug)]
pub enum Ty {
    Field,
    Array(Box<Self>, usize),
    Bool,
}

#[derive(Debug)]
pub enum ComparisonOp {
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Equal,
}

#[derive(Debug)]
pub enum Expression {
    Literal(String),
    Function(Function),
    Variable(String),
    Comparison(ComparisonOp, Box<Expression>, Box<Expression>),
    True,
    False,
}

#[derive(Debug)]
pub struct Function {
    pub name: String,
    pub arguments: Vec<(String, Ty)>,
    pub return_type: Option<Ty>,
    pub body: Vec<Statement>,
}

impl Function {
    pub fn parse_fn(
        ctx: &Context,
        tokens: &mut impl Iterator<Item = Token>,
    ) -> Result<Self, Error> {
        // parse function name
        let name = tokens.next().ok_or(Error {
            error: ErrorTy::InvalidFunctionSignature,
            span: (ctx.offset + ctx.inline_offset, 1),
        })?;

        // it's possible that there are no spaces: main(first_arg:type,etc.)
        // or a bunch of spaces: main ( first_arg : type , etc. )
        // should we enforce a single style?

        let func = Self {
            name: todo!(),
            arguments: vec![],
            return_type: todo!(),
            body: todo!(),
        };
        let name = tokens.next().ok_or(Error {
            error: ErrorTy::InvalidModule,
            span: (ctx.offset + ctx.inline_offset, 1),
        })?;

        Ok(func)
    }
}

#[derive(Debug)]
pub enum Statement {
    Assign { lhs: String, rhs: Expression },
    Assert(Expression),
    Return(Expression),
    Comment(String),
}

#[derive(Debug)]
pub enum Root {
    Use(Path),
    Function(Function),
    Comment(String),
}

pub fn parse(code: &'static str) -> Result<Vec<Root>, Error> {
    let mut root = vec![];
    let mut ctx = Context::default();

    for line in code.lines() {
        parse_line(&mut ctx, &mut root, line)?;
        println!("yo");
        ctx.offset += line.len();
        ctx.inline_offset = 0;
        dbg!(&ctx);
    }

    Ok(root)
}

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
pub enum Token {
    Keyword(Keyword),
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
    Comment,               // //
    Greater,               // >
    Less,                  // <
    Assign,                // =
    Equal,                 // ==
    Plus,                  // +
    Minus,                 // -
    Mul,                   // *
}

impl Token {
    /// No whitespace expected from the argument. This function is called by [Token::parse] internally.
    fn parse_(ctx: &mut Context, s: &str) -> Result<Vec<Self>, Error> {
        let mut tokens = vec![];
        let mut thing = None;

        let mut chars = s.chars().peekable();
        loop {
            let c = if let Some(c) = chars.next() { c } else { break };

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
                    if let Some(keyword) = Keyword::parse(&thing) {
                        tokens.push(Token::Keyword(keyword));
                    } else {
                        tokens.push(Token::AlphaNumeric_(thing));
                    }
                }
                (false, None) => (),
            }

            match c {
                ',' => tokens.push(Token::Comma),
                ':' => {
                    let next_c = chars.peek();
                    if matches!(next_c, Some(&':')) {
                        tokens.push(Token::Colon);
                        chars.next();
                    } else {
                        tokens.push(Token::DoubleColon)
                    }
                }
                '(' => tokens.push(Token::OpenParen),
                ')' => tokens.push(Token::CloseParen),
                '[' => tokens.push(Token::OpenBracket),
                ']' => tokens.push(Token::CloseBracket),
                '{' => tokens.push(Token::OpenCurlyBracket),
                '}' => tokens.push(Token::CloseCurlyBracket),
                ';' => tokens.push(Token::SemiColon),
                '/' => {
                    let next_c = chars.peek();
                    if matches!(next_c, Some(&'/')) {
                        tokens.push(Token::Comment);
                        chars.next();
                    } else {
                        tokens.push(Token::Division);
                    }
                }
                '>' => tokens.push(Token::Greater),
                '<' => tokens.push(Token::Less),
                '=' => {
                    let next_c = chars.peek();
                    if matches!(next_c, Some(&'=')) {
                        tokens.push(Token::Equal);
                        chars.next();
                    } else {
                        tokens.push(Token::Assign);
                    }
                }
                '+' => tokens.push(Token::Plus),
                '-' => tokens.push(Token::Minus),
                '*' => tokens.push(Token::Mul),
                _ => {
                    return Err(Error {
                        error: ErrorTy::InvalidToken,
                        span: (ctx.offset + ctx.inline_offset, 1),
                    })
                }
            }
        }

        Ok(tokens)
    }

    pub fn parse(ctx: &mut Context, line: &str) -> Result<Vec<Self>, Error> {
        let line = line.trim();
        let blocks = line.split_whitespace();

        let mut tokens = vec![];
        for block in blocks {
            tokens.extend(Self::parse_(ctx, block)?);
        }

        Ok(tokens)
    }
}

pub fn parse_line(
    ctx: &mut Context,
    root: &mut Vec<Root>,
    line: &'static str,
) -> Result<(), Error> {
    // parse tokens as vec
    let mut tokens = Token::parse(ctx, line)?.into_iter().peekable();

    // get first token
    let token = if let Some(token) = tokens.next() {
        token
    } else {
        return Ok(()); // empty line
    };

    // match special keywords
    match token {
        Token::Keyword(Keyword::Use) => {
            let path = Path::parse_path(ctx, &mut tokens)?;
            root.push(Root::Use(path));

            // end of line
            if !matches!(tokens.next(), Some(Token::SemiColon)) {
                return Err(Error {
                    error: ErrorTy::InvalidEndOfLine,
                    span: (ctx.offset + ctx.inline_offset, 1),
                });
            }
        }
        Token::Keyword(Keyword::Fn) => {
            let func = Function::parse_fn(ctx, &mut tokens)?;
            root.push(Root::Function(func));
        }
        Token::Comment => {
            let comment = if line.len() < 3 { "" } else { &line[3..] };
            root.push(Root::Comment(comment.to_string()));
        }
        _ => {
            return Err(Error {
                error: ErrorTy::InvalidToken,
                span: (ctx.offset + ctx.inline_offset, 1),
            })
        }
    }

    // we should have parsed everything in the line
    if tokens.len() > 0 {
        return Err(Error {
            error: ErrorTy::InvalidEndOfLine,
            span: (ctx.offset + ctx.inline_offset, 1),
        });
    }

    Ok(())
}

pub fn is_valid_module(module: &str) -> bool {
    let mut chars = module.chars();
    !module.is_empty()
        && chars.next().unwrap().is_ascii_alphabetic()
        && chars.all(|c| c.is_alphanumeric() || c == '_')
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
    fn test_parse() {
        match parse(CODE) {
            Ok(root) => {
                println!("{:#?}", root);
            }
            Err(e) => {
                println!("{:?}", e);
            }
        }
    }
}
