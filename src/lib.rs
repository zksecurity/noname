//! noname project

pub mod asm;
pub mod ast;
pub mod constants;
pub mod error;
pub mod field;
pub mod lexer;
pub mod parser;
pub mod stdlib;
pub mod tokens;
pub mod type_checker;

#[cfg(test)]
pub mod tests;
