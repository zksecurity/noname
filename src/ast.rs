use crate::{
    error::Error,
    parser::{Root, AST},
    stdlib,
};

pub struct Compiler;

impl Compiler {
    pub fn compile(ast: AST) -> Result<(), ()> {
        for root in ast.0 {
            match root {
                // `use crypto::poseidon;`
                Root::Use(path) => {
                    let path = &mut path.0.into_iter();
                    let root_module = path.next().expect("empty imports can't be parsed");

                    if root_module == "std" {
                        stdlib::parse_std_import(path)?;
                    } else {
                        unimplemented!();
                    }
                }

                // `fn main() { ... }`
                Root::Function(function) => {
                    println!("fn {}() {{", function.name);
                    println!("}}");
                }

                // ignore comments
                Root::Comment(comment) => (),
            }
        }

        // TODO: where is function main?

        Ok(())
    }
}

struct Scope {
    pub variables: Vec<String>,
    pub functions: Vec<String>,
    pub types: Vec<String>,
}

impl Scope {
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            functions: Vec::new(),
            types: Vec::new(),
        }
    }
}
