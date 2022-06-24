use crate::{
    parser::{FunctionSig, Root, AST},
    stdlib,
};

pub struct Compiler;

impl Compiler {
    pub fn compile(ast: AST) -> Result<(), &'static str> {
        let mut scope = Scope::default();

        for root in ast.0 {
            match root {
                // `use crypto::poseidon;`
                Root::Use(path) => {
                    let path = &mut path.0.into_iter();
                    let root_module = path.next().expect("empty imports can't be parsed");

                    let (functions, types) = if root_module == "std" {
                        stdlib::parse_std_import(path)?
                    } else {
                        unimplemented!()
                    };

                    scope.functions.extend(functions);
                    scope.types.extend(types);
                }

                // `fn main() { ... }`
                Root::Function(function) => {
                    // TODO: support other functions
                    if function.name != "main" {
                        unimplemented!();
                    }
                }

                // ignore comments
                Root::Comment(_comment) => (),
            }
        }

        // TODO: where is function main?

        Ok(())
    }

    fn analyze_function() {}
}

#[derive(Default)]
struct Scope {
    pub variables: Vec<String>,
    pub functions: Vec<FunctionSig>,
    pub types: Vec<String>,
}

impl Scope {}
