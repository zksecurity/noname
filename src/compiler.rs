//! This module is a wrapper API around noname.
//! It is important that user-facing features use the functions here,
//! as they attach the correct filename and source code to errors.
//! It does that by transforming our [Error] type into a [miette::Error] type for all functions here.
//! (via the [IntoMiette] trait that we define here.)

use std::collections::HashMap;

use miette::{IntoDiagnostic, NamedSource};

use crate::{
    circuit_writer::CircuitWriter,
    constants::Field,
    error::Result,
    inputs::JsonInputs,
    lexer::Token,
    parser::AST,
    type_checker::{Dependencies, TypeChecker},
    witness::{CompiledCircuit, Witness},
};

/// Contains the association between a counter and the corresponding filename and source code.
#[derive(Debug)]
pub struct Sources {
    /// A counter representing the last inserted source.
    id: usize,

    /// Maps a filename id to its filename and source code.
    pub map: HashMap<usize, (String, String)>,
}

impl Sources {
    pub fn new() -> Self {
        let mut map = HashMap::new();
        map.insert(
            0,
            ("<BUILTIN>".to_string(), "<SEE NONAME CODE>".to_string()),
        );
        Self { id: 0, map }
    }

    pub fn add(&mut self, filename: String, source: String) -> usize {
        self.id += 1;
        self.map.insert(self.id, (filename, source));
        self.id
    }

    pub fn get(&self, id: &usize) -> Option<&(String, String)> {
        self.map.get(id)
    }
}

//
// Wrapper functions that can be used to easily compile or get the TAST of a file.
// Note that the error does not contain the source code yet
// (which [miette] requires for nice errors).
//

pub trait IntoMiette<T> {
    fn into_miette(self, sources: &Sources) -> miette::Result<T>;
}

impl<T> IntoMiette<T> for Result<T> {
    fn into_miette(self, sources: &Sources) -> miette::Result<T> {
        match self {
            Ok(res) => Ok(res),
            Err(err) => {
                let filename_id = err.span.filename_id;
                let (filename, source) = sources
                    .get(&filename_id)
                    .expect("couldn't find source")
                    .clone();
                let report: miette::Report = err.into();
                return Err(report.with_source_code(NamedSource::new(filename, source)));
            }
        }
    }
}

pub fn get_tast(
    sources: &mut Sources,
    filename: String,
    code: String,
    deps: &Dependencies,
) -> miette::Result<TypeChecker> {
    get_tast_inner(sources, filename, code, deps).into_miette(sources)
}

/// This should not be used directly. Check [get_tast] instead.
pub fn get_tast_inner(
    sources: &mut Sources,
    filename: String,
    code: String,
    deps: &Dependencies,
) -> Result<TypeChecker> {
    // save filename and source code
    let filename_id = sources.add(filename, code);
    let code = &sources.map[&filename_id].1;

    // parse
    let tokens = Token::parse(filename_id, &code)?;
    let ast = AST::parse(filename_id, tokens)?;

    // get AST
    TypeChecker::analyze(ast, deps)
}

pub fn compile(
    sources: &Sources,
    tast: TypeChecker,
    deps: Dependencies,
) -> miette::Result<CompiledCircuit> {
    CircuitWriter::generate_circuit(tast, deps).into_miette(sources)
}

pub fn generate_witness(
    compiled_circuit: &CompiledCircuit,
    sources: &Sources,
    public_inputs: JsonInputs,
    private_inputs: JsonInputs,
) -> miette::Result<(Witness, Vec<Field>, Vec<Field>)> {
    compiled_circuit
        .generate_witness(public_inputs, private_inputs)
        .into_miette(sources)
}
