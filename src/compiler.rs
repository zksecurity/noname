//! This module is a wrapper API around noname.
//! It is important that user-facing features use the functions here,
//! as they attach the correct filename and source code to errors.
//! It does that by transforming our [Error] type into a [`miette::Error`] type for all functions here.
//! (via the [`IntoMiette`] trait that we define here.)

use std::collections::HashMap;

use miette::NamedSource;

use crate::{
    backends::Backend, circuit_writer::CircuitWriter, cli::packages::UserRepo, error::Result,
    inputs::JsonInputs, lexer::Token, name_resolution::NAST, parser::AST,
    type_checker::TypeChecker, witness::CompiledCircuit,
};

/// Contains the association between a counter and the corresponding filename and source code.
#[derive(Debug)]
pub struct Sources {
    /// A counter representing the last inserted source.
    id: usize,

    /// Maps a filename id to its filename and source code.
    pub map: HashMap<usize, (String, String)>,
}

impl Default for Sources {
    fn default() -> Self {
        Self::new()
    }
}

impl Sources {
    #[must_use]
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

    #[must_use]
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
                Err(report.with_source_code(NamedSource::new(filename, source)))
            }
        }
    }
}

pub fn typecheck_next_file<B: Backend>(
    typechecker: &mut TypeChecker<B>,
    this_module: Option<UserRepo>,
    sources: &mut Sources,
    filename: String,
    code: String,
    node_id: usize,
) -> miette::Result<usize> {
    typecheck_next_file_inner(typechecker, this_module, sources, filename, code, node_id)
        .into_miette(sources)
}

/// This should not be used directly. Check [`get_tast`] instead.
pub fn typecheck_next_file_inner<B: Backend>(
    typechecker: &mut TypeChecker<B>,
    this_module: Option<UserRepo>,
    sources: &mut Sources,
    filename: String,
    code: String,
    node_id: usize,
) -> Result<usize> {
    let is_lib = this_module.is_some();

    // parsing to name resolution
    let (nast, new_node_id) = get_nast(this_module, sources, filename, code, node_id)?;

    // type checker
    typechecker.analyze(nast, is_lib)?;

    Ok(new_node_id)
}

pub fn get_nast<B: Backend>(
    this_module: Option<UserRepo>,
    sources: &mut Sources,
    filename: String,
    code: String,
    node_id: usize,
) -> Result<(NAST<B>, usize)> {
    // save filename and source code
    let filename_id = sources.add(filename, code);
    let code = &sources.map[&filename_id].1;

    // lexer
    let tokens = Token::parse(filename_id, code)?;
    if std::env::var("NONAME_VERBOSE").is_ok() {
        println!("lexer succeeded");
    }

    // parser
    let (ast, new_node_id) = AST::parse(filename_id, tokens, node_id)?;
    if std::env::var("NONAME_VERBOSE").is_ok() {
        println!("parser succeeded");
    }

    // name resolution
    let nast = NAST::resolve_modules(this_module, ast)?;
    if std::env::var("NONAME_VERBOSE").is_ok() {
        println!("name resolution succeeded");
    }

    Ok((nast, new_node_id))
}

pub fn compile<B: Backend>(
    sources: &Sources,
    tast: TypeChecker<B>,
    backend: B,
) -> miette::Result<CompiledCircuit<B>> {
    CircuitWriter::generate_circuit(tast, backend).into_miette(sources)
}

pub fn generate_witness<B: Backend>(
    compiled_circuit: &CompiledCircuit<B>,
    sources: &Sources,
    public_inputs: JsonInputs,
    private_inputs: JsonInputs,
) -> miette::Result<B::GeneratedWitness> {
    compiled_circuit
        .generate_witness(public_inputs, private_inputs)
        .into_miette(sources)
}
