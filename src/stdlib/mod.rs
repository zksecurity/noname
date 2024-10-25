use crate::{
    backends::Backend,
    circuit_writer::{CircuitWriter, VarInfo},
    cli::packages::UserRepo,
    compiler::{typecheck_next_file, Sources},
    constants::Span,
    error::Result,
    imports::FnKind,
    lexer::Token,
    parser::{
        types::{FnSig, GenericParameters},
        ParserCtx,
    },
    type_checker::{FnInfo, TypeChecker},
    var::Var,
};
use std::path::Path;

pub mod bits;
pub mod builtins;
pub mod crypto;

/// The directory under [NONAME_DIRECTORY] containing the native stdlib.
pub const STDLIB_DIRECTORY: &str = "src/stdlib/native/";

pub enum AllStdModules {
    Builtins,
    Crypto,
    Bits,
}

impl AllStdModules {
    pub fn all_std_modules() -> Vec<AllStdModules> {
        vec![
            AllStdModules::Builtins,
            AllStdModules::Crypto,
            AllStdModules::Bits,
        ]
    }

    pub fn get_parsed_fns<B: Backend>(&self) -> Vec<FnInfo<B>> {
        match self {
            AllStdModules::Builtins => builtins::BuiltinsLib::get_parsed_fns(),
            AllStdModules::Crypto => crypto::CryptoLib::get_parsed_fns(),
            AllStdModules::Bits => bits::BitsLib::get_parsed_fns(),
        }
    }

    pub fn get_name(&self) -> &'static str {
        match self {
            AllStdModules::Builtins => builtins::BuiltinsLib::MODULE,
            AllStdModules::Crypto => crypto::CryptoLib::MODULE,
            AllStdModules::Bits => bits::BitsLib::MODULE,
        }
    }
}

type FnInfoType<B: Backend> = fn(
    &mut CircuitWriter<B>,
    &GenericParameters,
    &[VarInfo<B::Field, B::Var>],
    Span,
) -> Result<Option<Var<B::Field, B::Var>>>;

trait Module {
    /// e.g. "crypto"
    const MODULE: &'static str;

    fn get_fns<B: Backend>() -> Vec<(&'static str, FnInfoType<B>)>;

    fn get_parsed_fns<B: Backend>() -> Vec<FnInfo<B>> {
        let fns = Self::get_fns();
        let mut res = Vec::with_capacity(fns.len());
        for (code, fn_handle) in fns {
            let ctx = &mut ParserCtx::default();
            // TODO: we should try to point to real noname files here (not 0)
            let mut tokens = Token::parse(0, code).unwrap();
            let sig = FnSig::parse(ctx, &mut tokens).unwrap();
            res.push(FnInfo {
                kind: FnKind::BuiltIn(sig, fn_handle),
                is_hint: false,
                span: Span::default(),
            });
        }
        res
    }
}

pub fn init_stdlib_dep<B: Backend>(
    sources: &mut Sources,
    tast: &mut TypeChecker<B>,
    node_id: usize,
    path_prefix: &str,
) -> usize {
    // list the stdlib dependency in order
    let libs = vec!["bits", "comparator", "multiplexer", "int"];

    let mut node_id = node_id;

    for lib in libs {
        let module = UserRepo::new(&format!("std/{}", lib));
        let prefix_stdlib = Path::new(path_prefix);
        let code = std::fs::read_to_string(prefix_stdlib.join(format!("{lib}/lib.no"))).unwrap();
        node_id =
            typecheck_next_file(tast, Some(module), sources, lib.to_string(), code, 0).unwrap();
    }

    node_id
}
