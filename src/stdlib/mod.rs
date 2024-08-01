use crate::{
    backends::Backend,
    circuit_writer::{CircuitWriter, VarInfo},
    constants::Span,
    error::Result,
    imports::FnKind,
    lexer::Token,
    parser::{
        types::{FnSig, GenericParameters},
        ParserCtx,
    },
    type_checker::FnInfo,
    var::Var,
};

pub mod bits;
pub mod builtins;
pub mod crypto;

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
                span: Span::default(),
            });
        }
        res
    }
}
