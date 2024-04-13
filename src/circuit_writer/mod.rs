use std::collections::{HashMap, HashSet};

use crate::{
    backends::{kimchi::KimchiVesta, r1cs::R1CS, Backend}, compiler::{GeneratedWitness, Sources}, constants::Span, error::{Error, ErrorKind, Result}, parser::{
        types::{AttributeKind, FnArg, TyKind},
        Expr,
    }, type_checker::{ConstInfo, FnInfo, FullyQualified, StructInfo, TypeChecker}, var::{CellVar, Value, Var}, witness::{CompiledCircuit, WitnessEnv}
};

pub use fn_env::{FnEnv, VarInfo};
use serde::{Deserialize, Serialize};
//use serde::{Deserialize, Serialize};
pub use writer::{Gate, GateKind, Wiring};

pub mod fn_env;
pub mod writer;

// enums for proving backends
#[derive(Debug)]
pub enum BackendKind {
    Kimchi(KimchiVesta),
    R1CS(R1CS),
}

//#[derive(Debug, Serialize, Deserialize)]
#[derive(Debug)]
pub struct CircuitWriter<B>
where
    B: Backend,
{
    /// The type checker state for the main module.
    // Important: this field must not be used directly.
    // This is because, depending on the value of [current_module],
    // the type checker state might be this one, or one of the ones in [dependencies].
    typed: TypeChecker<B>,

    /// The constraint backend for the circuit.
    /// For now, this needs to be exposed for the kimchi prover for kimchi specific low level data.
    /// So we might make this private if the prover facilities can be deprecated.
    pub backend: B,

    /// Once this is set, you can generate a witness (and can't modify the circuit?)
    // Note: I don't think we need this, but it acts as a nice redundant failsafe.
    pub(crate) finalized: bool,

    /// Size of the public input.
    pub(crate) public_input_size: usize,

    /// If a public output is set, this will be used to store its [Var].
    /// The public output generation works as follows:
    /// 1. This cvar is created and inserted in the circuit (gates) during compilation of the public input
    ///    (as the public output is the end of the public input)
    /// 2. When the `return` statement of the circuit is parsed,
    ///    it will set this `public_output` variable again to the correct vars.
    /// 3. During witness generation, the public output computation
    ///    is delayed until the very end.
    pub(crate) public_output: Option<Var<B::Field>>,

    /// Indexes used by the private inputs
    /// (this is useful to check that they appear in the circuit)
    pub(crate) private_input_indices: Vec<usize>,
}

/// Debug information related to a single row in a circuit.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DebugInfo {
    /// The place in the original source code that created that gate.
    pub span: Span,

    /// A note on why this was added
    pub note: String,
}

impl<B: Backend> CircuitWriter<B> {
    pub fn expr_type(&self, expr: &Expr) -> Option<&TyKind> {
        self.typed.expr_type(expr)
    }

    // TODO: can we get rid of this?
    pub fn node_type(&self, node_id: usize) -> Option<&TyKind> {
        self.typed.node_type(node_id)
    }

    pub fn struct_info(&self, qualified: &FullyQualified) -> Option<&StructInfo> {
        self.typed.struct_info(qualified)
    }

    pub fn fn_info(&self, qualified: &FullyQualified) -> Option<&FnInfo<B>> {
        self.typed.fn_info(qualified)
    }

    pub fn const_info(&self, qualified: &FullyQualified) -> Option<&ConstInfo<B::Field>> {
        self.typed.const_info(qualified)
    }

    pub fn size_of(&self, typ: &TyKind) -> usize {
        self.typed.size_of(typ)
    }

    pub fn add_local_var(
        &self,
        fn_env: &mut FnEnv<B::Field>,
        var_name: String,
        var_info: VarInfo<B::Field>,
    ) {
        // check for consts first
        let qualified = FullyQualified::local(var_name.clone());
        if let Some(_cst_info) = self.typed.const_info(&qualified) {
            panic!(
                "type checker bug: we already have a constant with the same name (`{var_name}`)!"
            );
        }

        //
        fn_env.add_local_var(var_name, var_info)
    }

    pub fn get_local_var(&self, fn_env: &FnEnv<B::Field>, var_name: &str) -> VarInfo<B::Field> {
        // check for consts first
        let qualified = FullyQualified::local(var_name.to_string());
        if let Some(cst_info) = self.typed.const_info(&qualified) {
            let var = Var::new_constant_typ(cst_info, cst_info.typ.span);
            return VarInfo::new(var, false, Some(TyKind::Field));
        }

        // then check for local variables
        fn_env.get_local_var(var_name)
    }

    /// Retrieves the [FnInfo] for the `main()` function.
    /// This function should only be called if we know there's a main function,
    /// if there's no main function it'll panic.
    pub fn main_info(&self) -> Result<&FnInfo<B>> {
        let qualified = FullyQualified::local("main".to_string());
        self.typed
            .fn_info(&qualified)
            .ok_or(self.error(ErrorKind::NoMainFunction, Span::default()))
    }

    pub fn error(&self, kind: ErrorKind, span: Span) -> Error {
        Error::new("constraint-generation", kind, span)
    }
}

impl<B: Backend> CircuitWriter<B> {
    /// Creates a global environment from the one created by the type checker.
    pub fn new(typed: TypeChecker<B>, backend: B) -> Self {
        Self {
            typed,
            backend,
            finalized: false,
            public_input_size: 0,
            public_output: None,
            private_input_indices: vec![],
        }
    }

    pub fn generate_circuit(typed: TypeChecker<B>, backend: B) -> Result<CompiledCircuit<B>>
    where
        B: Backend,
    {
        // create circuit writer
        let mut circuit_writer = CircuitWriter::new(typed, backend);

        // get main function
        let main_fn_info = circuit_writer.main_info()?;

        let function = match &main_fn_info.kind {
            crate::imports::FnKind::BuiltIn(_, _) => unreachable!(),
            crate::imports::FnKind::Native(fn_sig) => fn_sig.clone(),
        };

        // create the main env
        let fn_env = &mut FnEnv::new();

        // create public and private inputs
        for FnArg {
            attribute,
            name,
            typ,
            ..
        } in &function.sig.arguments
        {
            // get length
            let len = match &typ.kind {
                TyKind::Field => 1,
                TyKind::Array(typ, len) => {
                    if !matches!(**typ, TyKind::Field) {
                        unimplemented!();
                    }
                    *len as usize
                }
                TyKind::Bool => 1,
                typ => circuit_writer.size_of(typ),
            };

            // create the variable
            let var = if let Some(attr) = attribute {
                if !matches!(attr.kind, AttributeKind::Pub) {
                    return Err(
                        circuit_writer.error(ErrorKind::InvalidAttribute(attr.kind), attr.span)
                    );
                }
                circuit_writer.add_public_inputs(name.value.clone(), len, name.span)
            } else {
                circuit_writer.add_private_inputs(name.value.clone(), len, name.span)
            };

            // constrain what needs to be constrained
            // (for example, booleans need to be constrained to be 0 or 1)
            // note: we constrain private inputs as well as public inputs
            // in theory we might not need to check the validity of public inputs,
            // but we are being extra cautious due to attacks
            // where the prover gives the verifier malformed inputs that look legit.
            // (See short address attacks in Ethereum.)
            circuit_writer.constrain_inputs_to_main(&var.cvars, &typ.kind, typ.span)?;

            // add argument variable to the ast env
            let mutable = false; // TODO: should we add a mut keyword in arguments as well?
            let var_info = VarInfo::new(var, mutable, Some(typ.kind.clone()));
            circuit_writer.add_local_var(fn_env, name.value.clone(), var_info);
        }

        // create public output
        if let Some(typ) = &function.sig.return_type {
            if typ.kind != TyKind::Field {
                unimplemented!();
            }

            // create it
            circuit_writer.add_public_outputs(1, typ.span);
        }

        // compile function
        let returned_cells = circuit_writer.compile_main_function(fn_env, &function)?;
        let main_span = circuit_writer.main_info().unwrap().span;
        let private_input_indices = circuit_writer.private_input_indices.clone();
        let public_output = circuit_writer.public_output.clone();

        circuit_writer.backend.finalize_circuit(
            public_output,
            returned_cells,
            private_input_indices,
            main_span,
        )?;

        // we finalized!
        circuit_writer.finalized = true;

        //
        Ok(CompiledCircuit::new(circuit_writer))
    }

    pub fn generate_asm(&self, sources: &Sources, debug: bool) -> String {
        self.backend.generate_asm(sources, debug)
    }

    pub fn generate_witness(
        &self,
        witness_env: &mut WitnessEnv<B::Field>,
    ) -> Result<GeneratedWitness<B>> {
        self.backend.generate_witness(witness_env, self.public_input_size)
    }
}
