use std::collections::HashMap;

use ark_ff::Field;
use itertools::chain;
//use serde::{Deserialize, Serialize};

use crate::{
    backends::Backend,
    circuit_writer::CircuitWriter,
    compiler::Sources,
    error::{Error, ErrorKind, Result},
    inputs::JsonInputs,
    type_checker::FnInfo,
};

#[derive(Debug, Default)]
pub struct WitnessEnv<F>
where
    F: Field,
{
    pub var_values: HashMap<String, Vec<F>>,

    pub cached_values: HashMap<usize, F>,
}

impl<F: Field> WitnessEnv<F> {
    pub fn add_value(&mut self, name: String, val: Vec<F>) {
        assert!(self.var_values.insert(name, val).is_none());
    }

    #[must_use]
    pub fn get_external(&self, name: &str) -> Vec<F> {
        // TODO: return an error instead of crashing
        self.var_values.get(name).unwrap().clone()
    }
}

/// The compiled circuit.
//#[derive(Serialize, Deserialize)]
pub struct CompiledCircuit<B: Backend> {
    pub circuit: CircuitWriter<B>,
}

impl<B: Backend> CompiledCircuit<B> {
    pub(crate) fn new(circuit: CircuitWriter<B>) -> Self {
        Self { circuit }
    }

    pub fn main_info(&self) -> &FnInfo<B> {
        self.circuit
            .main_info()
            .expect("constrait-writer bug: no main function found in witness generation")
    }

    pub fn asm(&self, sources: &Sources, debug: bool) -> String {
        self.circuit.backend.generate_asm(sources, debug)
    }

    pub fn generate_witness(
        &self,
        mut public_inputs: JsonInputs,
        mut private_inputs: JsonInputs,
    ) -> Result<B::GeneratedWitness> {
        let mut env = WitnessEnv::default();

        // get info on main
        let main_info = self.main_info();
        let main_sig = match &main_info.kind {
            crate::imports::FnKind::BuiltIn(_, _) => unreachable!(),
            crate::imports::FnKind::Native(fn_sig) => &fn_sig.sig,
        };

        // create the argument's variables?
        for arg in &main_sig.arguments {
            let name = &arg.name.value;

            let input = if arg.is_public() {
                public_inputs.0.remove(name).ok_or_else(|| {
                    Error::new(
                        "runtime",
                        ErrorKind::MissingPublicArg(name.clone()),
                        arg.span,
                    )
                })?
            } else {
                private_inputs.0.remove(name).ok_or_else(|| {
                    Error::new(
                        "runtime",
                        ErrorKind::MissingPrivateArg(name.clone()),
                        arg.span,
                    )
                })?
            };

            let fields = self
                .parse_single_input(input, &arg.typ.kind)
                .map_err(|e| Error::new("runtime", ErrorKind::ParsingError(e), arg.span))?;

            env.add_value(name.clone(), fields.clone());
        }

        // ensure that we've used all of the inputs provided
        if let Some(name) = chain![private_inputs.0.keys(), public_inputs.0.keys()].next() {
            return Err(Error::new(
                "runtime",
                ErrorKind::UnusedInput(name.clone()),
                main_info.span,
            ));
        }

        self.circuit.generate_witness(&mut env)
    }
}
