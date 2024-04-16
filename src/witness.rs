use std::collections::HashMap;

use ark_ff::{Field, Zero};
use itertools::{chain, izip, Itertools};
//use serde::{Deserialize, Serialize};

use crate::{
    backends::{kimchi::GeneratedWitness, Backend},
    circuit_writer::{CircuitWriter, Gate},
    compiler::Sources,
    constants::NUM_REGISTERS,
    error::{Error, ErrorKind, Result},
    helpers::PrettyField,
    inputs::JsonInputs,
    type_checker::FnInfo,
    var::{CellVar, Value},
};

#[derive(Debug, Default)]
pub struct WitnessEnv<F>
where
    F: Field,
{
    pub var_values: HashMap<String, Vec<F>>,

    pub cached_values: HashMap<CellVar, F>,
}

impl<F: Field> WitnessEnv<F> {
    pub fn add_value(&mut self, name: String, val: Vec<F>) {
        assert!(self.var_values.insert(name, val).is_none());
    }

    pub fn get_external(&self, name: &str) -> Vec<F> {
        // TODO: return an error instead of crashing
        self.var_values.get(name).unwrap().clone()
    }
}


impl<F: Field + PrettyField> Witness<F> {
    /// kimchi uses a transposed witness
    pub fn to_kimchi_witness(&self) -> [Vec<F>; NUM_REGISTERS] {
        let transposed = vec![Vec::with_capacity(self.0.len()); NUM_REGISTERS];
        let mut transposed: [_; NUM_REGISTERS] = transposed.try_into().unwrap();
        for row in &self.0 {
            for (col, field) in row.iter().enumerate() {
                transposed[col].push(*field);
            }
        }
        transposed
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn debug(&self) {
        for (row, values) in self.0.iter().enumerate() {
            let values = values.iter().map(|v| v.pretty()).join(" | ");
            println!("{row} - {values}");
        }
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
        self.circuit.generate_asm(sources, debug)
    }

    pub fn compiled_gates(&self) -> &[Gate<B>] {
        self.circuit.compiled_gates()
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
