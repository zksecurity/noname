use std::collections::{HashMap, HashSet};

use crate::{
    cli::packages::UserRepo,
    constants::{Field, Span},
    error::{Error, ErrorKind, Result},
    parser::{
        types::{AttributeKind, FnArg, Ident, TyKind, UsePath},
        Expr,
    },
    type_checker::{ConstInfo, FnInfo, FullyQualified, StructInfo, TypeChecker},
    var::{CellVar, Value, Var},
    witness::CompiledCircuit,
};

pub use fn_env::{FnEnv, VarInfo};
use miette::NamedSource;
use serde::{Deserialize, Serialize};
//use serde::{Deserialize, Serialize};
pub use writer::{Gate, GateKind, Wiring};

use self::writer::PendingGate;

pub mod fn_env;
pub mod writer;

//#[derive(Debug, Serialize, Deserialize)]
#[derive(Debug)]
pub struct CircuitWriter {
    /// The type checker state for the main module.
    // Important: this field must not be used directly.
    // This is because, depending on the value of [current_module],
    // the type checker state might be this one, or one of the ones in [dependencies].
    typed: TypeChecker,

    /// Once this is set, you can generate a witness (and can't modify the circuit?)
    // Note: I don't think we need this, but it acts as a nice redundant failsafe.
    pub(crate) finalized: bool,

    /// This is used to give a distinct number to each variable during circuit generation.
    pub(crate) next_variable: usize,

    /// This is how you compute the value of each variable during witness generation.
    /// It is created during circuit generation.
    pub(crate) witness_vars: HashMap<usize, Value>,

    /// The execution trace table with vars as placeholders.
    /// It is created during circuit generation,
    /// and used by the witness generator.
    pub(crate) rows_of_vars: Vec<Vec<Option<CellVar>>>,

    /// The gates created by the circuit generation.
    gates: Vec<Gate>,

    /// The wiring of the circuit.
    /// It is created during circuit generation.
    pub(crate) wiring: HashMap<usize, Wiring>,

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
    pub(crate) public_output: Option<Var>,

    /// Indexes used by the private inputs
    /// (this is useful to check that they appear in the circuit)
    pub(crate) private_input_indices: Vec<usize>,

    /// If set to false, a single generic gate will be used per double generic gate.
    /// This can be useful for debugging.
    pub(crate) double_generic_gate_optimization: bool,

    /// This is used to implement the double generic gate,
    /// which encodes two generic gates.
    pub(crate) pending_generic_gate: Option<PendingGate>,

    /// We cache the association between a constant and its _constrained_ variable,
    /// this is to avoid creating a new constraint every time we need to hardcode the same constant.
    pub(crate) cached_constants: HashMap<Field, CellVar>,

    /// A vector of debug information that maps to each row of the created circuit.
    pub(crate) debug_info: Vec<DebugInfo>,
}

/// Debug information related to a single row in a circuit.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DebugInfo {
    /// The place in the original source code that created that gate.
    pub span: Span,

    /// A note on why this was added
    pub note: String,
}

impl CircuitWriter {
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

    pub fn fn_info(&self, qualified: &FullyQualified) -> Option<&FnInfo> {
        self.typed.fn_info(qualified)
    }

    pub fn const_info(&self, qualified: &FullyQualified) -> Option<&ConstInfo> {
        self.typed.const_info(qualified)
    }

    pub fn size_of(&self, typ: &TyKind) -> usize {
        self.typed.size_of(typ)
    }

    pub fn add_local_var(&self, fn_env: &mut FnEnv, var_name: String, var_info: VarInfo) {
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

    pub fn get_local_var(&self, fn_env: &FnEnv, var_name: &str) -> VarInfo {
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
    pub fn main_info(&self) -> Result<&FnInfo> {
        let qualified = FullyQualified::local("main".to_string());
        self.typed
            .fn_info(&qualified)
            .ok_or(self.error(ErrorKind::NoMainFunction, Span::default()))
    }

    pub fn error(&self, kind: ErrorKind, span: Span) -> Error {
        Error::new("constraint-generation", kind, span)
    }
}

impl CircuitWriter {
    /// Creates a global environment from the one created by the type checker.
    fn new(typed: TypeChecker, double_generic_gate_optimization: bool) -> Self {
        Self {
            typed,
            finalized: false,
            next_variable: 0,
            witness_vars: HashMap::new(),
            rows_of_vars: vec![],
            gates: vec![],
            wiring: HashMap::new(),
            public_input_size: 0,
            public_output: None,
            private_input_indices: vec![],
            double_generic_gate_optimization,
            pending_generic_gate: None,
            cached_constants: HashMap::new(),
            debug_info: vec![],
        }
    }

    pub fn generate_circuit(
        typed: TypeChecker,
        double_generic_gate_optimization: bool,
    ) -> Result<CompiledCircuit> {
        // create circuit writer
        let mut circuit_writer = CircuitWriter::new(typed, double_generic_gate_optimization);

        // get main function
        let qualified = FullyQualified::local("main".to_string());
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
        circuit_writer.compile_main_function(fn_env, &function)?;

        // important: there might still be a pending generic gate
        if let Some(pending) = circuit_writer.pending_generic_gate.take() {
            circuit_writer.add_gate(
                pending.label,
                GateKind::DoubleGeneric,
                pending.vars,
                pending.coeffs,
                pending.span,
            );
        }

        // for sanity check, we make sure that every cellvar created has ended up in a gate
        let mut written_vars = HashSet::new();
        for row in &circuit_writer.rows_of_vars {
            row.iter().flatten().for_each(|cvar| {
                written_vars.insert(cvar.index);
            });
        }

        for var in 0..circuit_writer.next_variable {
            if !written_vars.contains(&var) {
                if circuit_writer.private_input_indices.contains(&var) {
                    // compute main sig
                    let (_main_sig, main_span) = {
                        let fn_info = circuit_writer.main_info()?.clone();

                        (fn_info.sig().clone(), fn_info.span)
                    };

                    // TODO: is this error useful?
                    return Err(circuit_writer.error(ErrorKind::PrivateInputNotUsed, main_span));
                } else {
                    panic!("there's a bug in the circuit_writer, some cellvar does not end up being a cellvar in the circuit!");
                }
            }
        }

        // kimchi hack
        if circuit_writer.gates.len() <= 2 {
            panic!("the circuit is either too small or does not constrain anything (TODO: better error)");
        }

        // we finalized!
        circuit_writer.finalized = true;

        //
        Ok(CompiledCircuit::new(circuit_writer))
    }
}
