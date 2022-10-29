use std::collections::{HashMap, HashSet};

use crate::{
    constants::{Field, Span},
    error::{Error, ErrorKind, Result},
    parser::{AttributeKind, FnArg, FnSig, Function, RootKind, Struct, TyKind},
    type_checker::{Dependencies, TypeChecker, TAST},
    var::{CellVar, Value, Var},
    witness::CompiledCircuit,
};

pub use fn_env::{FnEnv, VarInfo};
pub use writer::{Gate, GateKind, Wiring};

use self::writer::PendingGate;

pub mod fn_env;
pub mod writer;

#[derive(Default, Debug)]
pub struct CircuitWriter {
    /// The source code that created this circuit.
    /// Useful for debugging and displaying user errors.
    // TODO: why is this here? probably doesn't need to be
    pub(crate) source: String,

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

    /// Used during the witness generation to check
    /// public and private inputs given by the prover.
    pub(crate) main: (FnSig, Span),

    /// The state of the type checker.
    pub(crate) typed: TypeChecker,

    /// Constants defined in the module/program.
    pub(crate) constants: HashMap<String, VarInfo>,

    /// If set to false, a single generic gate will be used per double generic gate.
    /// This can be useful for debugging.
    pub(crate) double_generic_gate_optimization: bool,

    /// This is used to implement the double generic gate,
    /// which encodes two generic gates.
    pub(crate) pending_generic_gate: Option<PendingGate>,

    /// We cache the association between a constant and its _constrained_ variable,
    /// this is to avoid creating a new constraint every time we need to hardcode the same constant.
    pub(crate) cached_constants: HashMap<Field, CellVar>,
}

impl CircuitWriter {
    pub fn generate_circuit(
        tast: TAST,
        deps: &Dependencies,
        code: &str,
    ) -> Result<CompiledCircuit> {
        // if there's no main function, then return an error
        let TAST {
            ast,
            typed_global_env,
        } = tast;

        if !typed_global_env.has_main {
            return Err(Error::new(ErrorKind::NoMainFunction, Span::default()));
        }

        let mut circuit_writer = CircuitWriter::new(code, typed_global_env);

        // make sure we can't call that several times
        if circuit_writer.finalized {
            panic!("circuit already finalized (TODO: return a proper error");
        }

        //
        // Process constants
        //

        // we detect struct or function definition
        let mut func_or_struct = None;

        for root in &ast.0 {
            match &root.kind {
                RootKind::Const(cst) => {
                    // important: no struct or function definition must appear before a constant declaration
                    if let Some(span) = func_or_struct {
                        return Err(Error::new(
                            ErrorKind::ConstDeclarationAfterStructOrFunction,
                            span,
                        ));
                    }
                    circuit_writer.add_constant_var(cst.name.value.clone(), cst.value, cst.span);
                }

                RootKind::Function(Function { span, .. })
                | RootKind::Struct(Struct { span, .. }) => func_or_struct = Some(*span),

                RootKind::Use(_) | RootKind::Comment(_) => (),
            }
        }

        //
        // Generate the circuit
        //

        for root in &ast.0 {
            match &root.kind {
                // imports (already dealt with in type checker)
                RootKind::Use(_path) => (),

                // `const thing = 42;`
                RootKind::Const(_cst) => {
                    // already took care of it previously
                    ()
                }

                // `fn main() { ... }`
                RootKind::Function(function) => {
                    // create the env
                    let fn_env = &mut FnEnv::new(&circuit_writer.constants);

                    // we only compile the main function
                    if !function.is_main() {
                        continue;
                    }

                    // if there are no arguments, return an error
                    // TODO: should we check this in the type checker?
                    if function.sig.arguments.is_empty() {
                        return Err(Error::new(ErrorKind::NoArgsInMain, function.span));
                    }

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
                            typ => circuit_writer.typed.size_of(deps, typ)?,
                        };

                        // create the variable
                        let var = if let Some(attr) = attribute {
                            if !matches!(attr.kind, AttributeKind::Pub) {
                                return Err(Error::new(
                                    ErrorKind::InvalidAttribute(attr.kind),
                                    attr.span,
                                ));
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
                        circuit_writer
                            .constrain_inputs_to_main(deps, &var.cvars, &typ.kind, typ.span)?;

                        // add argument variable to the ast env
                        let mutable = false; // TODO: should we add a mut keyword in arguments as well?
                        let var_info = VarInfo::new(var, mutable, Some(typ.kind.clone()));
                        fn_env.add_var(name.value.clone(), var_info);
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
                    circuit_writer.compile_main_function(fn_env, deps, function)?;
                }

                // struct definition (already dealt with in type checker)
                RootKind::Struct(_struct) => (),

                // ignore comments
                // TODO: we could actually preserve the comment in the ASM!
                RootKind::Comment(_comment) => (),
            }
        }

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
                    return Err(Error::new(
                        ErrorKind::PrivateInputNotUsed,
                        circuit_writer.main.1,
                    ));
                } else {
                    panic!("there's a bug in the circuit_writer, some cellvar does not end up being a cellvar in the circuit!");
                }
            }
        }

        // we finalized!
        circuit_writer.finalized = true;

        Ok(CompiledCircuit::new(circuit_writer))
    }
}
