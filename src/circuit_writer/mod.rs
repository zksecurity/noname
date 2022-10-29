use std::collections::{HashMap, HashSet};

use crate::{
    cli::packages::UserRepo,
    constants::{Field, Span},
    error::{Error, ErrorKind, Result},
    parser::{
        AttributeKind, Expr, FnArg, FnSig, Function, Ident, RootKind, Struct, TyKind, UsePath,
    },
    type_checker::{Dependencies, FnInfo, StructInfo, TypeChecker},
    var::{CellVar, Value, Var},
    witness::CompiledCircuit,
};

pub use fn_env::{FnEnv, VarInfo};
pub use writer::{Gate, GateKind, Wiring};

use self::writer::PendingGate;

pub mod fn_env;
pub mod writer;

#[derive(Debug)]
pub struct CircuitWriter {
    /// The source code of this module.
    /// Useful for debugging and displaying user errors.
    pub(crate) source: String,

    /// The type checker state for the main module.
    // Important: this field must not be used directly.
    // This is because, depending on the value of [current_module],
    // the type checker state might be this one, or one of the ones in [dependencies].
    typed: TypeChecker,

    /// The type checker state and source for the dependencies.
    // TODO: perhaps merge {source, typed} in this type?
    dependencies: Dependencies,

    /// The current module. If not set, the main module.
    // Note: this can be an alias that came from a 3rd party library.
    // For example, a 3rd party library might have written `use a::b as c;`.
    // For this reason we must store this as a fully-qualified module.
    current_module: Option<UserRepo>,

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
}

impl CircuitWriter {
    /// Retrieves the type checker associated to the current module being parsed.
    /// It is possible, when we jump to third-party libraries' code,
    /// that we need access to their type checker state instead of the main module one.
    pub fn current_type_checker(&self) -> &TypeChecker {
        if let Some(current_module) = &self.current_module {
            self.dependencies
                .get_type_checker(current_module)
                .expect(&format!(
                    "bug in the compiler: couldn't find current module: {:?}",
                    current_module
                ))
        } else {
            &self.typed
        }
    }

    pub fn expr_type(&self, expr: &Expr) -> Option<&TyKind> {
        dbg!("calling expr_type for", expr);
        let curr_type_checker = self.current_type_checker();
        dbg!("current type checker used from", &self.current_module);
        curr_type_checker.node_types.get(&expr.node_id)
    }

    pub fn node_type(&self, node_id: usize) -> Option<&TyKind> {
        let curr_type_checker = self.current_type_checker();
        curr_type_checker.node_types.get(&node_id)
    }

    pub fn struct_info(&self, name: &str) -> Option<&StructInfo> {
        let curr_type_checker = self.current_type_checker();
        curr_type_checker.struct_info(name)
    }

    pub fn fn_info(&self, name: &str) -> Option<&FnInfo> {
        let curr_type_checker = self.current_type_checker();
        curr_type_checker.functions.get(name)
    }

    pub fn size_of(&self, typ: &TyKind) -> Result<usize> {
        let curr_type_checker = self.current_type_checker();
        curr_type_checker.size_of(&self.dependencies, typ)
    }

    pub fn resolve_module(&self, module: &Ident) -> Result<&UsePath> {
        let curr_type_checker = self.current_type_checker();
        curr_type_checker.modules.get(&module.value).ok_or_else(|| {
            Error::new(
                ErrorKind::UndefinedModule(module.value.clone()),
                module.span,
            )
        })
    }

    pub fn get_fn(&self, module: &Option<Ident>, fn_name: &Ident) -> Result<FnInfo> {
        if let Some(module) = module {
            // we may be parsing a function from a 3rd-party library
            // which might also come from another 3rd-party library
            let module = self.resolve_module(module)?;
            self.dependencies.get_fn(module, fn_name)
        } else {
            let curr_type_checker = self.current_type_checker();
            let fn_info = curr_type_checker
                .functions
                .get(&fn_name.value)
                .cloned()
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::UndefinedFunction(fn_name.value.clone()),
                        fn_name.span,
                    )
                })?;
            Ok(fn_info)
        }
    }

    pub fn get_struct(&self, module: &Option<Ident>, struct_name: &Ident) -> Result<StructInfo> {
        if let Some(module) = module {
            // we may be parsing a struct from a 3rd-party library
            // which might also come from another 3rd-party library
            let module = self.resolve_module(module)?;
            self.dependencies.get_struct(module, struct_name)
        } else {
            let curr_type_checker = self.current_type_checker();
            let struct_info = curr_type_checker
                .struct_info(&struct_name.value)
                .ok_or(Error::new(
                    ErrorKind::UndefinedStruct(struct_name.value.clone()),
                    struct_name.span,
                ))?
                .clone();
            Ok(struct_info)
        }
    }

    pub fn add_local_var(&self, fn_env: &mut FnEnv, var_name: String, var_info: VarInfo) {
        // check for consts first
        let type_checker = self.current_type_checker();
        if let Some(_cst_info) = type_checker.constants.get(&var_name) {
            panic!(
                "type checker bug: we already have a constant with the same name (`{var_name}`)!"
            );
        }

        //
        fn_env.add_local_var(var_name, var_info)
    }

    pub fn get_local_var(&self, fn_env: &FnEnv, var_name: &str) -> VarInfo {
        // check for consts first
        let type_checker = self.current_type_checker();
        if let Some(cst_info) = type_checker.constants.get(var_name) {
            let var = Var::new_constant(cst_info.value, cst_info.typ.span);
            return VarInfo::new(var, false, Some(TyKind::Field));
        }

        // then check for local variables
        fn_env.get_local_var(var_name)
    }

    /// Retrieves the [FnInfo] for the `main()` function.
    /// This function should only be called if we know there's a main function,
    /// if there's no main function it'll panic.
    pub fn main_info(&self) -> &FnInfo {
        self.typed
            .fn_info("main")
            .expect("bug in the compiler: main not found")
    }
}

impl CircuitWriter {
    pub fn generate_circuit(
        typed: TypeChecker,
        deps: Dependencies,
        code: &str,
    ) -> Result<CompiledCircuit> {
        // create circuit writer
        let mut circuit_writer = CircuitWriter::new(code, typed, deps);

        // get main function
        let main_fn_info = circuit_writer
            .typed
            .functions
            .get("main")
            .ok_or(Error::new(ErrorKind::NoMainFunction, Span::default()))?;

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
                typ => circuit_writer.size_of(typ)?,
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
                        let fn_info = circuit_writer.typed.functions.get("main").cloned().unwrap();

                        (fn_info.sig().clone(), fn_info.span)
                    };

                    // TODO: is this error useful?
                    return Err(Error::new(ErrorKind::PrivateInputNotUsed, main_span));
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
