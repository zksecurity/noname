use std::collections::HashMap;

use crate::{
    backends::Backend,
    constants::Span,
    error::{Error, ErrorKind, Result},
    parser::{
        types::{ArraySize, AttributeKind, FnArg, TyKind},
        Expr,
    },
    type_checker::{ConstInfo, FnInfo, FullyQualified, StructInfo, TypeChecker},
    var::Var,
    witness::{CompiledCircuit, WitnessEnv},
};

pub use fn_env::{FnEnv, VarInfo};
use serde::{Deserialize, Serialize};
use writer::ComputedExpr;
//use serde::{Deserialize, Serialize};
pub use writer::{Gate, GateKind, Wiring};

pub mod fn_env;
pub mod writer;

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

    /// If a public output is set, this will be used to store its [Var].
    /// The public output generation works as follows:
    /// 1. This cvar is created and inserted in the circuit (gates) during compilation of the public input
    ///    (as the public output is the end of the public input)
    /// 2. When the `return` statement of the circuit is parsed,
    ///    it will set this `public_output` variable again to the correct vars.
    /// 3. During witness generation, the public output computation
    ///    is delayed until the very end.
    pub(crate) public_output: Option<Var<B::Field, B::Var>>,
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
        fn_env: &mut FnEnv<B::Field, B::Var>,
        var_name: String,
        var_info: VarInfo<B::Field, B::Var>,
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

    pub fn get_local_var(
        &self,
        fn_env: &FnEnv<B::Field, B::Var>,
        var_name: &str,
    ) -> VarInfo<B::Field, B::Var> {
        // check for consts first
        let qualified = FullyQualified::local(var_name.to_string());
        if let Some(cst_info) = self.typed.const_info(&qualified) {
            let var = Var::new_constant_typ(cst_info, cst_info.typ.span);
            let span = var.span;
            let ce = ComputedExpr::new(writer::ComputedExprKind::Field(var), span);
            return VarInfo::new(ce, false, Some(TyKind::Field));
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
    fn new(typed: TypeChecker<B>, backend: B) -> Self {
        Self {
            typed,
            backend,
            public_output: None,
        }
    }

    pub fn generate_circuit(typed: TypeChecker<B>, backend: B) -> Result<CompiledCircuit<B>> {
        // create circuit writer
        let mut circuit_writer = CircuitWriter::new(typed, backend);

        // get main function
        let qualified = FullyQualified::local("main".to_string());
        let main_fn_info = circuit_writer.main_info()?;

        let function = match &main_fn_info.kind {
            crate::imports::FnKind::BuiltIn(_, _) => unreachable!(),
            crate::imports::FnKind::Native(fn_sig) => fn_sig.clone(),
        };

        // initialize the circuit
        circuit_writer.backend.init_circuit();

        // create the main env
        // todo: might need to add all the consts to the main env before anything else
        let fn_env = &mut FnEnv::new();

        // create public output
        if let Some(typ) = &function.sig.return_type {
            if typ.kind != TyKind::Field {
                unimplemented!();
            }

            // create it
            circuit_writer.add_public_outputs(1, typ.span);
        }

        // public inputs should be handled first
        for arg in function.sig.arguments.iter().filter(|arg| arg.is_public()) {
            match &arg.attribute {
                Some(attr) => {
                    if !matches!(attr.kind, AttributeKind::Pub) {
                        return Err(
                            circuit_writer.error(ErrorKind::InvalidAttribute(attr.kind), attr.span)
                        );
                    }
                }
                None => panic!("public arguments must have a pub attribute"),
            }
            circuit_writer.handle_arg(arg, fn_env, CircuitWriter::add_public_inputs)?;
        }

        // then handle private inputs
        for arg in function.sig.arguments.iter().filter(|arg| !arg.is_public()) {
            circuit_writer.handle_arg(arg, fn_env, CircuitWriter::add_private_inputs)?;
        }

        // compile function
        let returned_cells = circuit_writer.compile_main_function(fn_env, &function)?;
        let main_span = circuit_writer.main_info().unwrap().span;
        let public_output = circuit_writer.public_output.clone();

        // constraint public outputs to the result of the circuit
        if let Some(public_output) = &public_output {
            let cvars = &public_output.cvars;

            for (pub_var, ret_var) in cvars.iter().zip(&returned_cells.clone().unwrap()) {
                circuit_writer
                    .backend
                    .assert_eq_var(pub_var.cvar().unwrap(), ret_var, main_span);
            }
        }

        circuit_writer
            .backend
            .finalize_circuit(public_output, returned_cells, main_span)?;

        //
        Ok(CompiledCircuit::new(circuit_writer))
    }

    /// A wrapper for the backend generate_witness
    pub fn generate_witness(
        &self,
        witness_env: &mut WitnessEnv<B::Field>,
    ) -> Result<B::GeneratedWitness> {
        self.backend.generate_witness(witness_env)
    }

    fn handle_arg(
        &mut self,
        arg: &FnArg,
        fn_env: &mut FnEnv<B::Field, B::Var>,
        handle_input: fn(&mut CircuitWriter<B>, String, usize, Span) -> Var<B::Field, B::Var>,
    ) -> Result<()> {
        let FnArg { name, typ, .. } = arg;

        // get length
        let len = match &typ.kind {
            TyKind::Field => 1,
            // todo: the size will get from const var
            TyKind::Array(typ, size) => {
                if !matches!(**typ, TyKind::Field) {
                    unimplemented!();
                }
                // *size as usize
                match size {
                    ArraySize::Number(n) => *n as usize,
                    ArraySize::ConstVar(_) => panic!(
                        "array argument with size in const var is not supported for main function"
                    ),
                }
            }
            TyKind::Bool => 1,
            typ => self.size_of(typ),
        };

        // create the variable
        let var = handle_input(self, name.value.clone(), len, name.span);
        // let span = var.span;
        // todo: restructure var to be a ComputedExpr
        let ce = self.compute_expr_arg(&typ.kind, var);

        // constrain what needs to be constrained
        // (for example, booleans need to be constrained to be 0 or 1)
        // note: we constrain private inputs as well as public inputs
        // in theory we might not need to check the validity of public inputs,
        // but we are being extra cautious due to attacks
        // where the prover gives the verifier malformed inputs that look legit.
        // (See short address attacks in Ethereum.)
        let cvars = &ce.clone().value().cvars;
        self.constrain_inputs_to_main(cvars, &typ.kind, typ.span)?;

        // add argument variable to the ast env
        let mutable = false; // TODO: should we add a mut keyword in arguments as well?
                             // let ce = ComputedExpr::new(writer::ComputedExprKind::FnCall(var), span);
        let var_info = VarInfo::new(ce, mutable, Some(typ.kind.clone()));
        self.add_local_var(fn_env, name.value.clone(), var_info);

        Ok(())
    }

    /// Maps the arguments to ComputedExpr.
    /// These are arguments are in a form of var array.
    /// With the type info, we can restructure the cvars as ComputexExpr,
    /// which can retain the structure of the type.
    fn compute_expr_arg(
        &self,
        typ: &TyKind,
        var: Var<B::Field, B::Var>,
    ) -> ComputedExpr<B::Field, B::Var> {
        let span = var.span;
        match typ {
            TyKind::Field => ComputedExpr::new(writer::ComputedExprKind::Field(var), span),
            TyKind::BigInt => ComputedExpr::new(writer::ComputedExprKind::Field(var), span),
            TyKind::Bool => ComputedExpr::new(writer::ComputedExprKind::Bool(var), span),
            TyKind::Custom { module, name } => {
                let qualified = FullyQualified::new(module, name);
                let struct_info = self
                    .struct_info(&qualified)
                    .expect("bug in the type checker: cannot find struct info");

                let mut custom = HashMap::new();

                let mut cvars = var.cvars;

                for (field, t) in &struct_info.fields {
                    // slice a range of the size from the beginning of var.cvars, and construct a new var with the rest of the cvars
                    let size = self.size_of(t);
                    let rest = cvars.split_off(size);
                    let new_var = Var::new(cvars, span);

                    let v = self.compute_expr_arg(t, new_var);
                    custom.insert(field.to_string(), v);

                    cvars = rest;
                }

                ComputedExpr::new(writer::ComputedExprKind::Struct(custom), span)
            }
            TyKind::Array(typ, size) => match size {
                ArraySize::Number(n) => {
                    let mut array = vec![];
                    let mut cvars = var.cvars;

                    let size = self.size_of(typ);

                    for _ in 0..*n {
                        let rest = cvars.split_off(size);
                        let new_var = Var::new(cvars, span);
                        let v = self.compute_expr_arg(typ, new_var);
                        array.push(v);
                        cvars = rest;
                    }

                    ComputedExpr::new(writer::ComputedExprKind::Array(array), span)
                }
                // todo: this enum case might be just useful for the main function arg
                // because it might only gauranttee the main function access to the const var,
                // which isn't changed and can be found in the type checker
                ArraySize::ConstVar(v) => {
                    // todo: what if it is not from a local module?
                    let qualified = FullyQualified::local(v.clone());
                    let cst_info = self
                        .const_info(&qualified)
                        .expect("bug in the type checker: cannot find constant info");

                    let n = crate::utils::to_u32(cst_info.value[0]) as usize;

                    let mut array = vec![];
                    let mut cvars = var.cvars;

                    let size = self.size_of(typ);

                    for _ in 0..n {
                        let rest = cvars.split_off(size);
                        let new_var = Var::new(cvars, span);
                        let v = self.compute_expr_arg(typ, new_var);
                        array.push(v);
                        cvars = rest;
                    }

                    ComputedExpr::new(writer::ComputedExprKind::Array(array), span)
                }
            },
        }
    }
}

mod test {
    use std::{path::Path, str::FromStr};

    use crate::{
        backends::r1cs::{self, R1csBls12381Field},
        compiler::{typecheck_next_file, Sources},
        error::{Error, ErrorKind, Result},
        inputs::parse_inputs,
        type_checker::TypeChecker,
    };

    use super::CircuitWriter;

    fn test_file(
        code: &str,
        public_inputs: &str,
        private_inputs: &str,
        expected_public_output: Vec<&str>,
    ) -> Result<()> {
        // parse inputs
        let public_inputs = parse_inputs(public_inputs).unwrap();
        let private_inputs = parse_inputs(private_inputs).unwrap();
        // compile
        let mut sources = Sources::new();
        let mut tast = TypeChecker::new();
        let this_module = None;
        let _node_id = typecheck_next_file(
            &mut tast,
            this_module,
            &mut sources,
            "".to_string(),
            code.to_string(),
            0,
        )
        .unwrap();

        let backend = r1cs::R1CS::<R1csBls12381Field>::new();
        let compiled_circuit = CircuitWriter::generate_circuit(tast, backend)?;

        // this should check the constraints
        let generated_witness = compiled_circuit
            .generate_witness(public_inputs.clone(), private_inputs.clone())
            .unwrap();

        let expected_public_output = expected_public_output
            .iter()
            .map(|x| crate::backends::r1cs::R1csBls12381Field::from_str(x).unwrap())
            .collect::<Vec<_>>();

        if generated_witness.outputs != expected_public_output {
            eprintln!("obtained by executing the circuit:");
            generated_witness
                .outputs
                .iter()
                .for_each(|x| eprintln!("- {x}"));
            eprintln!("passed as output by the verifier:");
            expected_public_output
                .iter()
                .for_each(|x| eprintln!("- {x}"));
            panic!("Obtained output does not match expected output");
        }

        Ok(())
    }

    #[test]
    fn test_invalid_direct_array_access() {
        const CODE: &str = r#"
            const size = 3;
            fn main(pub xx: Field) {
                let mut yy = [xx; size];
                yy[3] = 1;
            }
            "#;

        let public_inputs = r#"{"xx": "1"}"#;
        let private_inputs = r#"{}"#;

        let result = test_file(
            CODE,
            public_inputs,
            private_inputs,
            vec![],
        );

        assert!(result.is_err(), "expected error");
        match result.unwrap_err() {
            Error {
                kind: ErrorKind::ArrayIndexOutOfBounds(path, 3, 2),
                ..
            } => {
                assert_eq!(path, "yy");
            },
            err => panic!("unexpected error: {:?}", err),
        }
    }

    #[test]
    fn test_invalid_nested_array_access_final_path() {
        const CODE: &str = r#"
            const size = 3;

            struct Thing {
                stuffs: [Field; 3],
            }

            fn main(pub xx: Field) {
                let mut yy = [Thing {stuffs: [xx; 3]}; size];
                yy[1].stuffs[3] = 1;
            }
            "#;

        let public_inputs = r#"{"xx": "1"}"#;
        let private_inputs = r#"{}"#;

        let result = test_file(
            CODE,
            public_inputs,
            private_inputs,
            vec![],
        );

        assert!(result.is_err(), "expected error");
        match result.unwrap_err() {
            Error {
                kind: ErrorKind::ArrayIndexOutOfBounds(path, 3, 2),
                ..
            } => {
                assert_eq!(path, "yy[1].stuffs");
            },
            err => panic!("unexpected error: {:?}", err),
        }
    }

    #[test]
    fn test_invalid_nested_array_access_early_path() {
        const CODE: &str = r#"
            const size = 3;

            struct Thing {
                stuffs: [Field; 3],
            }

            fn main(pub xx: Field) {
                let mut yy = [Thing {stuffs: [xx; 3]}; size];
                yy[3].stuffs[1] = 1;
            }
            "#;

        let public_inputs = r#"{"xx": "1"}"#;
        let private_inputs = r#"{}"#;

        let result = test_file(
            CODE,
            public_inputs,
            private_inputs,
            vec![],
        );

        assert!(result.is_err(), "expected error");
        match result.unwrap_err() {
            Error {
                kind: ErrorKind::ArrayIndexOutOfBounds(path, 3, 2),
                ..
            } => {
                assert_eq!(path, "yy");
            },
            err => panic!("unexpected error: {:?}", err),
        }
    }
}
