use std::{
    collections::{HashMap, HashSet},
    fmt::{self, Display, Formatter},
    ops::Neg,
    vec,
};

use ark_ff::{One, Zero};
use num_bigint::BigUint;
use num_traits::Num as _;

use crate::{
    asm, boolean,
    constants::{Field, Span, NUM_REGISTERS},
    error::{Error, ErrorKind, Result},
    field,
    imports::FnKind,
    parser::{
        AttributeKind, Expr, ExprKind, FnArg, FnSig, Function, Op2, RootKind, Stmt, StmtKind,
        TyKind,
    },
    syntax::is_type,
    type_checker::{TypedGlobalEnv, TAST},
    var::{CellVar, ConstOrCell, Value, Var},
    witness::CompiledCircuit,
};

//
// Data structures
//

/// The environment of the module/program.
#[derive(Debug)]
pub struct GlobalEnv {
    typed: TypedGlobalEnv,

    /// Constants defined in the module/program.
    constants: HashMap<String, VarInfo>,
}

impl GlobalEnv {
    /// Creates a global environment from the one created by the type checker.
    pub fn new(typed_global_env: TypedGlobalEnv) -> Self {
        Self {
            typed: typed_global_env,
            constants: HashMap::new(),
        }
    }

    /// Stores type information about a local variable.
    /// Note that we forbid shadowing at all scopes.
    pub fn add_constant(&mut self, name: String, constant: Field, span: Span) {
        let var = Var::new_constant(constant, span);

        let var_info = VarInfo::new(var, false, Some(TyKind::Field));

        if self.constants.insert(name.clone(), var_info).is_some() {
            panic!("constant `{name}` already exists (TODO: better error)");
        }
    }

    /// Retrieves type information on a constantiable, given a name.
    /// If the constantiable is not in scope, return false.
    // TODO: return an error no?
    pub fn get_constant(&self, ident: &str) -> Option<&VarInfo> {
        self.constants.get(ident)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GateKind {
    DoubleGeneric,
    Poseidon,
}

impl From<GateKind> for kimchi::circuits::gate::GateType {
    fn from(gate_kind: GateKind) -> Self {
        use kimchi::circuits::gate::GateType::*;
        match gate_kind {
            GateKind::DoubleGeneric => Generic,
            GateKind::Poseidon => Poseidon,
        }
    }
}

// TODO: this could also contain the span that defined the gate!
#[derive(Debug)]
pub struct Gate {
    /// Type of gate
    pub typ: GateKind,

    /// col -> (row, col)
    // TODO: do we want to do an external wiring instead?
    //    wiring: HashMap<u8, (u64, u8)>,

    /// Coefficients
    pub coeffs: Vec<Field>,

    /// The place in the original source code that created that gate.
    pub span: Span,

    /// A note on why this was added
    pub note: &'static str,
}

impl Gate {
    pub fn to_kimchi_gate(&self, row: usize) -> kimchi::circuits::gate::CircuitGate<Field> {
        kimchi::circuits::gate::CircuitGate {
            typ: self.typ.into(),
            wires: kimchi::circuits::wires::Wire::new(row),
            coeffs: self.coeffs.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Cell {
    pub row: usize,
    pub col: usize,
}

impl Display for Cell {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({},{})", self.row, self.col)
    }
}

#[derive(Debug, Clone)]
pub enum Wiring {
    /// Not yet wired (just indicates the position of the cell itself)
    NotWired(Cell),
    /// The wiring (associated to different spans)
    Wired(Vec<(Cell, Span)>),
}

/// Represents a variable in the circuit, or a reference to one.
/// Note that mutable variables are always passed as references,
/// as one needs to have access to the variable name to be able to reassign it in the environment.
pub enum VarOrRef {
    /// A [Var].
    Var(Var),

    /// A reference to a noname variable in the environment.
    /// Potentially narrowing it down to a range of cells in that variable.
    /// For example, `x[2]` would be represented with "x" and the range `(2, 1)`,
    /// if `x` is an array of `Field` elements.
    Ref {
        var_name: String,
        start: usize,
        len: usize,
    },
}

impl VarOrRef {
    fn constant(&self) -> Option<Field> {
        match self {
            VarOrRef::Var(var) => var.constant(),
            VarOrRef::Ref { .. } => None,
        }
    }

    /// Returns the value within the variable or pointer.
    /// If it is a pointer, we lose information about the original variable,
    /// thus calling this function is aking to passing the variable by value.
    pub fn value(self, env: &FnEnv, global_env: &GlobalEnv) -> Var {
        match self {
            VarOrRef::Var(var) => var,
            VarOrRef::Ref {
                var_name,
                start,
                len,
            } => {
                let var_info = env.get_var(global_env, &var_name);
                let cvars = var_info.var.range(start, len).to_vec();
                Var::new(cvars, var_info.var.span)
            }
        }
    }

    pub fn from_var_info(var_name: String, var_info: VarInfo) -> Self {
        if var_info.mutable {
            Self::Ref {
                var_name,
                start: 0,
                len: var_info.var.len(),
            }
        } else {
            Self::Var(var_info.var)
        }
    }

    fn narrow(&self, start: usize, len: usize) -> Self {
        match self {
            VarOrRef::Var(var) => {
                let cvars = var.range(start, len).to_vec();
                VarOrRef::Var(Var::new(cvars, var.span))
            }

            //      old_start
            //      |
            //      v
            // |----[-----------]-----| <-- var.cvars
            //       <--------->
            //         old_len
            //
            //
            //          start
            //          |
            //          v
            //      |---[-----]-|
            //           <--->
            //            len
            //
            VarOrRef::Ref {
                var_name,
                start: old_start,
                len: old_len,
            } => {
                // ensure that the new range is contained in the older range
                assert!(start < *old_len); // lower bound
                assert!(start + len <= *old_len); // upper bound
                assert!(len > 0); // empty range not allowed

                Self::Ref {
                    var_name: var_name.clone(),
                    start: old_start + start,
                    len,
                }
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            VarOrRef::Var(var) => var.len(),
            VarOrRef::Ref { len, .. } => *len,
        }
    }
}

//
// Circuit Writer (also used by witness generation)
//

#[derive(Default, Debug)]
pub struct CircuitWriter {
    /// The source code that created this circuit.
    /// Useful for debugging and displaying user errors.
    pub source: String,

    /// Once this is set, you can generate a witness (and can't modify the circuit?)
    pub finalized: bool,

    /// This is used to give a distinct number to each variable during circuit generation.
    pub next_variable: usize,

    /// This is how you compute the value of each variable during witness generation.
    /// It is created during circuit generation.
    pub witness_vars: HashMap<usize, Value>,

    /// The execution trace table with vars as placeholders.
    /// It is created during circuit generation,
    /// and used by the witness generator.
    pub rows_of_vars: Vec<Vec<Option<CellVar>>>,

    /// The gates created by the circuit generation.
    gates: Vec<Gate>,

    /// The wiring of the circuit.
    /// It is created during circuit generation.
    pub wiring: HashMap<usize, Wiring>,

    /// Size of the public input.
    pub public_input_size: usize,

    /// If a public output is set, this will be used to store its [Var].
    /// The public output generation works as follows:
    /// 1. This cvar is created and inserted in the circuit (gates) during compilation of the public input
    ///    (as the public output is the end of the public input)
    /// 2. When the `return` statement of the circuit is parsed,
    ///    it will set this `public_output` variable again to the correct vars.
    /// 3. During witness generation, the public output computation
    ///    is delayed until the very end.
    pub public_output: Option<Var>,

    /// Indexes used by the private inputs
    /// (this is useful to check that they appear in the circuit)
    pub private_input_indices: Vec<usize>,

    /// Used during the witness generation to check
    /// public and private inputs given by the prover.
    pub main: (FnSig, Span),
}

impl CircuitWriter {
    pub fn generate_circuit(tast: TAST, code: &str) -> Result<CompiledCircuit> {
        // if there's no main function, then return an error
        let TAST {
            ast,
            typed_global_env,
        } = tast;

        if !typed_global_env.has_main {
            return Err(Error::new(ErrorKind::NoMainFunction, Span::default()));
        }

        let (main_sig, main_span) = {
            let fn_info = typed_global_env.functions.get("main").cloned().unwrap();

            (fn_info.sig().clone(), fn_info.span)
        };

        let mut circuit_writer = CircuitWriter {
            source: code.to_string(),
            main: (main_sig, main_span),
            ..CircuitWriter::default()
        };

        // make sure we can't call that several times
        if circuit_writer.finalized {
            panic!("circuit already finalized (TODO: return a proper error");
        }

        let global_env = &mut GlobalEnv::new(typed_global_env);

        for root in &ast.0 {
            match &root.kind {
                // imports (already dealt with in type checker)
                RootKind::Use(_path) => (),

                // `const thing = 42;`
                RootKind::Const(cst) => {
                    global_env.add_constant(cst.name.value.clone(), cst.value, cst.span);
                }

                // `fn main() { ... }`
                RootKind::Function(function) => {
                    // create the env
                    let fn_env = &mut FnEnv::default();

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
                        let len = match &typ.kind {
                            TyKind::Field => 1,
                            TyKind::Array(typ, len) => {
                                if !matches!(**typ, TyKind::Field) {
                                    unimplemented!();
                                }
                                *len as usize
                            }
                            TyKind::Bool => 1,
                            typ => global_env.typed.size_of(typ),
                        };

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

                        // add argument variable to the ast env
                        let mutable = false; // TODO: should we add a mut keyword in arguments as well?
                        let var_info = VarInfo::new(var, mutable, Some(typ.kind.clone()));
                        fn_env.add_var(global_env, name.value.clone(), var_info);
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
                    circuit_writer.compile_main_function(global_env, fn_env, function)?;
                }

                // struct definition (already dealt with in type checker)
                RootKind::Struct(_struct) => (),

                // ignore comments
                // TODO: we could actually preserve the comment in the ASM!
                RootKind::Comment(_comment) => (),
            }
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

    /// Returns the compiled gates of the circuit.
    pub fn compiled_gates(&self) -> &[Gate] {
        if !self.finalized {
            panic!("Circuit not finalized yet!");
        }
        &self.gates
    }

    fn compile_stmt(
        &mut self,
        global_env: &GlobalEnv,
        fn_env: &mut FnEnv,
        stmt: &Stmt,
    ) -> Result<Option<VarOrRef>> {
        match &stmt.kind {
            StmtKind::Assign { mutable, lhs, rhs } => {
                // compute the rhs
                let rhs_var = self
                    .compute_expr(global_env, fn_env, rhs)?
                    .ok_or_else(|| Error::new(ErrorKind::CannotComputeExpression, stmt.span))?;

                // obtain the actual values
                let rhs_var = rhs_var.value(fn_env, global_env);

                let typ = global_env.typed.expr_type(rhs).cloned();
                let var_info = VarInfo::new(rhs_var, *mutable, typ);

                // store the new variable
                // TODO: do we really need to store that in the scope? That's not an actual var in the scope that's an internal var...
                fn_env.add_var(global_env, lhs.value.clone(), var_info);
            }

            StmtKind::ForLoop { var, range, body } => {
                for ii in range.range() {
                    fn_env.nest();

                    let cst_var = Var::new_constant(ii.into(), var.span);
                    let var_info = VarInfo::new(cst_var, false, Some(TyKind::Field));
                    fn_env.add_var(global_env, var.value.clone(), var_info);

                    self.compile_block(global_env, fn_env, body)?;

                    fn_env.pop();
                }
            }
            StmtKind::Expr(expr) => {
                // compute the expression
                let var = self.compute_expr(global_env, fn_env, expr)?;

                // make sure it does not return any value.
                assert!(var.is_none());
            }
            StmtKind::Return(expr) => {
                let var = self
                    .compute_expr(global_env, fn_env, expr)?
                    .ok_or_else(|| Error::new(ErrorKind::CannotComputeExpression, stmt.span))?;

                // we already checked in type checking that this is not an early return
                return Ok(Some(var));
            }
            StmtKind::Comment(_) => (),
        }

        Ok(None)
    }

    /// might return something?
    fn compile_block(
        &mut self,
        global_env: &GlobalEnv,
        fn_env: &mut FnEnv,
        stmts: &[Stmt],
    ) -> Result<Option<Var>> {
        fn_env.nest();
        for stmt in stmts {
            let res = self.compile_stmt(global_env, fn_env, stmt)?;
            if let Some(var) = res {
                // a block doesn't return a pointer, only values
                let var = var.value(fn_env, global_env);

                // we already checked for early returns in type checking
                return Ok(Some(var));
            }
        }
        fn_env.pop();
        Ok(None)
    }

    fn compile_native_function_call(
        &mut self,
        global_env: &GlobalEnv,
        function: &Function,
        args: Vec<VarInfo>,
    ) -> Result<Option<Var>> {
        assert!(!function.is_main());

        // create new fn_env
        let fn_env = &mut FnEnv::new();

        // set arguments
        assert_eq!(function.sig.arguments.len(), args.len());

        for (name, var_info) in function.sig.arguments.iter().zip(args) {
            fn_env.add_var(global_env, name.name.value.clone(), var_info);
        }

        // compile it and potentially return a return value
        self.compile_block(global_env, fn_env, &function.body)
    }

    /// Compile a function. Used to compile `main()` only for now
    fn compile_main_function(
        &mut self,
        global_env: &GlobalEnv,
        fn_env: &mut FnEnv,
        function: &Function,
    ) -> Result<()> {
        assert!(function.is_main());

        // compile the block
        let returned = self.compile_block(global_env, fn_env, &function.body)?;

        // we're expecting something returned?
        match (function.sig.return_type.as_ref(), returned) {
            (None, None) => Ok(()),
            (Some(expected), None) => Err(Error::new(ErrorKind::MissingReturn, expected.span)),
            (None, Some(returned)) => Err(Error::new(ErrorKind::UnexpectedReturn, returned.span)),
            (Some(_expected), Some(returned)) => {
                // make sure there are no constants in the returned value
                let mut returned_cells = vec![];
                for r in &returned.cvars {
                    match r {
                        ConstOrCell::Cell(c) => returned_cells.push(c),
                        ConstOrCell::Const(_) => {
                            return Err(Error::new(ErrorKind::ConstantInOutput, returned.span))
                        }
                    }
                }

                // store the return value in the public input that was created for that ^
                let public_output = self
                    .public_output
                    .as_ref()
                    .expect("bug in the compiler: missing public output");

                for (pub_var, ret_var) in public_output.cvars.iter().zip(returned_cells) {
                    // replace the computation of the public output vars with the actual variables being returned here
                    let var_idx = pub_var.idx().unwrap();
                    let prev = self
                        .witness_vars
                        .insert(var_idx, Value::PublicOutput(Some(*ret_var)));
                    assert!(prev.is_some());
                }

                Ok(())
            }
        }
    }

    pub fn asm(&self, debug: bool) -> String {
        asm::generate_asm(&self.source, &self.gates, &self.wiring, debug)
    }

    pub fn new_internal_var(&mut self, val: Value, span: Span) -> CellVar {
        // create new var
        let var = CellVar::new(self.next_variable, span);
        self.next_variable += 1;

        // store it in the circuit_writer
        self.witness_vars.insert(var.index, val);

        var
    }

    fn compute_expr(
        &mut self,
        global_env: &GlobalEnv,
        fn_env: &mut FnEnv,
        expr: &Expr,
    ) -> Result<Option<VarOrRef>> {
        match &expr.kind {
            // `module::fn_name(args)`
            ExprKind::FnCall {
                module,
                fn_name,
                args,
            } => {
                // compute the arguments
                // module::fn_name(args)
                //                 ^^^^
                let mut vars = Vec::with_capacity(args.len());
                for arg in args {
                    // get the variable behind the expression
                    let var = self
                        .compute_expr(global_env, fn_env, arg)?
                        .ok_or_else(|| Error::new(ErrorKind::CannotComputeExpression, arg.span))?;

                    // we pass variables by values always
                    let var = var.value(fn_env, global_env);

                    let typ = global_env.typed.expr_type(arg).cloned();
                    let mutable = false; // TODO: mut keyword in arguments?
                    let var_info = VarInfo::new(var, mutable, typ);

                    vars.push(var_info);
                }

                // retrieve the function in the env
                if let Some(module) = module {
                    // module::fn_name(args)
                    // ^^^^^^
                    let module = global_env.typed.modules.get(&module.value).ok_or_else(|| {
                        Error::new(
                            ErrorKind::UndefinedModule(module.value.clone()),
                            module.span,
                        )
                    })?;
                    let fn_info = module.functions.get(&fn_name.value).ok_or_else(|| {
                        Error::new(
                            ErrorKind::UndefinedFunction(fn_name.value.clone()),
                            fn_name.span,
                        )
                    })?;

                    match &fn_info.kind {
                        FnKind::BuiltIn(_, handle) => {
                            let res = handle(self, &vars, expr.span);
                            res.map(|r| r.map(VarOrRef::Var))
                        }
                        FnKind::Native(_) => todo!(),
                        FnKind::Main(_) => Err(Error::new(ErrorKind::RecursiveMain, expr.span)),
                    }
                } else {
                    // fn_name(args)
                    // ^^^^^^^
                    let fn_info =
                        global_env
                            .typed
                            .functions
                            .get(&fn_name.value)
                            .ok_or_else(|| {
                                Error::new(
                                    ErrorKind::UndefinedFunction(fn_name.value.clone()),
                                    fn_name.span,
                                )
                            })?;

                    match &fn_info.kind {
                        FnKind::BuiltIn(_sig, handle) => {
                            let res = handle(self, &vars, expr.span);
                            res.map(|r| r.map(VarOrRef::Var))
                        }
                        FnKind::Native(func) => {
                            let res = self.compile_native_function_call(global_env, func, vars);
                            res.map(|r| r.map(VarOrRef::Var))
                        }
                        FnKind::Main(_) => Err(Error::new(ErrorKind::RecursiveMain, expr.span)),
                    }
                }
            }

            ExprKind::FieldAccess { lhs, rhs } => {
                // get var behind lhs
                let lhs_var = self
                    .compute_expr(global_env, fn_env, lhs)?
                    .ok_or_else(|| Error::new(ErrorKind::CannotComputeExpression, lhs.span))?;

                // get struct info behind lhs
                let lhs_struct = global_env
                    .typed
                    .expr_type(lhs)
                    .ok_or_else(|| Error::new(ErrorKind::CannotComputeExpression, lhs.span))?;

                let self_struct = match lhs_struct {
                    TyKind::Custom(s) => s,
                    _ => {
                        panic!("could not figure out struct implementing that method call")
                    }
                };

                let struct_info = global_env
                    .typed
                    .struct_info(self_struct)
                    .expect("struct info not found for custom struct");

                // find range of field
                let mut start = 0;
                let mut len = 0;
                for (field, field_typ) in &struct_info.fields {
                    if field == &rhs.value {
                        len = global_env.typed.size_of(field_typ);
                        break;
                    }

                    start += global_env.typed.size_of(field_typ);
                }

                // narrow the variable to the given range
                let var = lhs_var.narrow(start, len);
                Ok(Some(var))
            }

            // `Thing.method(args)` or `thing.method(args)`
            ExprKind::MethodCall {
                lhs,
                method_name,
                args,
            } => {
                // figure out the name of the custom struct
                let lhs_typ = global_env
                    .typed
                    .expr_type(lhs)
                    .expect("method call on what?");

                let struct_name = match lhs_typ {
                    TyKind::Custom(c) => c,
                    _ => panic!("method call only work on custom types"),
                };

                // get var of `self`
                // (might be `None` if it's a static method call)
                let self_var = self.compute_expr(global_env, fn_env, lhs)?;

                // find method info
                let struct_info = global_env
                    .typed
                    .struct_info(struct_name)
                    .expect("could not find struct info");

                let func = struct_info
                    .methods
                    .get(&method_name.value)
                    .expect("could not find method");

                // if method has a `self` argument, manually add it to the list of argument
                let mut vars = vec![];
                if let Some(first_arg) = func.sig.arguments.first() {
                    if first_arg.name.value == "self" {
                        let self_var = self_var.ok_or_else(|| {
                            Error::new(ErrorKind::NotAStaticMethod, method_name.span)
                        })?;

                        // TODO: for now we pass `self` by value as well
                        let mutable = false;
                        let self_var = self_var.value(fn_env, global_env);

                        let self_var_info = VarInfo::new(self_var, mutable, Some(lhs_typ.clone()));
                        vars.insert(0, self_var_info);
                    }
                } else {
                    assert!(self_var.is_none());
                }

                // compute the arguments
                for arg in args {
                    let var = self
                        .compute_expr(global_env, fn_env, arg)?
                        .ok_or_else(|| Error::new(ErrorKind::CannotComputeExpression, arg.span))?;

                    // TODO: for now we pass `self` by value as well
                    let mutable = false;
                    let var = var.value(fn_env, global_env);

                    let typ = global_env.typed.expr_type(arg).cloned();
                    let var_info = VarInfo::new(var, mutable, typ);

                    vars.push(var_info);
                }

                // execute method
                let res = self.compile_native_function_call(global_env, func, vars);
                res.map(|r| r.map(VarOrRef::Var))
            }

            ExprKind::IfElse { cond, then_, else_ } => {
                let cond = self
                    .compute_expr(global_env, fn_env, cond)?
                    .unwrap()
                    .value(fn_env, global_env);
                let then_ = self
                    .compute_expr(global_env, fn_env, then_)?
                    .unwrap()
                    .value(fn_env, global_env);
                let else_ = self
                    .compute_expr(global_env, fn_env, else_)?
                    .unwrap()
                    .value(fn_env, global_env);

                let res = field::if_else(self, &cond, &then_, &else_, expr.span);

                Ok(Some(VarOrRef::Var(res)))
            }

            ExprKind::Assignment { lhs, rhs } => {
                // figure out the local var  of lhs
                let lhs = self.compute_expr(global_env, fn_env, lhs)?.unwrap();

                // figure out the var of what's on the right
                let rhs = self.compute_expr(global_env, fn_env, rhs)?.unwrap();
                let rhs_var = match rhs {
                    VarOrRef::Var(var) => var,
                    VarOrRef::Ref {
                        var_name,
                        start,
                        len,
                    } => {
                        let var_info = fn_env.get_var(global_env, &var_name);
                        let cvars = var_info.var.range(start, len).to_vec();
                        Var::new(cvars, var_info.var.span)
                    }
                };

                // replace the left with the right
                match lhs {
                    VarOrRef::Var(_) => panic!("can't reassign this non-mutable variable"),
                    VarOrRef::Ref {
                        var_name,
                        start,
                        len,
                    } => {
                        fn_env.reassign_var_range(&var_name, rhs_var, start, len);
                    }
                }

                Ok(None)
            }

            ExprKind::BinaryOp { op, lhs, rhs, .. } => {
                let lhs = self.compute_expr(global_env, fn_env, lhs)?.unwrap();
                let rhs = self.compute_expr(global_env, fn_env, rhs)?.unwrap();

                let lhs = lhs.value(fn_env, global_env);
                let rhs = rhs.value(fn_env, global_env);

                let res = match op {
                    Op2::Addition => field::add(self, &lhs[0], &rhs[0], expr.span),
                    Op2::Subtraction => field::sub(self, &lhs[0], &rhs[0], expr.span),
                    Op2::Multiplication => field::mul(self, &lhs[0], &rhs[0], expr.span),
                    Op2::Equality => field::equal(self, &lhs, &rhs, expr.span),
                    Op2::BoolAnd => boolean::and(self, &lhs[0], &rhs[0], expr.span),
                    Op2::BoolOr => boolean::or(self, &lhs[0], &rhs[0], expr.span),
                    Op2::Division => todo!(),
                };

                Ok(Some(VarOrRef::Var(res)))
            }

            ExprKind::Negated(b) => {
                let var = self.compute_expr(global_env, fn_env, b)?.unwrap();

                let var = var.value(fn_env, global_env);

                let res = boolean::not(self, &var[0], expr.span);
                Ok(Some(VarOrRef::Var(res)))
            }

            ExprKind::BigInt(b) => {
                let biguint = BigUint::from_str_radix(b, 10).expect("failed to parse number.");
                let ff = Field::try_from(biguint).map_err(|_| {
                    Error::new(ErrorKind::CannotConvertToField(b.to_string()), expr.span)
                })?;

                let res = VarOrRef::Var(Var::new_constant(ff, expr.span));
                Ok(Some(res))
            }

            ExprKind::Bool(b) => {
                let value = if *b { Field::one() } else { Field::zero() };
                let res = VarOrRef::Var(Var::new_constant(value, expr.span));
                Ok(Some(res))
            }

            ExprKind::Variable { module, name } => {
                if module.is_some() {
                    panic!("accessing module variables not supported yet");
                }

                if is_type(&name.value) {
                    // if it's a type we return nothing
                    // (most likely what follows is a static method call)
                    Ok(None)
                } else {
                    let var_info = fn_env.get_var(global_env, &name.value);

                    let res = VarOrRef::from_var_info(name.value.clone(), var_info);
                    Ok(Some(res))
                }
            }

            ExprKind::ArrayAccess { array, idx } => {
                // retrieve var of array
                let var = self
                    .compute_expr(global_env, fn_env, array)?
                    .expect("array access on non-array");

                // compute the index
                let idx_var = self
                    .compute_expr(global_env, fn_env, idx)?
                    .ok_or_else(|| Error::new(ErrorKind::CannotComputeExpression, expr.span))?;
                let idx = idx_var
                    .constant()
                    .ok_or_else(|| Error::new(ErrorKind::ExpectedConstant, expr.span))?;
                let idx: BigUint = idx.into();
                let idx: usize = idx.try_into().unwrap();

                // retrieve the type of the elements in the array
                let array_typ = global_env
                    .typed
                    .expr_type(array)
                    .expect("cannot find type of array");

                let elem_type = match array_typ {
                    TyKind::Array(ty, array_len) => {
                        if idx >= (*array_len as usize) {
                            return Err(Error::new(
                                ErrorKind::ArrayIndexOutOfBounds(idx, *array_len as usize - 1),
                                expr.span,
                            ));
                        }
                        ty
                    }
                    _ => panic!("expected array"),
                };

                // compute the size of each element in the array
                let len = global_env.typed.size_of(elem_type);

                // compute the real index
                let start = idx * len;

                // out-of-bound checks
                if start >= var.len() || start + len > var.len() {
                    return Err(Error::new(
                        ErrorKind::ArrayIndexOutOfBounds(start, var.len()),
                        expr.span,
                    ));
                }

                // index into the var
                let var = var.narrow(start, len);

                //
                Ok(Some(var))
            }

            ExprKind::ArrayDeclaration(items) => {
                let mut cvars = vec![];

                for item in items {
                    let var = self.compute_expr(global_env, fn_env, item)?.unwrap();
                    let to_extend = var.value(fn_env, global_env).cvars.clone();
                    cvars.extend(to_extend);
                }

                let var = VarOrRef::Var(Var::new(cvars, expr.span));

                Ok(Some(var))
            }

            ExprKind::CustomTypeDeclaration {
                struct_name: _,
                fields,
            } => {
                // create the struct by just concatenating all of its cvars
                let mut cvars = vec![];
                for (_field, rhs) in fields {
                    let var = self.compute_expr(global_env, fn_env, rhs)?.unwrap();
                    let to_extend = var.value(fn_env, global_env).cvars.clone();
                    cvars.extend(to_extend);
                }
                let var = VarOrRef::Var(Var::new(cvars, expr.span));

                //
                Ok(Some(var))
            }
        }
    }

    // TODO: dead code?
    pub fn compute_constant(&self, var: CellVar, span: Span) -> Result<Field> {
        match &self.witness_vars.get(&var.index) {
            Some(Value::Constant(c)) => Ok(*c),
            Some(Value::LinearCombination(lc, cst)) => {
                let mut res = *cst;
                for (coeff, var) in lc {
                    res += self.compute_constant(*var, span)? * *coeff;
                }
                Ok(res)
            }
            Some(Value::Mul(lhs, rhs)) => {
                let lhs = self.compute_constant(*lhs, span)?;
                let rhs = self.compute_constant(*rhs, span)?;
                Ok(lhs * rhs)
            }
            _ => Err(Error::new(ErrorKind::ExpectedConstant, span)),
        }
    }

    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }

    // TODO: we should cache constants to avoid creating a new variable for each constant
    /// This should be called only when you want to constrain a constant for real.
    /// Gates that handle constants should always make sure to call this function when they want them constrained.
    pub fn add_constant(
        &mut self,
        label: Option<&'static str>,
        value: Field,
        span: Span,
    ) -> CellVar {
        let var = self.new_internal_var(Value::Constant(value), span);

        let zero = Field::zero();
        self.add_gate(
            label.unwrap_or("hardcode a constant"),
            GateKind::DoubleGeneric,
            vec![Some(var)],
            vec![Field::one(), zero, zero, zero, value.neg()],
            span,
        );

        var
    }

    /// creates a new gate, and the associated row in the witness/execution trace.
    // TODO: add_gate instead of gates?
    pub fn add_gate(
        &mut self,
        note: &'static str,
        typ: GateKind,
        vars: Vec<Option<CellVar>>,
        coeffs: Vec<Field>,
        span: Span,
    ) {
        // sanitize
        assert!(coeffs.len() <= NUM_REGISTERS);
        assert!(vars.len() <= NUM_REGISTERS);

        // construct the execution trace with vars, for the witness generation
        self.rows_of_vars.push(vars.clone());

        // get current row
        // important: do that before adding the gate below
        let row = self.gates.len();

        // add gate
        self.gates.push(Gate {
            typ,
            coeffs,
            span,
            note,
        });

        // wiring (based on vars)
        for (col, var) in vars.iter().enumerate() {
            if let Some(var) = var {
                let curr_cell = Cell { row, col };
                self.wiring
                    .entry(var.index)
                    .and_modify(|w| match w {
                        Wiring::NotWired(cell) => {
                            *w = Wiring::Wired(vec![(*cell, var.span), (curr_cell, span)])
                        }
                        Wiring::Wired(ref mut cells) => {
                            cells.push((curr_cell, span));
                        }
                    })
                    .or_insert(Wiring::NotWired(curr_cell));
            }
        }
    }

    pub fn add_public_inputs(&mut self, name: String, num: usize, span: Span) -> Var {
        let mut cvars = Vec::with_capacity(num);

        for idx in 0..num {
            // create the var
            let cvar = self.new_internal_var(Value::External(name.clone(), idx), span);
            cvars.push(ConstOrCell::Cell(cvar));

            // create the associated generic gate
            self.add_gate(
                "add public input",
                GateKind::DoubleGeneric,
                vec![Some(cvar)],
                vec![Field::one()],
                span,
            );
        }

        self.public_input_size += num;

        Var::new(cvars, span)
    }

    pub fn add_public_outputs(&mut self, num: usize, span: Span) {
        assert!(self.public_output.is_none());

        let mut cvars = Vec::with_capacity(num);
        for _ in 0..num {
            // create the var
            let cvar = self.new_internal_var(Value::PublicOutput(None), span);
            cvars.push(ConstOrCell::Cell(cvar));

            // create the associated generic gate
            self.add_gate(
                "add public output",
                GateKind::DoubleGeneric,
                vec![Some(cvar)],
                vec![Field::one()],
                span,
            );
        }
        self.public_input_size += num;

        // store it
        let res = Var::new(cvars, span);
        self.public_output = Some(res);
    }

    pub fn add_private_inputs(&mut self, name: String, num: usize, span: Span) -> Var {
        let mut cvars = Vec::with_capacity(num);

        for idx in 0..num {
            // create the var
            let cvar = self.new_internal_var(Value::External(name.clone(), idx), span);
            cvars.push(ConstOrCell::Cell(cvar));
            self.private_input_indices.push(cvar.index);
        }

        Var::new(cvars, span)
    }
}

//
// Local Environment
//

/// Is used to store functions' scoped variables.
/// This include inputs and output of the function,  as well as local variables.
/// You can think of it as a function call stack.
#[derive(Default, Debug, Clone)]
pub struct FnEnv {
    /// The current nesting level.
    /// Starting at 0 (top level), and increasing as we go into a block.
    current_scope: usize,

    /// Used by the private and public inputs,
    /// and any other external variables created in the circuit
    /// This needs to be garbage collected when we exit a scope.
    /// Note: The `usize` is the scope in which the variable was created.
    vars: HashMap<String, (usize, VarInfo)>,
}

/// Information about a variable.
#[derive(Debug, Clone)]
pub struct VarInfo {
    /// The variable.
    pub var: Var,

    // TODO: we could also store this on the expression node... but I think this is lighter
    pub mutable: bool,

    /// We keep track of the type of variables, eventhough we're not in the typechecker anymore,
    /// because we need to know the type for method calls.
    // TODO: why is this an option?
    pub typ: Option<TyKind>,
}

impl VarInfo {
    pub fn new(var: Var, mutable: bool, typ: Option<TyKind>) -> Self {
        Self { var, mutable, typ }
    }

    pub fn reassign(&self, var: Var) -> Self {
        Self {
            var,
            mutable: self.mutable,
            typ: self.typ.clone(),
        }
    }

    pub fn reassign_range(&self, var: Var, start: usize, len: usize) -> Self {
        // sanity check
        assert_eq!(var.len(), len);

        // create new cvars by modifying a specific range
        let mut cvars = self.var.cvars.clone();
        let cvars_range = &mut cvars[start..start + len];
        cvars_range.copy_from_slice(&var.cvars);

        let var = Var::new(cvars, self.var.span);

        Self {
            var,
            mutable: self.mutable,
            typ: self.typ.clone(),
        }
    }
}

impl FnEnv {
    /// Creates a new FnEnv
    pub fn new() -> Self {
        Self::default()
    }

    /// Enters a scoped block.
    pub fn nest(&mut self) {
        self.current_scope += 1;
    }

    /// Exits a scoped block.
    pub fn pop(&mut self) {
        self.current_scope = self.current_scope.checked_sub(1).expect("scope bug");

        // remove variables as we exit the scope
        // (we don't need to keep them around to detect shadowing,
        // as we already did that in type checker)
        let mut to_delete = vec![];
        for (name, (scope, _)) in self.vars.iter() {
            if *scope > self.current_scope {
                to_delete.push(name.clone());
            }
        }
        for d in to_delete {
            self.vars.remove(&d);
        }
    }

    /// Returns true if a scope is a prefix of our scope.
    pub fn is_in_scope(&self, prefix_scope: usize) -> bool {
        self.current_scope >= prefix_scope
    }

    /// Stores type information about a local variable.
    /// Note that we forbid shadowing at all scopes.
    pub fn add_var(&mut self, global_env: &GlobalEnv, var_name: String, var_info: VarInfo) {
        if global_env.get_constant(&var_name).is_some() {
            panic!("cannot shadow global variable {}", var_name);
        }

        let scope = self.current_scope;

        if self
            .vars
            .insert(var_name.clone(), (scope, var_info))
            .is_some()
        {
            panic!("type checker error: var {var_name} already exists");
        }
    }

    /// Retrieves type information on a variable, given a name.
    /// If the variable is not in scope, return false.
    // TODO: return an error no?
    pub fn get_var(&self, global_env: &GlobalEnv, var_name: &str) -> VarInfo {
        // look for global constants first
        if let Some(var_info) = global_env.get_constant(var_name) {
            return var_info.clone();
        }

        // if not found, then look into local variables
        let (scope, var_info) = self
            .vars
            .get(var_name)
            .unwrap_or_else(|| panic!("type checking bug: local variable {var_name} not found"));
        if !self.is_in_scope(*scope) {
            panic!("type checking bug: local variable not in scope");
        }

        var_info.clone()
    }

    pub fn reassign_var(&mut self, var_name: &str, var: Var) {
        // get the scope first, we don't want to modify that
        let (scope, var_info) = self
            .vars
            .get(var_name)
            .expect("type checking bug: local variable for reassigning not found");

        if !self.is_in_scope(*scope) {
            panic!("type checking bug: local variable for reassigning not in scope");
        }

        if !var_info.mutable {
            panic!("type checking bug: local variable for reassigning is not mutable");
        }

        let var_info = var_info.reassign(var);
        self.vars.insert(var_name.to_string(), (*scope, var_info));
    }

    /// Same as [Self::reassign_var], but only reassigns a specific range of the variable.
    pub fn reassign_var_range(&mut self, var_name: &str, var: Var, start: usize, len: usize) {
        // get the scope first, we don't want to modify that
        let (scope, var_info) = self
            .vars
            .get(var_name)
            .expect("type checking bug: local variable for reassigning not found");

        if !self.is_in_scope(*scope) {
            panic!("type checking bug: local variable for reassigning not in scope");
        }

        if !var_info.mutable {
            panic!("type checking bug: local variable for reassigning is not mutable");
        }

        let var_info = var_info.reassign_range(var, start, len);
        self.vars.insert(var_name.to_string(), (*scope, var_info));
    }
}
