use std::fmt::{self, Display, Formatter};

use ark_ff::{One, Zero};
use kimchi::circuits::wires::Wire;
use num_bigint::BigUint;
use num_traits::Num as _;
use serde::{Deserialize, Serialize};

use crate::{
    backends::{kimchi::VestaField, Backend},
    circuit_writer::{CircuitWriter, DebugInfo, FnEnv, VarInfo},
    constants::Span,
    constraints::{boolean, field},
    error::{ErrorKind, Result},
    imports::FnKind,
    parser::{
        types::{FunctionDef, Stmt, StmtKind, TyKind},
        Expr, ExprKind, Op2,
    },
    syntax::is_type,
    type_checker::FullyQualified,
    var::{ConstOrCell, Value, Var, VarOrRef},
};

//
// Data structures
//

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GateKind {
    Zero,
    DoubleGeneric,
    Poseidon,
}

impl From<GateKind> for kimchi::circuits::gate::GateType {
    fn from(gate_kind: GateKind) -> Self {
        use kimchi::circuits::gate::GateType::*;
        match gate_kind {
            GateKind::Zero => Zero,
            GateKind::DoubleGeneric => Generic,
            GateKind::Poseidon => Poseidon,
        }
    }
}

// TODO: this could also contain the span that defined the gate!
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gate {
    /// Type of gate
    pub typ: GateKind,

    /// Coefficients
    #[serde(skip)]
    pub coeffs: Vec<VestaField>,
}

impl Gate {
    pub fn to_kimchi_gate(&self, row: usize) -> kimchi::circuits::gate::CircuitGate<VestaField> {
        kimchi::circuits::gate::CircuitGate {
            typ: self.typ.into(),
            wires: Wire::for_row(row),
            coeffs: self.coeffs.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Cell {
    pub row: usize,
    pub col: usize,
}

impl Display for Cell {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({},{})", self.row, self.col)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Wiring {
    /// Not yet wired (just indicates the position of the cell itself)
    NotWired(AnnotatedCell),
    /// The wiring (associated to different spans)
    Wired(Vec<AnnotatedCell>),
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq)]
pub struct AnnotatedCell {
    pub(crate) cell: Cell,
    pub(crate) debug: DebugInfo,
}

impl PartialEq for AnnotatedCell {
    fn eq(&self, other: &Self) -> bool {
        self.cell == other.cell
    }
}

impl PartialOrd for AnnotatedCell {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.cell.partial_cmp(&other.cell)
    }
}

impl Ord for AnnotatedCell {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cell.cmp(&other.cell)
    }
}

/// This records the steps to access a child element of a struct or an array variable.
/// These steps are used to navigate through a variable in the scope in order to update its value.
/// The access pattern can narrow down to either a single field, struct or array. For example:
/// `houses[1].rooms[2].room_size`
///
/// The access to the field called "room_size" at the end can be represented as:
/// Access {
///   var_name: "houses", // the name of the variable in the current scope
///   steps: [ Array(1), Field("rooms"), Array(2), Field("room_size") ],
///   expr: ...           // represent the expression that leads to the value of "room_size"
/// }
///
/// where the "Array" / "Field" enum represents the `AccessKind` of a step.
///
/// This Access type is currently constructed from `compute_expr` cases:
/// - Expr::Variable
/// - Expr::FieldAccess
/// - Expr::ArrayAccess
///
/// where `Expr::Variable` case initializes the Access with an empty steps.
/// `Expr::FieldAccess` or `Expr::ArrayAccess` constructs the steps.
///
/// Then the `Expr::Assignment` uses the Access to trace and update the value of the targeted variable in the scope.
#[derive(Debug, Clone)]
pub struct Access<F, V>
where
    F: BackendField,
    V: BackendVar,
{
    pub var_name: String,
    pub steps: Vec<AccessKind>,
    // the current access node
    pub expr: Box<ComputedExpr<F, V>>,
}

impl<F: BackendField, V: BackendVar> Access<F, V> {
    pub fn new(var_name: &str, steps: &[AccessKind], expr: Box<ComputedExpr<F, V>>) -> Self {
        Self {
            var_name: var_name.to_string(),
            steps: steps.to_vec(),
            expr,
        }
    }
}

impl<F: BackendField, V: BackendVar> Display for Access<F, V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut s = self.var_name.clone();
        s.push_str(
            &self
                .steps
                .iter()
                .map(ToString::to_string)
                .collect_vec()
                .join(""),
        );

        write!(f, "{}", s)
    }
}

/// This represents which kind of access for a step.
#[derive(Debug, Clone)]
pub enum AccessKind {
    /// Access to a field of a struct
    Field(String),
    /// Access to an array element at a specific index
    Array(usize),
}

impl Display for AccessKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            AccessKind::Field(field) => write!(f, ".{}", field),
            AccessKind::Array(index) => write!(f, "[{}]", index),
        }
    }
}

/// Represents a computed expression from a `Expr`.
/// This is useful to propagate the computed values from the call to `compute_expr`,
/// while retaining the structural information of a computed expression.
#[derive(Debug, Clone)]
pub struct ComputedExpr<F, V>
where
    F: BackendField,
    V: BackendVar,
{
    kind: ComputedExprKind<F, V>,
    span: Span,
}

#[derive(Debug, Clone)]
pub enum ComputedExprKind<F, V>
where
    F: BackendField,
    V: BackendVar,
{
    /// Structures behind a custom struct can be recursive, so it embeds the ComputExpr.
    Struct(BTreeMap<String, ComputedExpr<F, V>>),
    /// Structures behind an array can be recursive, so it embeds the ComputExpr.
    Array(Vec<ComputedExpr<F, V>>),
    Bool(Var<F, V>),
    Field(Var<F, V>),
    /// Access to a variable in the scope.
    Access(Access<F, V>),
    /// Represents the results of a builtin function call.
    /// Because we don't know the exact type of the result, we store it as a Var.
    /// We may deprecate this once it is able to type check the builtin functions,
    /// so that the result can be inferred.
    FnCallResult(Var<F, V>),
}

impl<F: BackendField, V: BackendVar> ComputedExpr<F, V> {
    pub fn new(kind: ComputedExprKind<F, V>, span: Span) -> Self {
        Self { kind, span }
    }

    /// Create a new `ComputedExpr` from a struct
    pub fn new_struct(fields: BTreeMap<String, ComputedExpr<F, V>>, span: Span) -> Self {
        Self::new(ComputedExprKind::Struct(fields), span)
    }

    /// Create a new `ComputedExpr` from an array
    pub fn new_array(array: Vec<ComputedExpr<F, V>>, span: Span) -> Self {
        Self::new(ComputedExprKind::Array(array), span)
    }

    /// Create a new `ComputedExpr` from a boolean variable
    pub fn new_bool(var: Var<F, V>, span: Span) -> Self {
        Self::new(ComputedExprKind::Bool(var), span)
    }

    /// Create a new `ComputedExpr` from a field variable
    pub fn new_field(var: Var<F, V>, span: Span) -> Self {
        Self::new(ComputedExprKind::Field(var), span)
    }

    /// Create a new `ComputedExpr` from a function call result
    pub fn new_fn_call_result(var: Var<F, V>, span: Span) -> Self {
        Self::new(ComputedExprKind::FnCallResult(var), span)
    }

    /// Create a new `ComputedExpr` from an access
    pub fn new_access(access: Access<F, V>, span: Span) -> Self {
        Self::new(ComputedExprKind::Access(access), span)
    }

    /// Get the underlying value as `Var` of the computed expression.
    pub fn value(self) -> Var<F, V> {
        match self.kind {
            ComputedExprKind::Array(array) => {
                let mut cvars = vec![];
                for elm in array {
                    // unfold an element of the array
                    let var = elm.value();
                    cvars.extend(var.cvars);
                }
                Var::new(cvars, self.span)
            }
            ComputedExprKind::Struct(fields) => {
                let mut cvars = vec![];
                for (_, el) in fields {
                    // unfold a field of the struct
                    let var = el.value();
                    cvars.extend(var.cvars);
                }
                Var::new(cvars, self.span)
            }
            ComputedExprKind::Bool(var) => var,
            ComputedExprKind::Field(var) => var,
            ComputedExprKind::FnCallResult(var) => var,
            ComputedExprKind::Access(access) => access.expr.value(),
        }
    }

    /// Get the underlying value as a constant
    pub fn constant(&self) -> Option<F> {
        match &self.access_inner().kind {
            ComputedExprKind::Field(var) => var.constant(),
            ComputedExprKind::FnCallResult(var) => var.constant(),
            _ => None,
        }
    }

    /// Get the underlying value as a struct
    pub fn struct_expr(&self) -> Option<BTreeMap<String, ComputedExpr<F, V>>> {
        match &self.access_inner().kind {
            ComputedExprKind::Struct(fields) => Some(fields.clone()),
            _ => None,
        }
    }

    /// Get the underlying value as an array
    pub fn array_expr(&self) -> Option<Vec<ComputedExpr<F, V>>> {
        match &self.access_inner().kind {
            ComputedExprKind::Array(array) => Some(array.clone()),
            ComputedExprKind::FnCallResult(var) => {
                // convert cvars to a vector of ComputExpr
                let mut array = vec![];
                for cvar in &var.cvars {
                    let var = Var::new_cvar(cvar.clone(), self.span);
                    let ce = ComputedExpr::new_field(var, self.span);
                    array.push(ce);
                }

                Some(array)
            }
            _ => None,
        }
    }

    /// Expects the current expression to be an access type of `ComputedExpr`
    pub fn access_expr(&self) -> Option<Access<F, V>> {
        match &self.kind {
            ComputedExprKind::Access(access) => Some(access.clone()),
            _ => None,
        }
    }

    /// Recursively unwrap the access type of `ComputedExpr` until it is not an access type.
    /// An access type can represents a variable that is made up by multiple different variables.
    /// Each variable can be represented by an access type.
    /// For example:
    /// ...
    /// let house2 = House { rooms: [] };
    /// let town1 = [house2];
    /// let town2 = [house1, town1[0], house3];
    ///
    /// Then when it tries to have access to town2[1],
    /// the value will be:
    /// Access {
    ///  var_name: "town1",
    ///  steps: [...],
    ///  expr: Access {
    ///   var_name: "house2",
    ///   steps: [],
    ///  }
    /// }
    ///
    /// This is because the town2[1] is initiated from the variable town[1], which eventually points to a variable `Access` to house2.
    /// In general, we only care about the value behind the `Access` type.
    /// Currently the info behind `Access` type is useful for updating the value of the variable in the scope,
    /// such as `ExprKind::Assignment`.
    pub fn access_inner(&self) -> &ComputedExpr<F, V> {
        match &self.kind {
            ComputedExprKind::Access(access) => {
                let e = &access.expr;
                match &e.kind {
                    ComputedExprKind::Access(next_access) => next_access.expr.access_inner(),
                    _ => e,
                }
            }
            // otherwise, return the current expr
            _ => self,
        }
    }

    // todo: refactor expr to be &mut self
    /// Uses the access steps to trace the target element and update its value.
    pub fn update_computed_expr(
        expr: &mut ComputedExpr<F, V>,
        steps: &[AccessKind],
        new_expr: ComputedExpr<F, V>,
    ) -> Result<()> {
        // if there are no more accesses, replace the current expression
        if steps.is_empty() {
            *expr = new_expr;
            return Ok(());
        }

        // skip the access kind expr
        // this is similar to the access_inner method
        if let ComputedExprKind::Access(access) = &mut expr.kind {
            return Self::update_computed_expr(&mut access.expr, steps, new_expr);
        }

        let next_access = &steps[0];
        let remaining_accesses = &steps[1..];

        match &mut expr.kind {
            ComputedExprKind::Struct(fields) => {
                match next_access {
                    AccessKind::Field(field_name) => {
                        match fields.get_mut(field_name) {
                            Some(field_expr) => {
                                // stepping into the struct field
                                Self::update_computed_expr(field_expr, remaining_accesses, new_expr)
                            }
                            None => Err(Error::new(
                                "constraint-generation",
                                ErrorKind::UnexpectedError("invalid field in struct"),
                                expr.span,
                            )),
                        }
                    }
                    _ => Err(Error::new(
                        "constraint-generation",
                        ErrorKind::UnexpectedError("expected a field access"),
                        expr.span,
                    )),
                }
            }
            ComputedExprKind::Array(elements) => {
                match next_access {
                    AccessKind::Array(index) => {
                        match elements.get_mut(*index) {
                            Some(element_expr) => {
                                // stepping into the array element
                                Self::update_computed_expr(
                                    element_expr,
                                    remaining_accesses,
                                    new_expr,
                                )
                            }
                            None => Err(Error::new(
                                "constraint-generation",
                                ErrorKind::UnexpectedError("access index is out of bounds"),
                                expr.span,
                            )),
                        }
                    }
                    _ => Err(Error::new(
                        "constraint-generation",
                        ErrorKind::UnexpectedError("expected an array access"),
                        expr.span,
                    )),
                }
            }
            _ => Err(Error::new(
                "constraint-generation",
                ErrorKind::UnexpectedError("only field or array access is allowed"),
                expr.span,
            )),
        }
    }
}

//
// Circuit Writer (also used by witness generation)
//

impl<B: Backend> CircuitWriter<B> {
    fn compile_stmt(
        &mut self,
        fn_env: &mut FnEnv<B::Field, B::Var>,
        stmt: &Stmt,
    ) -> Result<Option<VarOrRef<B>>> {
        match &stmt.kind {
            StmtKind::Assign { mutable, lhs, rhs } => {
                // compute the rhs
                let rhs_var = self
                    .compute_expr(fn_env, rhs)?
                    .ok_or_else(|| self.error(ErrorKind::CannotComputeExpression, stmt.span))?;

                // obtain the actual values
                let rhs_var = rhs_var.value(self, fn_env);

                let typ = self.expr_type(rhs).cloned();
                let var_info = VarInfo::new(rhs_var, *mutable, typ);

                // store the new variable
                // TODO: do we really need to store that in the scope? That's not an actual var in the scope that's an internal var...
                self.add_local_var(fn_env, lhs.value.clone(), var_info);
            }

            StmtKind::ForLoop { var, range, body } => {
                for ii in range.range() {
                    fn_env.nest();

                    let cst_var = Var::new_constant(ii.into(), var.span);
                    let var_info = VarInfo::new(cst_var, false, Some(TyKind::Field));
                    self.add_local_var(fn_env, var.value.clone(), var_info);

                    self.compile_block(fn_env, body)?;

                    fn_env.pop();
                }
            }
            StmtKind::Expr(expr) => {
                // compute the expression
                let var = self.compute_expr(fn_env, expr)?;

                // make sure it does not return any value.
                assert!(var.is_none());
            }
            StmtKind::Return(expr) => {
                let var = self
                    .compute_expr(fn_env, expr)?
                    .ok_or_else(|| self.error(ErrorKind::CannotComputeExpression, stmt.span))?;

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
        fn_env: &mut FnEnv<B::Field, B::Var>,
        stmts: &[Stmt],
    ) -> Result<Option<Var<B::Field, B::Var>>> {
        fn_env.nest();
        for stmt in stmts {
            let res = self.compile_stmt(fn_env, stmt)?;
            if let Some(var) = res {
                // a block doesn't return a pointer, only values
                let var = var.value(self, fn_env);

                // we already checked for early returns in type checking
                return Ok(Some(var));
            }
        }
        fn_env.pop();
        Ok(None)
    }

    fn compile_native_function_call(
        &mut self,
        function: &FunctionDef,
        args: Vec<VarInfo<B::Field, B::Var>>,
    ) -> Result<Option<Var<B::Field, B::Var>>> {
        assert!(!function.is_main());

        // create new fn_env
        let fn_env = &mut FnEnv::new();

        // set arguments
        assert_eq!(function.sig.arguments.len(), args.len());

        for (name, var_info) in function.sig.arguments.iter().zip(args) {
            self.add_local_var(fn_env, name.name.value.clone(), var_info);
        }

        // compile it and potentially return a return value
        self.compile_block(fn_env, &function.body)
    }

    pub(crate) fn constrain_inputs_to_main(
        &mut self,
        input: &[ConstOrCell<B::Field, B::Var>],
        input_typ: &TyKind,
        span: Span,
    ) -> Result<()> {
        match input_typ {
            TyKind::Field => (),
            TyKind::Bool => {
                assert_eq!(input.len(), 1);
                boolean::check(self, &input[0], span);
            }
            TyKind::Array(tykind, _) => {
                let el_size = self.size_of(tykind);
                for el in input.chunks(el_size) {
                    self.constrain_inputs_to_main(el, tykind, span)?;
                }
            }
            TyKind::Custom {
                module,
                name: struct_name,
            } => {
                let qualified = FullyQualified::new(module, &struct_name);
                let struct_info = self
                    .struct_info(&qualified)
                    .ok_or(self.error(ErrorKind::UnexpectedError("struct not found"), span))?
                    .clone();

                let mut offset = 0;
                for (_field_name, field_typ) in &struct_info.fields {
                    let len = self.size_of(field_typ);
                    let range = offset..(offset + len);
                    self.constrain_inputs_to_main(&input[range], field_typ, span)?;
                    offset += len;
                }
            }
            TyKind::BigInt => unreachable!(),
        };
        Ok(())
    }

    /// Compile a function. Used to compile `main()` only for now
    pub(crate) fn compile_main_function(
        &mut self,
        fn_env: &mut FnEnv<B::Field, B::Var>,
        function: &FunctionDef,
    ) -> Result<Option<Vec<B::Var>>> {
        assert!(function.is_main());

        // compile the block
        let returned = self.compile_block(fn_env, &function.body)?;

        // we're expecting something returned?
        match (function.sig.return_type.as_ref(), returned) {
            (None, None) => Ok(None),
            (Some(expected), None) => Err(self.error(ErrorKind::MissingReturn, expected.span)),
            (None, Some(returned)) => Err(self.error(ErrorKind::UnexpectedReturn, returned.span)),
            (Some(_expected), Some(returned)) => {
                // make sure there are no constants in the returned value
                let mut returned_cells = vec![];
                for r in &returned.cvars {
                    match r {
                        ConstOrCell::Cell(c) => returned_cells.push(c.clone()),
                        ConstOrCell::Const(_) => {
                            return Err(self.error(ErrorKind::ConstantInOutput, returned.span))
                        }
                    }
                }

                self.public_output
                    .as_ref()
                    .expect("bug in the compiler: missing public output");

                Ok(Some(returned_cells))
            }
        }
    }

    fn compute_expr(
        &mut self,
        fn_env: &mut FnEnv<B::Field, B::Var>,
        expr: &Expr,
    ) -> Result<Option<VarOrRef<B>>> {
        match &expr.kind {
            // `module::fn_name(args)`
            ExprKind::FnCall {
                module,
                fn_name,
                args,
            } => {
                // sanity check
                if fn_name.value == "main" {
                    return Err(self.error(ErrorKind::RecursiveMain, expr.span));
                }

                // retrieve the function in the env
                let qualified = FullyQualified::new(module, &fn_name.value);
                let fn_info = self
                    .fn_info(&qualified)
                    .ok_or_else(|| {
                        self.error(
                            ErrorKind::UndefinedFunction(fn_name.value.clone()),
                            fn_name.span,
                        )
                    })?
                    .clone();

                // compute the arguments
                // module::fn_name(args)
                //                 ^^^^
                let mut vars = Vec::with_capacity(args.len());
                for arg in args {
                    // get the variable behind the expression
                    let var = self
                        .compute_expr(fn_env, arg)?
                        .ok_or_else(|| self.error(ErrorKind::CannotComputeExpression, arg.span))?;

                    // we pass variables by values always
                    let var = var.value(self, fn_env);

                    let typ = self.expr_type(arg).cloned();
                    let mutable = false; // TODO: mut keyword in arguments?
                    let var_info = VarInfo::new(var, mutable, typ);

                    vars.push(var_info);
                }

                let res = match &fn_info.kind {
                    // assert() <-- for example
                    FnKind::BuiltIn(_sig, handle) => {
                        let res = handle(self, &vars, expr.span);
                        res.map(|r| r.map(VarOrRef::Var))
                    }

                    // fn_name(args)
                    // ^^^^^^^
                    FnKind::Native(func) => {
                        // module::fn_name(args)
                        // ^^^^^^
                        self.compile_native_function_call(&func, vars)
                            .map(|r| r.map(VarOrRef::Var))
                    }
                };

                //
                res
            }

            ExprKind::FieldAccess { lhs, rhs } => {
                // get var behind lhs
                let lhs_var = self
                    .compute_expr(fn_env, lhs)?
                    .ok_or_else(|| self.error(ErrorKind::CannotComputeExpression, lhs.span))?;

                // get struct info behind lhs
                let lhs_struct = self
                    .expr_type(lhs)
                    .ok_or_else(|| self.error(ErrorKind::CannotComputeExpression, lhs.span))?;

                let (module, self_struct) = match lhs_struct {
                    TyKind::Custom { module, name } => (module, name),
                    _ => {
                        panic!("could not figure out struct implementing that method call")
                    }
                };

                let qualified = FullyQualified::new(module, self_struct);
                let struct_info = self
                    .struct_info(&qualified)
                    .expect("struct info not found for custom struct");

                // find range of field
                let mut start = 0;
                let mut len = 0;
                for (field, field_typ) in &struct_info.fields {
                    if field == &rhs.value {
                        len = self.size_of(field_typ);
                        break;
                    }

                    start += self.size_of(field_typ);
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
                let lhs_typ = self.expr_type(lhs).expect("method call on what?").clone();

                let (module, struct_name) = match &lhs_typ {
                    TyKind::Custom { module, name } => (module, name),
                    _ => {
                        return Err(self.error(
                            ErrorKind::UnexpectedError("method call only work on custom types"),
                            lhs.span,
                        ))
                    }
                };

                // get var of `self`
                // (might be `None` if it's a static method call)
                let self_var = self.compute_expr(fn_env, lhs)?;

                // find method info
                let qualified = FullyQualified::new(module, struct_name);
                let struct_info = self
                    .struct_info(&qualified)
                    .ok_or(self.error(
                        ErrorKind::UnexpectedError("struct not found"),
                        method_name.span,
                    ))?
                    .clone();
                let func = struct_info
                    .methods
                    .get(&method_name.value)
                    .expect("could not find method");

                // if method has a `self` argument, manually add it to the list of argument
                let mut vars = vec![];
                if let Some(first_arg) = func.sig.arguments.first() {
                    if first_arg.name.value == "self" {
                        let self_var = self_var.ok_or_else(|| {
                            self.error(ErrorKind::NotAStaticMethod, method_name.span)
                        })?;

                        // TODO: for now we pass `self` by value as well
                        let mutable = false;
                        let self_var = self_var.value(self, fn_env);

                        let self_var_info = VarInfo::new(self_var, mutable, Some(lhs_typ.clone()));
                        vars.insert(0, self_var_info);
                    }
                } else {
                    assert!(self_var.is_none());
                }

                // compute the arguments
                for arg in args {
                    let var = self
                        .compute_expr(fn_env, arg)?
                        .ok_or_else(|| self.error(ErrorKind::CannotComputeExpression, arg.span))?;

                    // TODO: for now we pass `self` by value as well
                    let mutable = false;
                    let var = var.value(self, fn_env);

                    let typ = self.expr_type(arg).cloned();
                    let var_info = VarInfo::new(var, mutable, typ);

                    vars.push(var_info);
                }

                // execute method
                self.compile_native_function_call(func, vars)
                    .map(|r| r.map(VarOrRef::Var))
            }

            ExprKind::IfElse { cond, then_, else_ } => {
                let cond = self
                    .compute_expr(fn_env, cond)?
                    .unwrap()
                    .value(self, fn_env);
                let then_ = self
                    .compute_expr(fn_env, then_)?
                    .unwrap()
                    .value(self, fn_env);
                let else_ = self
                    .compute_expr(fn_env, else_)?
                    .unwrap()
                    .value(self, fn_env);

                let res = field::if_else(self, &cond, &then_, &else_, expr.span);

                Ok(Some(VarOrRef::Var(res)))
            }

            ExprKind::Assignment { lhs, rhs } => {
                // figure out the local var  of lhs
                let lhs = self.compute_expr(fn_env, lhs)?.unwrap();

                // figure out the var of what's on the right
                let rhs = self.compute_expr(fn_env, rhs)?.unwrap();
                let rhs_var = match rhs {
                    VarOrRef::Var(var) => var,
                    VarOrRef::Ref {
                        var_name,
                        start,
                        len,
                    } => {
                        let var_info = self.get_local_var(fn_env, &var_name);
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
                let lhs = self.compute_expr(fn_env, lhs)?.unwrap();
                let rhs = self.compute_expr(fn_env, rhs)?.unwrap();

                let lhs = lhs.value(self, fn_env);
                let rhs = rhs.value(self, fn_env);

                let res = match op {
                    Op2::Addition => field::add(self, &lhs[0], &rhs[0], expr.span),
                    Op2::Subtraction => field::sub(self, &lhs[0], &rhs[0], expr.span),
                    Op2::Multiplication => field::mul(self, &lhs[0], &rhs[0], expr.span),
                    Op2::Equality => field::equal(self, &lhs, &rhs, expr.span),
                    Op2::Inequality => field::not_equal(self, &lhs, &rhs, expr.span),
                    Op2::BoolAnd => boolean::and(self, &lhs[0], &rhs[0], expr.span),
                    Op2::BoolOr => boolean::or(self, &lhs[0], &rhs[0], expr.span),
                    Op2::Division => todo!(),
                };

                Ok(Some(VarOrRef::Var(res)))
            }

            ExprKind::Negated(b) => {
                let var = self.compute_expr(fn_env, b)?.unwrap();

                let var = var.value(self, fn_env);

                todo!()
            }

            ExprKind::Not(b) => {
                let var = self.compute_expr(fn_env, b)?.unwrap();

                let var = var.value(self, fn_env);

                let res = boolean::not(self, &var[0], expr.span.merge_with(b.span));
                Ok(Some(VarOrRef::Var(res)))
            }

            ExprKind::BigUInt(b) => {
                let ff = B::Field::try_from(b.to_owned()).map_err(|_| {
                    self.error(ErrorKind::CannotConvertToField(b.to_string()), expr.span)
                })?;

                let res = VarOrRef::Var(Var::new_constant(ff, expr.span));
                Ok(Some(res))
            }

            ExprKind::Bool(b) => {
                let value = if *b {
                    B::Field::one()
                } else {
                    B::Field::zero()
                };
                let res = VarOrRef::Var(Var::new_constant(value, expr.span));
                Ok(Some(res))
            }

            ExprKind::Variable { module, name } => {
                // if it's a type we return nothing
                // (most likely what follows is a static method call)
                if is_type(&name.value) {
                    return Ok(None);
                }

                // search for constants first
                let qualified = FullyQualified::new(module, &name.value);
                let var_info = if let Some(cst_info) = self.const_info(&qualified) {
                    let var = Var::new_constant_typ(cst_info, name.span);
                    VarInfo::new(var, false, Some(cst_info.typ.kind.clone()))
                } else {
                    // if no constant found, look in the function's scope
                    // remember: we can do this because the type checker already checked that we didn't shadow constant vars
                    self.get_local_var(fn_env, &name.value)
                };

                let res = VarOrRef::from_var_info(name.value.clone(), var_info);
                Ok(Some(res))
            }

            ExprKind::ArrayAccess { array, idx } => {
                // retrieve var of array
                let var = self
                    .compute_expr(fn_env, array)?
                    .expect("array access on non-array");

                // compute the index
                let idx_var = self
                    .compute_expr(fn_env, idx)?
                    .ok_or_else(|| self.error(ErrorKind::CannotComputeExpression, expr.span))?;
                let idx = idx_var
                    .constant()
                    .ok_or_else(|| self.error(ErrorKind::ExpectedConstant, expr.span))?;
                let idx: BigUint = idx.into();
                let idx: usize = idx.try_into().unwrap();

                // retrieve the type of the elements in the array
                let array_typ = self.expr_type(array).expect("cannot find type of array");

                let elem_type = match array_typ {
                    TyKind::Array(ty, array_len) => {
                        if idx >= (*array_len as usize) {
                            return Err(self.error(
                                ErrorKind::ArrayIndexOutOfBounds(idx, *array_len as usize - 1),
                                expr.span,
                            ));
                        }
                        ty
                    }
                    _ => panic!("expected array"),
                };

                // compute the size of each element in the array
                let len = self.size_of(elem_type);

                // compute the real index
                let start = idx * len;

                // out-of-bound checks
                if start >= var.len() || start + len > var.len() {
                    return Err(self.error(
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
                    let var = self.compute_expr(fn_env, item)?.unwrap();
                    let to_extend = var.value(self, fn_env).cvars.clone();
                    cvars.extend(to_extend);
                }

                let var = VarOrRef::Var(Var::new(cvars, expr.span));

                Ok(Some(var))
            }

            ExprKind::CustomTypeDeclaration { custom: _, fields } => {
                // create the struct by just concatenating all of its cvars
                let mut cvars = vec![];
                for (_field, rhs) in fields {
                    let var = self.compute_expr(fn_env, rhs)?.unwrap();
                    let to_extend = var.value(self, fn_env).cvars.clone();
                    cvars.extend(to_extend);
                }
                let var = VarOrRef::Var(Var::new(cvars, expr.span));

                //
                Ok(Some(var))
            }
        }
    }

    pub fn add_public_inputs(
        &mut self,
        name: String,
        num: usize,
        span: Span,
    ) -> Var<B::Field, B::Var> {
        let mut cvars = Vec::with_capacity(num);

        for idx in 0..num {
            let cvar = self
                .backend
                .add_public_input(Value::External(name.clone(), idx), span);
            cvars.push(ConstOrCell::Cell(cvar));
        }

        Var::new(cvars, span)
    }

    pub fn add_public_outputs(&mut self, num: usize, span: Span) {
        assert!(self.public_output.is_none());

        let mut cvars = Vec::with_capacity(num);
        for _ in 0..num {
            let cvar = self
                .backend
                .add_public_output(Value::PublicOutput(None), span);
            cvars.push(ConstOrCell::Cell(cvar));
        }

        // store it
        let res = Var::new(cvars, span);
        self.public_output = Some(res);
    }

    pub fn add_private_inputs(
        &mut self,
        name: String,
        num: usize,
        span: Span,
    ) -> Var<B::Field, B::Var> {
        let mut cvars = Vec::with_capacity(num);

        for idx in 0..num {
            // create the var
            let cvar = self
                .backend
                .add_private_input(Value::External(name.clone(), idx), span);
            cvars.push(ConstOrCell::Cell(cvar));
        }

        Var::new(cvars, span)
    }
}

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub(crate) struct PendingGate {
    pub label: &'static str,
    #[serde(skip)]
    pub coeffs: Vec<VestaField>,
    pub vars: Vec<Option<crate::backends::kimchi::KimchiCellVar>>,
    pub span: Span,
}
