use crate::{
    ast::{CircuitVar, Compiler, FuncType},
    constants::Span,
};

const POSEIDON_FN: &str = "poseidon(input: [Field; 3]) -> [Field; 3]";

pub const CRYPTO_FNS: [(&str, FuncType); 1] = [(POSEIDON_FN, poseidon)];

fn poseidon(compiler: &mut Compiler, vars: &[CircuitVar], span: Span) -> Option<CircuitVar> {
    assert_eq!(vars.len(), 1);
    let input = &vars[0].vars;

    assert_eq!(input.len(), 3);
    let x0 = input[0];
    let x1 = input[1];
    let x2 = input[2];

    unimplemented!();
}
