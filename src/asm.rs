use std::ops::Neg;

use num_bigint::BigUint;

use crate::{
    ast::Gate,
    constants::{Field, Span},
};

pub fn generate_asm(source: &str, gates: &[Gate]) -> String {
    let mut res = "".to_string();

    fn find_exact_line(source: &str, span: Span) -> (usize, usize, &str) {
        let ss = source.as_bytes();
        let mut start = span.0;
        let mut end = span.0 + span.1;
        while start > 0 && (ss[start - 1] as char) != '\n' {
            start -= 1;
        }
        while end < source.len() && (ss[end] as char) != '\n' {
            end += 1;
        }

        let line = &source[start..end];

        let line_number = source[..start].matches('\n').count() + 1;

        (line_number, start, line)
    }

    for Gate { typ, coeffs, span } in gates {
        // source
        let (line_number, start, line) = find_exact_line(source, *span);
        res.push_str("\n\n----\n");
        res.push_str(&format!("{line_number}: {line}\n"));
        for _ in start..span.0 {
            res.push_str(" ");
        }
        res.push_str(&"^".repeat(span.1));
        res.push_str("----\n");

        // gate coeffs
        let (vars, coeffs) = parse_coeffs(coeffs);
        res.push_str(&vars);

        // gate
        res.push_str(&format!("{typ:?}"));
        res.push_str("<");
        res.push_str(&coeffs.join(","));
        res.push_str(">\n");
    }

    res
}

fn parse_coeff(x: Field) -> String {
    // TODO: if it's bigger than n/2 then it should be a negative number
    let bigint: BigUint = x.into();
    let inv: BigUint = x.neg().into(); // gettho way of splitting the field into positive and negative elements
    if inv < bigint {
        format!("-{}", inv)
    } else {
        bigint.to_string()
    }
}

fn parse_coeffs(coeffs: &[Field]) -> (String, Vec<String>) {
    let mut vars = String::new();
    let coeffs = coeffs
        .iter()
        .map(|x| {
            let s = parse_coeff(*x);
            if s.len() < 5 {
                s
            } else {
                let var = format!("c{}", vars.len());
                vars.push_str(&format!("{var}={s}\n"));
                var
            }
        })
        .collect();
    (vars, coeffs)
}
