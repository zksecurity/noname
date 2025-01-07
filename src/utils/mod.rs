use crate::{
    backends::Backend,
    circuit_writer::VarInfo,
    constants::Span,
    error::{Error, ErrorKind, Result},
    helpers::PrettyField,
    mast::Mast,
    parser::types::{ModulePath, TyKind},
    type_checker::FullyQualified,
    var::ConstOrCell,
    witness::WitnessEnv,
};
use std::fmt::Write;
use std::slice::Iter;

use num_traits::One;
use regex::{Captures, Regex};

pub fn noname_version() -> String {
    format!("@ noname.{}\n", env!("CARGO_PKG_VERSION"))
}

pub fn display_source(
    res: &mut String,
    sources: &crate::compiler::Sources,
    debug_infos: &[crate::circuit_writer::DebugInfo],
) {
    for crate::circuit_writer::DebugInfo { span, note: _ } in debug_infos {
        // find filename and source
        let (file, source) = sources.get(&span.filename_id).expect("source not found");

        // top corner
        res.push('╭');
        res.push_str(&"─".repeat(80));
        res.push('\n');

        // display filename
        writeln!(res, "│ FILE: {}", file).unwrap();
        writeln!(res, "│{s}", s = "─".repeat(80)).unwrap();

        // source
        res.push_str("│ ");
        let (line_number, start, line) = find_exact_line(source, *span);
        let header = format!("{line_number}: ");
        writeln!(res, "{header}{line}").unwrap();

        // caption
        res.push('│');
        res.push_str(&" ".repeat(header.len() + 1 + span.start - start));
        res.push_str(&"^".repeat(span.len));
        res.push('\n');
    }

    // bottom corner
    res.push('╰');
    res.push_str(&"─".repeat(80));
    res.push('\n');
}

pub(crate) fn find_exact_line(source: &str, span: crate::constants::Span) -> (usize, usize, &str) {
    let ss = source.as_bytes();
    let mut start = span.start;
    let mut end = span.end();
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

pub fn title(res: &mut String, s: &str) {
    writeln!(res, "╭{s}╮", s = "─".repeat(s.len())).unwrap();
    writeln!(res, "│{s}│", s = s).unwrap();
    writeln!(res, "╰{s}╯", s = "─".repeat(s.len())).unwrap();
    writeln!(res).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_lines() {
        const SRC: &str = "abcd
efgh
ijkl
mnop
qrst
uvwx
yz
";
        assert_eq!(
            find_exact_line(SRC, crate::constants::Span::new(0, 5, 6)),
            (2, 5, "efgh\nijkl")
        );
    }
}

// for failable replacer this is the recommended approach by the author of regex lib https://github.com/rust-lang/regex/issues/648#issuecomment-590072186
// I have made Fn -> FnMut because our replace mutates the iterator by moving it forward
fn replace_all(
    re: &Regex,
    haystack: &str,
    mut replacement: impl FnMut(&Captures) -> Result<String>,
) -> Result<String> {
    let mut new = String::with_capacity(haystack.len());
    let mut last_match = 0;
    for caps in re.captures_iter(haystack) {
        let m = caps.get(0).unwrap();
        new.push_str(&haystack[last_match..m.start()]);
        new.push_str(&replacement(&caps)?);
        last_match = m.end();
    }
    new.push_str(&haystack[last_match..]);
    Ok(new)
}

pub fn log_string_type<B: Backend>(
    backend: &B,
    logs_iter: &mut Iter<'_, (String, Span, VarInfo<B::Field, B::Var>)>,
    str: &str,
    witness: &mut WitnessEnv<B::Field>,
    typed: &Mast<B>,
    span: &Span,
) -> Result<String> {
    let re = Regex::new(r"\{\s*\}").unwrap();
    let replacer = |_: &Captures| {
        let (_, span, var) = match logs_iter.next() {
            Some((str, span, var)) => (str, span, var),
            None => return Err(Error::new("log", ErrorKind::InsufficientVariables, *span)),
        };
        let replacement = match &var.typ {
            Some(TyKind::Field { .. }) => match &var.var[0] {
                ConstOrCell::Const(cst) => Ok(cst.pretty()),
                ConstOrCell::Cell(cell) => {
                    let val = backend.compute_var(witness, cell).unwrap();
                    Ok(val.pretty())
                }
            },
            // Bool
            Some(TyKind::Bool) => match &var.var[0] {
                ConstOrCell::Const(cst) => {
                    let val = *cst == B::Field::one();
                    Ok(val.to_string())
                }
                ConstOrCell::Cell(cell) => {
                    let val = backend.compute_var(witness, cell)? == B::Field::one();
                    Ok(val.to_string())
                }
            },

            // Array
            Some(TyKind::Array(b, s)) => {
                let (output, remaining) =
                    log_array_type(backend, &var.var.cvars, b, *s, witness, typed, span).unwrap();
                assert!(remaining.is_empty());
                Ok(output)
            }

            // Custom types
            Some(TyKind::Custom {
                module,
                name: struct_name,
            }) => {
                let mut string_vec = Vec::new();
                let (output, remaining) = log_custom_type(
                    backend,
                    module,
                    struct_name,
                    typed,
                    &var.var.cvars,
                    witness,
                    span,
                    &mut string_vec,
                )
                .unwrap();
                assert!(remaining.is_empty());
                Ok(output)
            }

            // GenericSizedArray
            Some(TyKind::GenericSizedArray(_, _)) => {
                unreachable!("GenericSizedArray should be monomorphized")
            }
            Some(TyKind::String(_)) => todo!("String cannot be in circuit yet"),
            // I cannot return an error here
            None => {
                return Err(Error::new(
                    "log",
                    ErrorKind::UnexpectedError("No type info for logging"),
                    *span,
                ))
            }
        };
        replacement
    };
    replace_all(&re, str, replacer)
}

pub fn log_array_type<B: Backend>(
    backend: &B,
    var_info_var: &[ConstOrCell<B::Field, B::Var>],
    base_type: &TyKind,
    size: u32,
    witness: &mut WitnessEnv<B::Field>,
    typed: &Mast<B>,
    span: &Span,
) -> Result<(String, Vec<ConstOrCell<B::Field, B::Var>>)> {
    match base_type {
        TyKind::Field { .. } => {
            let values: Vec<String> = var_info_var
                .iter()
                .take(size as usize)
                .map(|cvar| match cvar {
                    ConstOrCell::Const(cst) => cst.pretty(),
                    ConstOrCell::Cell(cell) => backend.compute_var(witness, cell).unwrap().pretty(),
                })
                .collect();

            let remaining = var_info_var[size as usize..].to_vec();
            Ok((format!("[{}]", values.join(", ")), remaining))
        }

        TyKind::Bool => {
            let values: Vec<String> = var_info_var
                .iter()
                .take(size as usize)
                .map(|cvar| match cvar {
                    ConstOrCell::Const(cst) => {
                        let val = *cst == B::Field::one();
                        val.to_string()
                    }
                    ConstOrCell::Cell(cell) => {
                        let val = backend.compute_var(witness, cell).unwrap() == B::Field::one();
                        val.to_string()
                    }
                })
                .collect();

            let remaining = var_info_var[size as usize..].to_vec();
            Ok((format!("[{}]", values.join(", ")), remaining))
        }

        TyKind::Array(inner_type, inner_size) => {
            let mut nested_result = Vec::new();
            let mut remaining = var_info_var.to_vec();
            for _ in 0..size {
                let (chunk_result, new_remaining) = log_array_type(
                    backend,
                    &remaining,
                    inner_type,
                    *inner_size,
                    witness,
                    typed,
                    span,
                )?;
                nested_result.push(chunk_result);
                remaining = new_remaining;
            }
            Ok((format!("[{}]", nested_result.join(", ")), remaining))
        }

        TyKind::Custom {
            module,
            name: struct_name,
        } => {
            let mut nested_result = Vec::new();
            let mut remaining = var_info_var.to_vec();
            for _ in 0..size {
                let mut string_vec = Vec::new();
                let (output, new_remaining) = log_custom_type(
                    backend,
                    module,
                    struct_name,
                    typed,
                    &remaining,
                    witness,
                    span,
                    &mut string_vec,
                )?;
                nested_result.push(format!("{}{}", struct_name, output));
                remaining = new_remaining;
            }
            Ok((format!("[{}]", nested_result.join(", ")), remaining))
        }

        TyKind::GenericSizedArray(_, _) => {
            unreachable!("GenericSizedArray should be monomorphized")
        }

        TyKind::String(_) => todo!("String type cannot be used in circuits!"),
    }
}
pub fn log_custom_type<B: Backend>(
    backend: &B,
    module: &ModulePath,
    struct_name: &String,
    typed: &Mast<B>,
    var_info_var: &[ConstOrCell<B::Field, B::Var>],
    witness: &mut WitnessEnv<B::Field>,
    span: &Span,
    string_vec: &mut Vec<String>,
) -> Result<(String, Vec<ConstOrCell<B::Field, B::Var>>)> {
    let qualified = FullyQualified::new(module, struct_name);
    let struct_info = typed
        .struct_info(&qualified)
        .ok_or(
            typed
                .0
                .error(ErrorKind::UnexpectedError("struct not found"), *span),
        )
        .unwrap();

    let mut remaining = var_info_var.to_vec();

    for (field_name, field_typ) in &struct_info.fields {
        let len = typed.size_of(field_typ);
        match field_typ {
            TyKind::Field { .. } => match &remaining[0] {
                ConstOrCell::Const(cst) => {
                    string_vec.push(format!("{field_name}: {}", cst.pretty()));
                    remaining = remaining[len..].to_vec();
                }
                ConstOrCell::Cell(cell) => {
                    let val = backend.compute_var(witness, cell)?;
                    string_vec.push(format!("{field_name}: {}", val.pretty()));
                    remaining = remaining[len..].to_vec();
                }
            },

            TyKind::Bool => match &remaining[0] {
                ConstOrCell::Const(cst) => {
                    let val = *cst == B::Field::one();
                    string_vec.push(format!("{field_name}: {}", val));
                    remaining = remaining[len..].to_vec();
                }
                ConstOrCell::Cell(cell) => {
                    let val = backend.compute_var(witness, cell)? == B::Field::one();
                    string_vec.push(format!("{field_name}: {}", val));
                    remaining = remaining[len..].to_vec();
                }
            },

            TyKind::Array(b, s) => {
                let (output, new_remaining) =
                    log_array_type(backend, &remaining, b, *s, witness, typed, span)?;
                string_vec.push(format!("{field_name}: {}", output));
                remaining = new_remaining;
            }

            TyKind::Custom {
                module,
                name: struct_name,
            } => {
                let mut custom_string_vec = Vec::new();
                let (output, new_remaining) = log_custom_type(
                    backend,
                    module,
                    struct_name,
                    typed,
                    &remaining,
                    witness,
                    span,
                    &mut custom_string_vec,
                )?;
                string_vec.push(format!("{}: {}{}", field_name, struct_name, output));
                remaining = new_remaining;
            }

            TyKind::GenericSizedArray(_, _) => {
                unreachable!("GenericSizedArray should be monomorphized")
            }
            TyKind::String(s) => {
                todo!("String cannot be a type for customs it is only for logging")
            }
        }
    }

    Ok((format!("{{ {} }}", string_vec.join(", ")), remaining))
}
