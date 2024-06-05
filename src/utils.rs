use std::fmt::Write;

#[must_use]
pub fn noname_version() -> String {
    format!("@ noname.{}\n\n", env!("CARGO_PKG_VERSION"))
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
        writeln!(res, "│ FILE: {file}").unwrap();
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

fn find_exact_line(source: &str, span: crate::constants::Span) -> (usize, usize, &str) {
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
    writeln!(res, "│{s}│").unwrap();
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
