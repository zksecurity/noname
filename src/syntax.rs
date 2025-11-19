//! A number of helper function to check the syntax of some types.

/// Returns true if the given string is a number in decimal.
pub fn is_numeric(s: &str) -> bool {
    s.chars().all(|c| c.is_ascii_digit())
}

/// Returns true if the given string is a hexadecimal string (0x...)
pub fn is_hexadecimal(s: &str) -> bool {
    let mut s = s.chars();
    let s0 = s.next();
    let s1 = s.next();
    if matches!((s0, s1), (Some('0'), Some('x') | Some('X'))) {
        s.all(|c| c.is_ascii_hexdigit())
    } else {
        false
    }
}

/// Returns true if the given string is an identifier or type
pub fn is_identifier_or_type(s: &str) -> bool {
    let mut chars = s.chars();
    let first_letter = chars.next().unwrap();
    // first char is a letter
    first_letter.is_alphabetic()
    // rest are lowercase alphanumeric or underscore
        && chars
            .all(|c| (c.is_ascii_alphabetic() && c.is_lowercase()) || c.is_numeric() || c == '_')
}

/// Returns true if the given string is an identifier
/// (starts with a lowercase letter)
pub fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    let first_letter = chars.next().unwrap();
    // first char is a letter
    first_letter.is_alphabetic() && first_letter.is_lowercase()
    // rest are lowercase alphanumeric or underscore
        && chars
            .all(|c| (c.is_ascii_alphabetic() && c.is_lowercase()) || c.is_numeric() || c == '_')
}

/// Returns true if the given string is generic parameter
pub fn is_generic_parameter(s: &str) -> bool {
    // should be at least 2 uppercase letters
    if s.len() < 2 {
        return false;
    }

    let mut chars = s.chars();
    // all should be uppercase alphabetic
    chars.all(|c| (c.is_ascii_alphabetic() && c.is_uppercase()))
}

/// Returns true if the given string is a type
/// Check camel case
pub fn is_type(s: &str) -> bool {
    let mut chars = s.chars();
    // must have at least two char
    if s.len() < 2 {
        return false;
    }

    let first_char = chars.next().unwrap();
    // first char is an uppercase letter
    // rest are lowercase alphanumeric
    first_char.is_alphabetic()
        && first_char.is_uppercase()
        && chars.all(|c| ((c.is_alphabetic() && c.is_lowercase()) || c.is_numeric()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syntax() {
        assert!(is_identifier_or_type("cond2"));
        assert!(is_identifier_or_type("Cond2"));
        assert!(!is_identifier_or_type("_cond2"));
        assert!(!is_identifier_or_type("2_cond2"));
        assert!(is_identifier_or_type("c_ond2"));
        assert!(is_identifier("cond2"));
        assert!(is_type("Cond2"));
        assert!(!is_type("C"));
        assert!(!is_generic_parameter("N"));
        assert!(!is_generic_parameter("n"));
        assert!(!is_generic_parameter("nn"));
        assert!(is_generic_parameter("NN"));
        assert!(!is_generic_parameter("N1"));
        assert!(!is_generic_parameter("N_"));
    }
}
