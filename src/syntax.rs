//! A number of helper function to check the syntax of some types.

/// Returns true if the given string is a number in decimal.
#[must_use]
pub fn is_numeric(s: &str) -> bool {
    s.chars().all(|c| c.is_ascii_digit())
}

/// Returns true if the given string is an hexadecimal string (0x...)
#[must_use]
pub fn is_hexadecimal(s: &str) -> bool {
    let mut s = s.chars();
    let s0 = s.next();
    let s1 = s.next();
    if matches!((s0, s1), (Some('0'), Some('x' | 'X'))) {
        s.all(|c| c.is_ascii_hexdigit())
    } else {
        false
    }
}

/// Returns true if the given string is an identifier or type
#[must_use]
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
#[must_use]
pub fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    let first_letter = chars.next().unwrap();
    // first char is a letter
    first_letter.is_alphabetic() && first_letter.is_lowercase()
    // rest are lowercase alphanumeric or underscore
        && chars
            .all(|c| (c.is_ascii_alphabetic() && c.is_lowercase()) || c.is_numeric() || c == '_')
}

/// Returns true if the given string is a type
/// (first letter is an uppercase)
#[must_use]
pub fn is_type(s: &str) -> bool {
    let mut chars = s.chars();
    let first_char = chars.next().unwrap();
    // first char is an uppercase letter
    // rest are lowercase alphanumeric
    first_char.is_alphabetic() && first_char.is_uppercase() && chars.all(char::is_alphanumeric)
    // TODO: check camel case?
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
    }
}
