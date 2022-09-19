# Paths

Paths are structures identifying snippets of code that look like this:

```rust
some_module::some_ident.stuff.z
```

The identifier `some_module`, appearing before the `::`, is an optional module, pointing to code that exists in another library. It is always lowercase.

The identifier `some_ident` is mandatory. It can represent a type (if it starts with a capital letter), a function name, a variable name, a constant name, etc.

More identifiers can be concatenated together to form a longer path (using `.`).
A path is represent like this internally:

```rust
/// A path represents a path to a type, a function, a method, or a constant.
/// It follows the syntax: `module::X` where `X` can be a type, a function, a method, or a constant.
/// In the case it is a method `a` on some type `A` then it would read:
/// `module::A.a`.
#[derive(Debug, Clone)]
pub struct Path {
    /// A module, if this is an foreign import.
    pub module: Option<Ident>,

    /// The name of the type, function, method, or constant.
    /// It's a vector because it can also be a struct access.
    pub name: Vec<Ident>,

    /// Its span.
    pub span: Span,
}
```

## Expressions using Path

A path does not represent an expression by itself. The following expressions make use of `path`:

```rust
pub enum ExprKind {
    /// `module::some_fn()`
    FnCall {
        name: Path,
        args: Vec<Expr>,
    },

    /// `module::SomeType.some_method()`
    MethodCall {
        self_name: Path,
        method_name: Ident,
        args: Vec<Expr>,
    },

    /// `module::some_var` or
    /// `module::SomeType.some_field.some_other_field`
    Identifier(Path),

    /// `module::SomeType.some_field[some_expr]` or
    /// `module::some_const[some_expr]`
    ArrayAccess {
        name: Path,
        idx: Box<Expr>,
    },

    /// `module::SomeType { field1: expr1; field2: expr2 }`
    CustomTypeDeclaration(Path, Vec<(Ident, Expr)>),
}
```
