# Grammar

The syntax of the noname language is described through its grammar.

We use a notation similar to the Backus-Naur Form (BNF)
to describe the grammar:

<pre>
land := city "|"
 ^        ^   ^
 |        |  terminal: a token
 |        |
 |      another non-terminal
 |
 non-terminal: definition of a piece of code
city := [ sign ] "," { house }
        ^            ^
        optional     |
                    0r or more houses
sign := /a-zA-Z_/
        ^
        regex-style definition
</pre>

There are some comments in the parser code (`parser.rs`) that attempt to define this grammar.

Essentially, it is made to look like Rust, but with some differences of philosophies:

* expressions cannot be statements, unless they return no value (act using side effects).
