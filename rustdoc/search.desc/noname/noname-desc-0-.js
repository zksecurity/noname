searchState.loadedDescShard("noname", 0, "This is a high-level language to write circuits that you …\nThis module is a wrapper API around noname. It is …\nUsed to parse public and private inputs to a program.\nThis adds a few utility functions for serializing and …\nA number of helper function to check the syntax of some …\nThis trait serves as an alias for a bundle of traits\nThis trait allows different backends to have different …\nThe circuit field / scalar field that the circuit is …\nThe generated witness type for the backend. Each backend …\nThe CellVar type for the backend. Different backend is …\nadd two vars\nadd a var with a constant\nThis should be called only when you want to constrain a …\nProcess a private input\nProcess a public input\nProcess a public output\nadd a constraint to assert a var equals a constant\nadd a constraint to assert a var equals another var\nBackends should implement this function to load and …\nFinalize the circuit by doing some sanitizing checks.\nReturns the argument unchanged.\nGenerate the asm for a backend.\nGenerate the witness for a backend.\nInit circuit\nCalls <code>U::from(self)</code>.\nmultiply a var with another var\nmultiply a var with a constant\nnegate a var\nCreate a new cell variable and record it. It increments …\nposeidon crypto builtin function for different backends\nsub two vars\nNumber of columns in the execution trace.\nWe use the scalar field of Vesta as our circuit field.\ncontains all the witness values\nASM-like language:\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\ncontains the public inputs, which are also part of the …\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nThis module contains the prover.\ncontains the public outputs, which are also part of the …\nkimchi uses a transposed witness\nVery dumb way to write an ordered hash set.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nreturns a proof and a public output\nan R1CS constraint Each constraint comprises of 3 linear …\nAn intermediate struct for SnarkjsExporter to reorder the …\nLinear combination of variables and constants. For …\nR1CS backend with bls12_381 field.\nAdds the private input cell vars.\nAdds the public input cell vars.\nAdds the public output cell vars.\nFinal checks for generating the circuit. todo: we might …\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nGenerate the witnesses This process should check if the …\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCreate a new CellVar and record in witness_vector vector. …\nReturns the number of constraints.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nAn error type associated with <code>R1CSWriter</code>.\nA struct to export r1cs circuit and witness to the snarkjs …\nAn error type associated with <code>WitnessWriter</code>.\nReturns the argument unchanged.\nReturns the argument unchanged.\nGenerate the r1cs file in snarkjs format. It uses the …\nGenerate the wtns file in snarkjs format.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nDebug information related to a single row in a circuit.\nThe constraint backend for the circuit. For now, this …\nReturns the argument unchanged.\nReturns the argument unchanged.\nA wrapper for the backend generate_witness\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nRetrieves the FnInfo for the <code>main()</code> function. This …\nA note on why this was added\nThe place in the original source code that created that …\nIs used to store functions’ scoped variables. This …\nInformation about a variable.\nStores type information about a local variable. Note that …\nReturns the argument unchanged.\nReturns the argument unchanged.\nRetrieves type information on a variable, given a name. If …\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nEnters a scoped block.\nCreates a new FnEnv\nExits a scoped block.\nSame as [Self::reassign_var], but only reassigns a …\nWe keep track of the type of variables, eventhough we’re …\nThe variable.\nNot yet wired (just indicates the position of the cell …\nThe wiring (associated to different spans)\nCoefficients\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nType of gate\nThe directory under the user home directory containing all …\nThe directory under NONAME_DIRECTORY containing all …\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nThis retrieves a dependency listed in the manifest file. …\nA dependency is a Github <code>user/repo</code> pair.\ndownload package from github\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nThis retrieves a dependency listed in the manifest file. …\nReturns the dependencies of a package (given it’s …\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nA valid package must have a valid <code>Noname.toml</code> as well as a …\nContains the association between a counter and the …\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nMaps a filename id to its filename and source code.\nThis should not be used directly. Check [get_tast] instead.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCustom types\nAdds two field elements\nThis takes variables that can be anything, and returns a …\nMultiplies two field elements\nNegates a field element\nReturns 1 if lhs != rhs, 0 otherwise\nSubtracts two variables, we only support variables that …\nContains the error value\nAn error in noname.\nThe type of error.\nContains the success value\nthis error is for testing. You can use it when you want to …\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nThe type of error.\nA hint as to where the error happened (e.g. type-checker, …\nCreates a new [Error] from an ErrorKind.\nIndicate where the error occurred in the source code.\nA trait to display [Field] in pretty ways.\nPrint a field in a negative form if it’s past the half …\nA module that contains only built-in functions.\nA built-in is just a handle to a function written in Rust.\nAn actual handle to the internal function to call to …\nThe different types of a noname function.\nA module that contains both built-in functions and native …\nA native function is represented as an AST.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nAn input is a name, and a list of field elements (in …\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nAllows constants to be defined\nThe <code>else</code> keyword\nThe boolean value <code>false</code>\nA function\nThe <code>for</code> keyword\nThe <code>if</code> keyword\nThe <code>in</code> keyword for iterating\nNew variable\nThe <code>mut</code> keyword for mutable variables\nPublic input\nReturn from a function\nAllows custom structs to be defined\nThe boolean value <code>true</code>\nImporting a library\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nSince std::iter::Peekable in Rust advances the iterator, I …\nLike next() except that it also stores the last seen token …\nLike Self::bump but errors with <code>err</code> pointing to the latest …\nLike Self::bump but errors if the token is not <code>typ</code>\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nPeeks into the next token without advancing the iterator.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nA context for the parser.\nThe file we’re parsing\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nUsed mainly for error reporting, when we don’t have a …\nReturns a new unique node id.\nA counter used to uniquely identify different nodes in the …\nAn array access, for example: <code>lhs[idx]</code>\n<code>[ ... ]</code>\n<code>let lhs = rhs</code>\nany numbers\n<code>lhs &lt;op&gt; rhs</code>\n<code>true</code> or <code>false</code>\n<code>name { fields }</code>\n<code>lhs.rhs</code>\n<code>lhs(args)</code>\n<code>if cond { then_ } else { else_ }</code>\n<code>lhs.method_name(args)</code>\n<code>-expr</code>\n<code>!bool_expr</code>\na variable or a type. For example, <code>mod::A</code>, <code>x</code>, <code>y</code>, etc.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nParses until it finds something it doesn’t know, then …\nis it surrounded by parenthesis?\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nThis is a type imported from another module, …\nThis is a type imported from another module.\nAn array of a fixed size.\nThis could be the same as Field, but we use this to also …\nA boolean (<code>true</code> or <code>false</code>).\nCustom / user-defined types\nThe main primitive type. ’Nuf said.\nFunction.\nAny kind of text that can represent a type, a variable, a …\nThis is a local type, not imported from another module.\nMethod defined on a custom type.\nThe module preceding structs, functions, or variables.\nThings you can have in a scope (including the root scope).\n(pub, ident, type)\n(pub, ident, type)\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nParse a function, without the <code>fn</code> keyword.\nReturns a list of statement parsed until seeing the end of …\nYou can use SerdeAs with serde_with in order to serialize …\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nYou can use this module for serialization and …\nYou can use this to deserialize an arkworks type with …\nYou can use this to serialize an arkworks type with serde …\nList of builtin function signatures.\na function returns builtin functions\na function returns crypto functions\nReturns true if the given string is an hexadecimal string …\nReturns true if the given string is an identifier (starts …\nReturns true if the given string is an identifier or type\nReturns true if the given string is a number in decimal.\nReturns true if the given string is a type (first letter …\nThe environment we use to type check a noname program.\nThis takes the AST produced by the parser, and performs …\ntype checks a function call. Note that this can also be a …\nThis module defines the context (or environment) that gets …\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nSet to <code>None</code> if the function is defined in the main module.\nKeeps track of the signature of a user-defined function.\nKeeps track of the signature of a user-defined struct.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nSome type information on local variables that we want to …\nThe environment we use to type check functions.\nIf the variable is a constant or not.\nReturns the argument unchanged.\nReturns the argument unchanged.\nRetrieves type information on a variable, given a name. If …\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nReturns true if a scope is a prefix of our scope.\nIf the variable can be mutated or not.\nEnters a scoped block.\nCreates a new TypeEnv\nExits a scoped block.\nThe span of the variable declaration.\nStores type information about a local variable. Note that …\nSome type information.\nA cell in the execution trace.\nA constant value.\nRepresents a cell in the execution trace.\nOr it’s a constant (for example, I wrote <code>2</code> in the code).\nA public or private input to the function There’s an …\nEither it’s a hint and can be computed from the outside.\nThe signature of a hint function\nReturns the inverse of the given variable. Note that it …\nOr it’s a linear combination of internal circuit …\nA public output. This is tracked separately as public …\nA reference to a noname variable in the environment. …\nA variable’s actual value in the witness can be computed …\nRepresents a variable in the noname language, or an …\nA Var.\nRepresents a variable in the circuit, or a reference to …\nThe type of variable.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nThe span that created the variable.\nThe compiled circuit.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.")