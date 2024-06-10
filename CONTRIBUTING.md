# Contributing to noname

You want to contribute to noname (or this file)? Look no further! 

The first thing you should look at, is our [list of easy issues](https://github.com/zksecurity/noname/issues?q=is%3Aopen+is%3Aissue+label%3Aeasy). Find something in there and ask for more information if you need it! We'll be happy to help you.

[The book](https://zksecurity.github.io/noname/) also has more information about internals, but keep in mind that it might not always be up to date (PRs to fix that are welcome). You can also check our [series of videos here](https://cryptologie.net/article/573).

General advice: 

* write the code you want to be able to compile, and if it does not compile post an issue. We can then discuss if this is something that should be implemented!
* setup debugging and step through the program to understand how noname compiles things, and how you can fix something that doesn't work! (see next section)
* any improvement to the user experience is welcome (better error messages, better documentation, better examples, better CLI, etc.)

## Setup & Debugging

In vscode you can create such a file to easily debug a command. (But you should also see a "debug" button above any tests within vscode.)

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "type": "lldb",
            "request": "launch",
            "name": "Debug",
            "program": "${workspaceFolder}/<executable file>",
            "args": [],
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "run cargo command",
            "type": "lldb",
            "request": "launch",
            "cwd": "${workspaceFolder}",
            "program": "${workspaceFolder}/target/debug/noname",
            "args": [
                "test",
                "--path",
                "examples/arithmetic.no",
                "--private-inputs",
                "{\"private_input\": \"2\"}",
                "--public-inputs",
                "{\"public_input\": \"2\"}",
                "--debug"
            ],
        },
    ]
}
```
