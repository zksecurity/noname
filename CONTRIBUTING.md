# Contributing to noname

**You want to contribute to noname (or this file)? Look no further!**

## Where to start contributing?

The first thing you should do is try using noname by yourself. If something doesn't compile, or seems confusing, then that's an opportunity to make it compile, improve the error, or improve the documentation. In general if something is confusing or lacking please post an issue!

Make sure you `git clone` this repository and then create an alias for running your local copy of noname (e.g. `alias noname_dev="cargo run --manifest-path ABSOLUTE_PATH_TO_THE_CLONED_REPO`).

Once you're a bit more familiar with how to play with noname, check our [list of easy issues to start contributing](https://github.com/zksecurity/noname/issues?q=is%3Aopen+is%3Aissue+label%3Aeasy). Find something in there and ask for more information if you need it! We'll be happy to help you.

Generally, a good way to hit your goal is to start by writing the code you want to be able to compile and work backward to make it compile.

## How to learn about the inners of noname?
 
The [noname book](https://zksecurity.github.io/noname/) has a lot of information about internals, but keep in mind that it might not always be up to date (PRs to fix that are welcome). 

For a gentle intro, you can check [a walkthrough of the codebase here](https://www.youtube.com/live/pQer-ua73Vo), as well as our [series of videos here](https://cryptologie.net/article/573).

## Setup & Debugging

In vscode you can create such a file to easily debug a command. (But you should also see a "debug" button above any tests within vscode.)

```json
{
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

In addition, we have a `--server-mode` that allows you to look at the different stages of compilation visually within your browser. Run it as:

```
$ noname build --server-mode
```
