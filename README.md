# rust-mcp-filesystem

A small filesystem MCP server written in Rust. It exposes only a directory you explicitly choose and provides four tools:

- `list_directory`
- `read_text_file`
- `write_text_file`
- `create_directory`

It intentionally does **not** expose delete, rename, shell execution, or unrestricted absolute paths.

## Safety model

- All paths are relative to `--root`.
- Absolute paths and `..` traversal are rejected.
- Filesystem access uses `cap-std`, so operations remain capability-confined to the opened root, including when symlinks are present.
- Existing files are not replaced unless the caller sends `overwrite: true`.
- Overwrites use a sibling temporary file followed by an atomic rename.
- Reads and writes have a configurable byte limit.
- `--read-only` disables both write tools.
- Logs go to stderr; stdout is reserved for MCP JSON-RPC.

The root should still contain only files you are comfortable exposing to an AI client. Do not point it at your home directory, `/`, credential directories, or source trees containing secrets.

## Requirements

- A current stable Rust toolchain
- An MCP client that supports local stdio servers

## Build and test

```bash
cargo test
cargo build --release
```

The resulting executable is:

```text
target/release/rust-mcp-filesystem
```

## Run manually

Read/write mode:

```bash
./target/release/rust-mcp-filesystem \
  --root /absolute/path/to/safe-workspace
```

Read-only mode:

```bash
./target/release/rust-mcp-filesystem \
  --root /absolute/path/to/safe-workspace \
  --read-only
```

Optional limits:

```bash
./target/release/rust-mcp-filesystem \
  --root /absolute/path/to/safe-workspace \
  --max-file-bytes 4194304 \
  --max-dir-entries 5000
```

The same settings can be provided through `MCP_FS_ROOT`, `MCP_FS_READ_ONLY`, `MCP_FS_MAX_FILE_BYTES`, and `MCP_FS_MAX_DIR_ENTRIES`.

## MCP client configuration

Build the release binary and configure your client to launch it. A typical stdio configuration is:

```json
{
  "mcpServers": {
    "rust-filesystem": {
      "command": "/absolute/path/to/rust-mcp-filesystem",
      "args": [
        "--root",
        "/absolute/path/to/safe-workspace"
      ]
    }
  }
}
```

Start with `--read-only`, verify the exposed directory, and enable writes only when needed.

## Example tool arguments

```json
{"path":"."}
```

```json
{"path":"notes/plan.md"}
```

```json
{
  "path":"notes/plan.md",
  "content":"# Plan\n\nHello from MCP.\n",
  "overwrite":false
}
```

## Current scope

This starter handles UTF-8 text files. Binary reads/writes, file search, moving, deletion, permissions, and multiple allowed roots are deliberately left out of the first version.

