use std::{
    ffi::OsStr,
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result};
use cap_std::{ambient_authority, fs::Dir};
use clap::Parser;
use rmcp::{
    ErrorData as McpError, ServiceExt,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, ContentBlock},
    schemars, tool, tool_router,
    transport::stdio,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Parser)]
#[command(version, about = "Root-confined filesystem MCP server")]
struct Cli {
    /// Directory exposed by the server. Required unless MCP_FS_ROOT is set.
    #[arg(long, env = "MCP_FS_ROOT")]
    root: PathBuf,

    /// Disable every tool that changes the filesystem.
    #[arg(long, env = "MCP_FS_READ_ONLY", default_value_t = false)]
    read_only: bool,

    /// Maximum bytes accepted by read_text_file and write_text_file.
    #[arg(long, env = "MCP_FS_MAX_FILE_BYTES", default_value_t = 1_048_576)]
    max_file_bytes: u64,

    /// Maximum entries returned by one list_directory call.
    #[arg(long, env = "MCP_FS_MAX_DIR_ENTRIES", default_value_t = 2_000)]
    max_dir_entries: usize,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct PathArgs {
    /// A path relative to the configured root. Absolute paths and `..` are rejected.
    path: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ListDirectoryArgs {
    /// A directory relative to the configured root. Defaults to `.`.
    path: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct WriteTextFileArgs {
    /// A file path relative to the configured root.
    path: String,
    /// UTF-8 text to write.
    content: String,
    /// Replace an existing file. Defaults to false.
    #[serde(default)]
    overwrite: bool,
}

#[derive(Debug, Serialize)]
struct DirectoryEntry {
    name: String,
    kind: &'static str,
}

#[derive(Clone)]
struct FilesystemServer {
    root: Arc<Dir>,
    root_display: Arc<PathBuf>,
    read_only: bool,
    max_file_bytes: u64,
    max_dir_entries: usize,
}

impl FilesystemServer {
    fn new(cli: &Cli) -> Result<Self> {
        let root_display = std::fs::canonicalize(&cli.root)
            .with_context(|| format!("cannot open root directory: {}", cli.root.display()))?;
        let metadata = std::fs::metadata(&root_display)?;
        anyhow::ensure!(metadata.is_dir(), "root is not a directory");

        // cap-std keeps all later operations capability-confined to this directory.
        let root = Dir::open_ambient_dir(&root_display, ambient_authority())?;

        Ok(Self {
            root: Arc::new(root),
            root_display: Arc::new(root_display),
            read_only: cli.read_only,
            max_file_bytes: cli.max_file_bytes,
            max_dir_entries: cli.max_dir_entries,
        })
    }

    fn success(message: impl Into<String>) -> CallToolResult {
        CallToolResult::success(vec![ContentBlock::text(message.into())])
    }

    fn failure(message: impl Into<String>) -> CallToolResult {
        CallToolResult::error(vec![ContentBlock::text(message.into())])
    }

    fn clean_path(raw: &str, allow_root: bool) -> std::result::Result<PathBuf, String> {
        if raw.as_bytes().contains(&0) {
            return Err("path contains a NUL byte".into());
        }

        let path = Path::new(raw);
        if path.is_absolute() {
            return Err("absolute paths are not allowed".into());
        }

        let mut cleaned = PathBuf::new();
        for component in path.components() {
            match component {
                Component::Normal(part) => cleaned.push(part),
                Component::CurDir => {}
                Component::ParentDir => return Err("parent traversal (`..`) is not allowed".into()),
                Component::RootDir | Component::Prefix(_) => {
                    return Err("absolute paths are not allowed".into());
                }
            }
        }

        if cleaned.as_os_str().is_empty() {
            if allow_root {
                Ok(PathBuf::from("."))
            } else {
                Err("path must name an item below the configured root".into())
            }
        } else {
            Ok(cleaned)
        }
    }

    fn ensure_writable(&self) -> std::result::Result<(), String> {
        if self.read_only {
            Err("server is running in read-only mode".into())
        } else {
            Ok(())
        }
    }

    fn read_text(&self, raw_path: &str) -> std::result::Result<String, String> {
        let path = Self::clean_path(raw_path, false)?;
        let file = self
            .root
            .open(&path)
            .map_err(|e| format!("cannot open file: {e}"))?;
        let metadata = file
            .metadata()
            .map_err(|e| format!("cannot inspect file: {e}"))?;

        if !metadata.is_file() {
            return Err("path is not a regular file".into());
        }
        if metadata.len() > self.max_file_bytes {
            return Err(format!(
                "file is {} bytes; limit is {} bytes",
                metadata.len(),
                self.max_file_bytes
            ));
        }

        let mut bytes = Vec::new();
        file.take(self.max_file_bytes.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(|e| format!("cannot read file: {e}"))?;
        if bytes.len() as u64 > self.max_file_bytes {
            return Err(format!(
                "file grew beyond the {}-byte limit",
                self.max_file_bytes
            ));
        }

        String::from_utf8(bytes)
            .map_err(|_| "file is not valid UTF-8; this server reads text only".into())
    }

    fn list_dir(&self, raw_path: &str) -> std::result::Result<String, String> {
        let path = Self::clean_path(raw_path, true)?;
        let dir = self
            .root
            .open_dir(&path)
            .map_err(|e| format!("cannot open directory: {e}"))?;
        let entries = dir
            .entries()
            .map_err(|e| format!("cannot list directory: {e}"))?;
        let mut result = Vec::new();

        for entry in entries.take(self.max_dir_entries + 1) {
            let entry = entry.map_err(|e| format!("cannot read directory entry: {e}"))?;
            if result.len() == self.max_dir_entries {
                return Err(format!(
                    "directory contains more than {} entries",
                    self.max_dir_entries
                ));
            }

            let file_type = entry
                .file_type()
                .map_err(|e| format!("cannot inspect entry: {e}"))?;
            let kind = if file_type.is_dir() {
                "directory"
            } else if file_type.is_file() {
                "file"
            } else if file_type.is_symlink() {
                "symlink"
            } else {
                "other"
            };

            result.push(DirectoryEntry {
                name: entry.file_name().to_string_lossy().into_owned(),
                kind,
            });
        }

        result.sort_by(|a, b| a.name.cmp(&b.name));
        serde_json::to_string_pretty(&result).map_err(|e| format!("cannot encode result: {e}"))
    }

    fn write_text(
        &self,
        raw_path: &str,
        content: &str,
        overwrite: bool,
    ) -> std::result::Result<String, String> {
        self.ensure_writable()?;
        let path = Self::clean_path(raw_path, false)?;

        if content.len() as u64 > self.max_file_bytes {
            return Err(format!(
                "content is {} bytes; limit is {} bytes",
                content.len(),
                self.max_file_bytes
            ));
        }

        if !overwrite {
            let mut options = cap_std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            let mut file = self
                .root
                .open_with(&path, &options)
                .map_err(|e| format!("cannot create file (already exists?): {e}"))?;
            file.write_all(content.as_bytes())
                .map_err(|e| format!("cannot write file: {e}"))?;
            file.sync_all()
                .map_err(|e| format!("cannot sync file: {e}"))?;
        } else {
            // Write a sibling temporary file, sync it, then atomically replace the target.
            let parent = path.parent().unwrap_or_else(|| Path::new("."));
            let name = path
                .file_name()
                .and_then(OsStr::to_str)
                .ok_or("invalid destination filename")?;
            let temp_name = format!(".{name}.mcp-{}.tmp", Uuid::new_v4());
            let temp_path = parent.join(temp_name);

            let write_result = (|| -> std::io::Result<()> {
                let mut options = cap_std::fs::OpenOptions::new();
                options.write(true).create_new(true);
                let mut file = self.root.open_with(&temp_path, &options)?;
                file.write_all(content.as_bytes())?;
                file.sync_all()?;
                drop(file);
                self.root.rename(&temp_path, &self.root, &path)?;
                Ok(())
            })();

            if let Err(error) = write_result {
                let _ = self.root.remove_file(&temp_path);
                return Err(format!("cannot replace file: {error}"));
            }
        }

        Ok(format!(
            "wrote {} bytes to {}",
            content.len(),
            path.display()
        ))
    }

    fn create_dir(&self, raw_path: &str) -> std::result::Result<String, String> {
        self.ensure_writable()?;
        let path = Self::clean_path(raw_path, false)?;
        self.root
            .create_dir_all(&path)
            .map_err(|e| format!("cannot create directory: {e}"))?;
        Ok(format!("created directory {}", path.display()))
    }
}

#[tool_router(server_handler)]
impl FilesystemServer {
    #[tool(
        description = "List files and directories below the configured root. Paths are relative; use `.` for the root."
    )]
    fn list_directory(
        &self,
        Parameters(args): Parameters<ListDirectoryArgs>,
    ) -> std::result::Result<CallToolResult, McpError> {
        let path = args.path.as_deref().unwrap_or(".");
        Ok(match self.list_dir(path) {
            Ok(json) => Self::success(json),
            Err(error) => Self::failure(error),
        })
    }

    #[tool(description = "Read one UTF-8 text file below the configured root.")]
    fn read_text_file(
        &self,
        Parameters(args): Parameters<PathArgs>,
    ) -> std::result::Result<CallToolResult, McpError> {
        Ok(match self.read_text(&args.path) {
            Ok(content) => Self::success(content),
            Err(error) => Self::failure(error),
        })
    }

    #[tool(
        description = "Write one UTF-8 text file below the configured root. Existing files require overwrite=true."
    )]
    fn write_text_file(
        &self,
        Parameters(args): Parameters<WriteTextFileArgs>,
    ) -> std::result::Result<CallToolResult, McpError> {
        Ok(
            match self.write_text(&args.path, &args.content, args.overwrite) {
                Ok(message) => Self::success(message),
                Err(error) => Self::failure(error),
            },
        )
    }

    #[tool(
        description = "Create a directory, including missing parent directories, below the configured root."
    )]
    fn create_directory(
        &self,
        Parameters(args): Parameters<PathArgs>,
    ) -> std::result::Result<CallToolResult, McpError> {
        Ok(match self.create_dir(&args.path) {
            Ok(message) => Self::success(message),
            Err(error) => Self::failure(error),
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rust_mcp_filesystem=info".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let server = FilesystemServer::new(&cli)?;

    tracing::info!(
        root = %server.root_display.display(),
        read_only = server.read_only,
        "starting filesystem MCP server"
    );

    let service = server.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::FilesystemServer;
    use std::path::PathBuf;

    #[test]
    fn accepts_normal_relative_paths() {
        assert_eq!(
            FilesystemServer::clean_path("notes/hello.txt", false).unwrap(),
            PathBuf::from("notes/hello.txt")
        );
    }

    #[test]
    fn rejects_parent_traversal() {
        assert!(FilesystemServer::clean_path("../secret.txt", false).is_err());
        assert!(FilesystemServer::clean_path("notes/../../secret.txt", false).is_err());
    }

    #[test]
    fn rejects_absolute_paths() {
        assert!(FilesystemServer::clean_path("/etc/passwd", false).is_err());
    }
}
