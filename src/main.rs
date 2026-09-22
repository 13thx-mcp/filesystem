use std::{
    collections::BTreeMap,
    ffi::OsStr,
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex, Weak},
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
use sha2::{Digest, Sha256};
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
    /// Required revision for a compare-and-swap replacement of an existing file.
    expected_revision: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ReadRangeArgs {
    /// A file path relative to the configured root.
    path: String,
    /// Zero-based byte offset.
    offset: u64,
    /// Maximum bytes to return. Must not exceed server file limit.
    max_bytes: u64,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SearchTextArgs {
    /// A file path relative to the configured root.
    path: String,
    /// Literal UTF-8 text to find. Empty queries are rejected.
    query: String,
    /// Maximum matches returned. Defaults to 100 and is capped at 100.
    #[serde(default = "default_search_matches")]
    max_matches: usize,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct PatchTextFileArgs {
    /// A file path relative to the configured root.
    path: String,
    /// SHA-256 revision returned by file_metadata/read_text_file_range.
    expected_revision: String,
    /// Zero-based inclusive byte offset of replaced range.
    start_byte: u64,
    /// Zero-based exclusive byte offset of replaced range.
    end_byte: u64,
    /// UTF-8 replacement text.
    replacement: String,
}

#[derive(Debug, Serialize)]
struct FileMetadata {
    path: String,
    size_bytes: u64,
    revision: String,
}

#[derive(Debug, Serialize)]
struct WriteResult {
    path: String,
    size_bytes: usize,
    revision: String,
}

#[derive(Debug, Serialize)]
struct SearchMatch {
    line: u64,
    start_byte: u64,
    end_byte: u64,
    preview: String,
    preview_truncated: bool,
}

#[derive(Debug, Serialize)]
struct SearchResult {
    path: String,
    revision: String,
    matches: Vec<SearchMatch>,
    truncated: bool,
}

fn default_search_matches() -> usize {
    100
}

const MAX_SEARCH_MATCHES: usize = 100;
const MAX_SEARCH_LINE_BYTES: usize = 4096;

#[derive(Debug, Serialize)]
struct DirectoryEntry {
    name: String,
    kind: &'static str,
}

#[derive(Clone)]
struct FilesystemServer {
    root: Arc<Dir>,
    root_display: Arc<PathBuf>,
    write_locks: Arc<Mutex<BTreeMap<PathBuf, Weak<Mutex<()>>>>>,
    read_only: bool,
    max_file_bytes: u64,
    max_dir_entries: usize,
    #[cfg(test)]
    before_cas_commit: Option<Arc<dyn Fn() + Send + Sync>>,
    #[cfg(test)]
    write_failure: Option<WriteFailure>,
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum WriteFailure {
    AfterTempSync,
    BeforeRename,
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
            write_locks: Arc::new(Mutex::new(BTreeMap::new())),
            read_only: cli.read_only,
            max_file_bytes: cli.max_file_bytes,
            max_dir_entries: cli.max_dir_entries,
            #[cfg(test)]
            before_cas_commit: None,
            #[cfg(test)]
            write_failure: None,
        })
    }

    fn write_lock(&self, path: &Path) -> Arc<Mutex<()>> {
        let mut locks = self
            .write_locks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(lock) = locks.get(path).and_then(Weak::upgrade) {
            return lock;
        }
        let lock = Arc::new(Mutex::new(()));
        locks.insert(path.to_path_buf(), Arc::downgrade(&lock));
        lock
    }

    fn success(message: impl Into<String>) -> CallToolResult {
        CallToolResult::success(vec![ContentBlock::text(message.into())])
    }

    fn failure(message: impl Into<String>) -> CallToolResult {
        let message = message.into();
        CallToolResult::structured_error(filesystem_error(&message))
    }

    fn success_json(value: impl Serialize) -> std::result::Result<CallToolResult, String> {
        serde_json::to_string(&value)
            .map(Self::success)
            .map_err(|error| format!("cannot encode result: {error}"))
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

    fn ensure_no_symlink_components(&self, path: &Path) -> std::result::Result<(), String> {
        let mut prefix = PathBuf::new();
        for component in path.components() {
            let Component::Normal(part) = component else {
                continue;
            };
            prefix.push(part);
            match self.root.symlink_metadata(&prefix) {
                Ok(metadata) if metadata.file_type().is_symlink() => {
                    return Err(format!(
                        "path contains a symlink component: {}",
                        prefix.display()
                    ));
                }
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
                Err(error) => {
                    return Err(format!(
                        "cannot inspect path component {}: {error}",
                        prefix.display()
                    ));
                }
            }
        }
        Ok(())
    }

    fn read_text(&self, raw_path: &str) -> std::result::Result<String, String> {
        let (_, bytes) = self.read_bytes(raw_path)?;
        String::from_utf8(bytes)
            .map_err(|_| "file is not valid UTF-8; this server reads text only".into())
    }

    fn read_bytes(&self, raw_path: &str) -> std::result::Result<(PathBuf, Vec<u8>), String> {
        let path = Self::clean_path(raw_path, false)?;
        self.ensure_no_symlink_components(&path)?;
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

        Ok((path, bytes))
    }

    fn metadata(&self, raw_path: &str) -> std::result::Result<FileMetadata, String> {
        let (path, bytes) = self.read_bytes(raw_path)?;
        Ok(FileMetadata {
            path: path.display().to_string(),
            size_bytes: bytes.len() as u64,
            revision: revision(&bytes),
        })
    }

    fn read_range(&self, args: &ReadRangeArgs) -> std::result::Result<FileMetadataText, String> {
        if args.max_bytes == 0 || args.max_bytes > self.max_file_bytes {
            return Err(format!(
                "max_bytes must be between 1 and {}",
                self.max_file_bytes
            ));
        }
        let (path, bytes) = self.read_bytes(&args.path)?;
        let start = usize::try_from(args.offset).map_err(|_| "offset exceeds platform limit")?;
        if start > bytes.len() {
            return Err("offset is beyond end of file".into());
        }
        let end = start
            .saturating_add(args.max_bytes as usize)
            .min(bytes.len());
        let text = String::from_utf8(bytes[start..end].to_vec())
            .map_err(|_| "requested range is not valid UTF-8 text".to_owned())?;
        Ok(FileMetadataText {
            path: path.display().to_string(),
            size_bytes: bytes.len() as u64,
            revision: revision(&bytes),
            offset: args.offset,
            text,
            truncated: end < bytes.len(),
        })
    }

    fn search_text_impl(&self, args: &SearchTextArgs) -> std::result::Result<SearchResult, String> {
        if args.query.is_empty() || args.query.len() > MAX_SEARCH_LINE_BYTES {
            return Err("query must be between 1 and 4096 bytes".into());
        }
        if args.max_matches == 0 || args.max_matches > MAX_SEARCH_MATCHES {
            return Err(format!(
                "max_matches must be between 1 and {MAX_SEARCH_MATCHES}"
            ));
        }
        let (path, bytes) = self.read_bytes(&args.path)?;
        let text = std::str::from_utf8(&bytes)
            .map_err(|_| "file is not valid UTF-8; this server searches text only")?;
        let mut matches = Vec::new();
        let mut offset = 0usize;
        let mut truncated = false;
        for (index, line) in text.split_inclusive('\n').enumerate() {
            for (start, found) in line.match_indices(&args.query) {
                if matches.len() == args.max_matches {
                    truncated = true;
                    break;
                }
                let (preview, preview_truncated) = bounded_preview(line, MAX_SEARCH_LINE_BYTES);
                matches.push(SearchMatch {
                    line: index as u64 + 1,
                    start_byte: (offset + start) as u64,
                    end_byte: (offset + start + found.len()) as u64,
                    preview,
                    preview_truncated,
                });
            }
            if truncated {
                break;
            }
            offset += line.len();
        }
        Ok(SearchResult {
            path: path.display().to_string(),
            revision: revision(&bytes),
            matches,
            truncated,
        })
    }

    fn patch_text(&self, args: &PatchTextFileArgs) -> std::result::Result<WriteResult, String> {
        let (_, bytes) = self.read_bytes(&args.path)?;
        let actual = revision(&bytes);
        if actual != args.expected_revision {
            return Err(format!(
                "stale_revision: expected {}, current {actual}",
                args.expected_revision
            ));
        }
        let text = String::from_utf8(bytes)
            .map_err(|_| "file is not valid UTF-8; this server patches text only")?;
        let start =
            usize::try_from(args.start_byte).map_err(|_| "start_byte exceeds platform limit")?;
        let end = usize::try_from(args.end_byte).map_err(|_| "end_byte exceeds platform limit")?;
        if start > end
            || end > text.len()
            || !text.is_char_boundary(start)
            || !text.is_char_boundary(end)
        {
            return Err("patch range must be ordered UTF-8 byte boundaries within file".into());
        }
        let mut patched = String::with_capacity(
            start
                .saturating_add(args.replacement.len())
                .saturating_add(text.len() - end),
        );
        patched.push_str(&text[..start]);
        patched.push_str(&args.replacement);
        patched.push_str(&text[end..]);
        if patched.len() as u64 > self.max_file_bytes {
            return Err(format!(
                "patched content is {} bytes; limit is {} bytes",
                patched.len(),
                self.max_file_bytes
            ));
        }
        self.write_text(&args.path, &patched, true, Some(&args.expected_revision))
    }

    fn list_dir(&self, raw_path: &str) -> std::result::Result<String, String> {
        let path = Self::clean_path(raw_path, true)?;
        self.ensure_no_symlink_components(&path)?;
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
        expected_revision: Option<&str>,
    ) -> std::result::Result<WriteResult, String> {
        self.ensure_writable()?;
        let path = Self::clean_path(raw_path, false)?;
        self.ensure_no_symlink_components(&path)?;
        let write_lock = self.write_lock(&path);
        let _write_guard = write_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        if let Some(expected) = expected_revision {
            let actual = match self.read_bytes(raw_path) {
                Ok((_, bytes)) => revision(&bytes),
                Err(_) => return Err("stale_revision: target file does not exist".into()),
            };
            if expected != actual {
                return Err(format!(
                    "stale_revision: expected {expected}, current {actual}"
                ));
            }
        }

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
                file.sync_all()
            })();
            if let Err(error) = write_result {
                let _ = self.root.remove_file(&temp_path);
                return Err(format!("cannot replace file: {error}"));
            }
            #[cfg(test)]
            if matches!(self.write_failure, Some(WriteFailure::AfterTempSync)) {
                let _ = self.root.remove_file(&temp_path);
                return Err("cannot replace file: injected temporary sync failure".into());
            }
            #[cfg(test)]
            if let Some(hook) = &self.before_cas_commit {
                hook();
            }
            if let Some(expected) = expected_revision {
                let actual = match self.read_bytes(raw_path) {
                    Ok((_, bytes)) => revision(&bytes),
                    Err(_) => {
                        let _ = self.root.remove_file(&temp_path);
                        return Err("stale_revision: target file does not exist".into());
                    }
                };
                if expected != actual {
                    let _ = self.root.remove_file(&temp_path);
                    return Err(format!(
                        "stale_revision: expected {expected}, current {actual}"
                    ));
                }
            }
            #[cfg(test)]
            if matches!(self.write_failure, Some(WriteFailure::BeforeRename)) {
                let _ = self.root.remove_file(&temp_path);
                return Err("cannot replace file: injected rename failure".into());
            }
            let replace_result = self.root.rename(&temp_path, &self.root, &path);
            if let Err(error) = replace_result {
                let _ = self.root.remove_file(&temp_path);
                return Err(format!("cannot replace file: {error}"));
            }
        }

        Ok(WriteResult {
            path: path.display().to_string(),
            size_bytes: content.len(),
            revision: revision(content.as_bytes()),
        })
    }

    fn create_dir(&self, raw_path: &str) -> std::result::Result<String, String> {
        self.ensure_writable()?;
        let path = Self::clean_path(raw_path, false)?;
        self.ensure_no_symlink_components(&path)?;
        self.root
            .create_dir_all(&path)
            .map_err(|e| format!("cannot create directory: {e}"))?;
        Ok(format!("created directory {}", path.display()))
    }
}

#[derive(Debug, Serialize)]
struct FileMetadataText {
    path: String,
    size_bytes: u64,
    revision: String,
    offset: u64,
    text: String,
    truncated: bool,
}

fn revision(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn filesystem_error(message: &str) -> serde_json::Value {
    let code = if message.starts_with("stale_revision:") {
        "stale_revision"
    } else {
        "filesystem_error"
    };
    serde_json::json!({
        "code": code,
        "message": message,
        "retryable": code == "stale_revision"
    })
}

fn bounded_preview(line: &str, max_bytes: usize) -> (String, bool) {
    if line.len() <= max_bytes {
        return (line.to_owned(), false);
    }
    let mut end = max_bytes;
    while !line.is_char_boundary(end) {
        end -= 1;
    }
    (line[..end].to_owned(), true)
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

    #[tool(description = "Return bounded file size and SHA-256 revision metadata.")]
    fn file_metadata(
        &self,
        Parameters(args): Parameters<PathArgs>,
    ) -> std::result::Result<CallToolResult, McpError> {
        match self.metadata(&args.path).and_then(Self::success_json) {
            Ok(result) => Ok(result),
            Err(error) => Ok(Self::failure(error)),
        }
    }

    #[tool(description = "Read a bounded UTF-8 byte range with SHA-256 revision metadata.")]
    fn read_text_file_range(
        &self,
        Parameters(args): Parameters<ReadRangeArgs>,
    ) -> std::result::Result<CallToolResult, McpError> {
        match self.read_range(&args).and_then(Self::success_json) {
            Ok(result) => Ok(result),
            Err(error) => Ok(Self::failure(error)),
        }
    }

    #[tool(description = "Search bounded UTF-8 text and return literal match byte ranges.")]
    fn search_text(
        &self,
        Parameters(args): Parameters<SearchTextArgs>,
    ) -> std::result::Result<CallToolResult, McpError> {
        match self.search_text_impl(&args).and_then(Self::success_json) {
            Ok(result) => Ok(result),
            Err(error) => Ok(Self::failure(error)),
        }
    }

    #[tool(
        description = "Write one UTF-8 text file below the configured root. Existing files require overwrite=true."
    )]
    fn write_text_file(
        &self,
        Parameters(args): Parameters<WriteTextFileArgs>,
    ) -> std::result::Result<CallToolResult, McpError> {
        Ok(
            match self.write_text(
                &args.path,
                &args.content,
                args.overwrite,
                args.expected_revision.as_deref(),
            ) {
                Ok(result) => match Self::success_json(result) {
                    Ok(result) => result,
                    Err(error) => Self::failure(error),
                },
                Err(error) => Self::failure(error),
            },
        )
    }

    #[tool(description = "Apply one revision-checked UTF-8 byte-range replacement atomically.")]
    fn patch_text_file(
        &self,
        Parameters(args): Parameters<PatchTextFileArgs>,
    ) -> std::result::Result<CallToolResult, McpError> {
        match self.patch_text(&args).and_then(Self::success_json) {
            Ok(result) => Ok(result),
            Err(error) => Ok(Self::failure(error)),
        }
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
    use super::{
        Cli, FilesystemServer, MAX_SEARCH_LINE_BYTES, MAX_SEARCH_MATCHES, PatchTextFileArgs,
        ReadRangeArgs, SearchTextArgs, WriteFailure, filesystem_error,
    };
    use std::{
        fs,
        path::PathBuf,
        sync::{
            Arc, Barrier,
            atomic::{AtomicU64, Ordering},
        },
        thread,
    };

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn fixture() -> (PathBuf, FilesystemServer) {
        let root = std::env::temp_dir().join(format!(
            "m7-filesystem-{}-{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).unwrap();
        let cli = Cli {
            root: root.clone(),
            read_only: false,
            max_file_bytes: 1024,
            max_dir_entries: 10,
        };
        (root, FilesystemServer::new(&cli).unwrap())
    }

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

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_components_for_reads_and_writes() {
        let (root, server) = fixture();
        let outside = root.with_extension("outside");
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("secret.txt"), "secret\n").unwrap();
        std::os::unix::fs::symlink(&outside, root.join("escape")).unwrap();

        assert!(server.read_text("escape/secret.txt").is_err());
        assert!(
            server
                .write_text("escape/new.txt", "blocked\n", true, None)
                .is_err()
        );
        assert!(!outside.join("new.txt").exists());
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(outside).unwrap();
    }

    #[test]
    fn stale_revision_has_typed_error_code() {
        assert_eq!(
            filesystem_error("stale_revision: old")["code"],
            "stale_revision"
        );
        assert_eq!(
            filesystem_error("cannot open file")["code"],
            "filesystem_error"
        );
    }

    #[test]
    fn metadata_range_and_revision_cas_detect_external_change() {
        let (root, server) = fixture();
        fs::write(root.join("note.txt"), "alpha\nbeta\n").unwrap();

        let metadata = server.metadata("note.txt").unwrap();
        assert_eq!(metadata.size_bytes, 11);
        assert!(metadata.revision.starts_with("sha256:"));

        let range = server
            .read_range(&ReadRangeArgs {
                path: "note.txt".into(),
                offset: 6,
                max_bytes: 4,
            })
            .unwrap();
        assert_eq!(range.text, "beta");
        assert!(range.truncated);

        fs::write(root.join("note.txt"), "external\n").unwrap();
        let conflict = server
            .write_text("note.txt", "lost\n", true, Some(&metadata.revision))
            .unwrap_err();
        assert!(conflict.starts_with("stale_revision:"));
        assert_eq!(server.read_text("note.txt").unwrap(), "external\n");

        let external_revision = server.metadata("note.txt").unwrap().revision;
        let written = server
            .write_text("note.txt", "next\n", true, Some(&external_revision))
            .unwrap();
        assert_eq!(server.read_text("note.txt").unwrap(), "next\n");
        assert_eq!(
            written.revision,
            server.metadata("note.txt").unwrap().revision
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn range_and_search_enforce_byte_line_and_match_bounds() {
        let (root, mut server) = fixture();
        server.max_file_bytes = 8192;
        let line = format!("needle{}", "x".repeat(5000));
        fs::write(root.join("note.txt"), format!("{line}\nneedle\n")).unwrap();

        assert!(
            server
                .read_range(&ReadRangeArgs {
                    path: "note.txt".into(),
                    offset: 0,
                    max_bytes: 0,
                })
                .is_err()
        );
        assert!(
            server
                .read_range(&ReadRangeArgs {
                    path: "note.txt".into(),
                    offset: 9_000,
                    max_bytes: 1,
                })
                .is_err()
        );
        assert!(
            server
                .search_text_impl(&SearchTextArgs {
                    path: "note.txt".into(),
                    query: "needle".into(),
                    max_matches: MAX_SEARCH_MATCHES + 1,
                })
                .is_err()
        );

        let search = server
            .search_text_impl(&SearchTextArgs {
                path: "note.txt".into(),
                query: "needle".into(),
                max_matches: 1,
            })
            .unwrap();
        assert!(search.truncated);
        assert_eq!(search.matches.len(), 1);
        assert!(search.matches[0].preview_truncated);
        assert!(search.matches[0].preview.len() <= MAX_SEARCH_LINE_BYTES);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn search_and_patch_are_bounded_and_revision_checked() {
        let (root, server) = fixture();
        fs::write(root.join("note.txt"), "αbeta\nbeta\n").unwrap();
        let metadata = server.metadata("note.txt").unwrap();

        let search = server
            .search_text_impl(&SearchTextArgs {
                path: "note.txt".into(),
                query: "beta".into(),
                max_matches: 1,
            })
            .unwrap();
        assert_eq!(search.matches.len(), 1);
        assert_eq!(search.matches[0].line, 1);
        assert_eq!(search.matches[0].start_byte, 2);
        assert!(search.truncated);

        let patched = server
            .patch_text(&PatchTextFileArgs {
                path: "note.txt".into(),
                expected_revision: metadata.revision.clone(),
                start_byte: 2,
                end_byte: 6,
                replacement: "next".into(),
            })
            .unwrap();
        assert_eq!(server.read_text("note.txt").unwrap(), "αnext\nbeta\n");
        assert_eq!(
            patched.revision,
            server.metadata("note.txt").unwrap().revision
        );

        let invalid_boundary = server.patch_text(&PatchTextFileArgs {
            path: "note.txt".into(),
            expected_revision: patched.revision.clone(),
            start_byte: 1,
            end_byte: 2,
            replacement: "x".into(),
        });
        assert!(invalid_boundary.is_err());

        let stale = server.patch_text(&PatchTextFileArgs {
            path: "note.txt".into(),
            expected_revision: metadata.revision,
            start_byte: 2,
            end_byte: 6,
            replacement: "lost".into(),
        });
        assert!(stale.unwrap_err().starts_with("stale_revision:"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cas_rechecks_before_rename_and_serializes_same_revision_writers() {
        let (root, mut server) = fixture();
        fs::write(root.join("note.txt"), "initial\n").unwrap();
        let initial = server.metadata("note.txt").unwrap().revision;
        let external_path = root.join("note.txt");
        server.before_cas_commit = Some(Arc::new(move || {
            fs::write(&external_path, "external\n").unwrap();
        }));
        let stale = server
            .write_text("note.txt", "lost\n", true, Some(&initial))
            .unwrap_err();
        assert!(stale.starts_with("stale_revision:"));
        assert_eq!(server.read_text("note.txt").unwrap(), "external\n");

        server.before_cas_commit = None;
        let revision = server.metadata("note.txt").unwrap().revision;
        let barrier = Arc::new(Barrier::new(3));
        let first_server = server.clone();
        let first_barrier = Arc::clone(&barrier);
        let first = thread::spawn(move || {
            first_barrier.wait();
            first_server.write_text("note.txt", "first\n", true, Some(&revision))
        });
        let revision = server.metadata("note.txt").unwrap().revision;
        let second_server = server.clone();
        let second_barrier = Arc::clone(&barrier);
        let second = thread::spawn(move || {
            second_barrier.wait();
            second_server.write_text("note.txt", "second\n", true, Some(&revision))
        });
        barrier.wait();
        let results = [first.join().unwrap(), second.join().unwrap()];
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(
            results
                .iter()
                .filter(|result| result
                    .as_ref()
                    .is_err_and(|error| error.starts_with("stale_revision:")))
                .count(),
            1
        );
        assert!(matches!(
            server.read_text("note.txt").unwrap().as_str(),
            "first\n" | "second\n"
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn injected_replace_failures_keep_old_bytes_and_remove_temporary_file() {
        let (root, mut server) = fixture();
        fs::write(root.join("note.txt"), "old\n").unwrap();
        let revision = server.metadata("note.txt").unwrap().revision;
        for failure in [WriteFailure::AfterTempSync, WriteFailure::BeforeRename] {
            server.write_failure = Some(failure);
            assert!(
                server
                    .write_text("note.txt", "new\n", true, Some(&revision))
                    .is_err()
            );
            assert_eq!(server.read_text("note.txt").unwrap(), "old\n");
            assert!(fs::read_dir(&root).unwrap().all(|entry| {
                !entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .contains(".mcp-")
            }));
        }
        fs::remove_dir_all(root).unwrap();
    }
}
