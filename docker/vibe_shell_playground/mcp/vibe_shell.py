import os
import stat
import shutil
import base64
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from fastmcp import FastMCP

mcp = FastMCP(name="VibeShellMCP")

WORKSPACE_ROOT = Path(
    os.environ.get("VIBESHELL_WORKSPACE", os.getcwd())
).expanduser().resolve()

MAX_READ_BYTES = 256 * 1024
MAX_FETCH_BYTES = 512 * 1024
DEFAULT_CMD_TIMEOUT = 60


def ok(data: Any = None, message: str = "OK") -> Dict[str, Any]:
    return {
        "success": True,
        "data": data,
        "error": None,
        "message": message,
    }


def fail(error: str, message: str = "Operation failed") -> Dict[str, Any]:
    return {
        "success": False,
        "data": None,
        "error": error,
        "message": message,
    }


def _strip_file_scheme(path_str: str) -> str:
    return path_str[7:] if path_str.startswith("file://") else path_str


def _resolve_in_workspace(path_str: str) -> Path:
    """
    Resolve a path and ensure it stays inside the workspace root.

    Args:
        path_str (str): A relative path, absolute path, path with '~',
            or file:// URI.

    Returns:
        Path: The resolved absolute path inside WORKSPACE_ROOT.

    Raises:
        ValueError: If the resolved path is outside WORKSPACE_ROOT.
    """
    raw = os.path.expanduser(_strip_file_scheme(path_str))

    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = WORKSPACE_ROOT / candidate

    resolved = candidate.resolve()

    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError:
        raise ValueError(f"Path is outside workspace: {resolved}")

    return resolved


@mcp.tool()
def read_file(filepath: str) -> Dict[str, Any]:
    """
    Read a file inside the workspace.

    Text files are returned as UTF-8 strings. Binary files are returned
    as base64 with the prefix "base64:". Large files may be truncated.

    Args:
        filepath (str): Path to the file inside the workspace. Supports
            relative paths, absolute paths, '~', and file:// URIs, as long
            as the final resolved path stays inside the workspace.

    Returns:
        Dict[str, Any]: A unified result object:
            {
                "success": bool,
                "data": {
                    "path": str,
                    "content": str,
                    "truncated": bool
                } | None,
                "error": str | None,
                "message": str
            }
    """
    try:
        path = _resolve_in_workspace(filepath)
        if not path.exists():
            return fail(f"File not found: {filepath}", message="Failed to read file.")
        if not path.is_file():
            return fail(f"Not a file: {filepath}", message="Failed to read file.")

        raw = path.read_bytes()
        truncated = False
        if len(raw) > MAX_READ_BYTES:
            raw = raw[:MAX_READ_BYTES]
            truncated = True

        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            content = "base64:" + base64.b64encode(raw).decode("ascii")

        return ok(
            {
                "path": str(path.relative_to(WORKSPACE_ROOT)),
                "content": content,
                "truncated": truncated,
            },
            message="File read successfully.",
        )
    except Exception as e:
        return fail(str(e), message="Failed to read file.")


@mcp.tool()
def create_new_file(filepath: str, contents: str) -> Dict[str, Any]:
    """
    Create a new UTF-8 text file inside the workspace.

    This tool will not overwrite an existing file. Parent directories are
    created automatically when needed.

    Args:
        filepath (str): Destination path inside the workspace.
        contents (str): Full text content to write into the new file.

    Returns:
        Dict[str, Any]: A unified result object:
            {
                "success": bool,
                "data": {
                    "path": str
                } | None,
                "error": str | None,
                "message": str
            }
    """
    try:
        path = _resolve_in_workspace(filepath)

        if path.exists():
            return fail(
                f"File already exists: {filepath}",
                message="Failed to create file.",
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

        return ok(
            {"path": str(path.relative_to(WORKSPACE_ROOT))},
            message="File created successfully.",
        )
    except Exception as e:
        return fail(str(e), message="Failed to create file.")


@mcp.tool()
def single_find_and_replace(
    filepath: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> Dict[str, Any]:
    """
    Replace exact text inside a file.

    Read the file first, then replace the exact text with matching whitespace
    and indentation.

    Args:
        filepath (str): Path to the target file inside the workspace.
        old_string (str): Exact text to search for.
        new_string (str): Replacement text.
        replace_all (bool): If True, replace all occurrences. If False,
            require exactly one occurrence.

    Returns:
        Dict[str, Any]: A unified result object:
            {
                "success": bool,
                "data": {
                    "path": str,
                    "replacements_made": int
                } | None,
                "error": str | None,
                "message": str
            }
    """
    try:
        path = _resolve_in_workspace(filepath)

        if not path.exists():
            return fail(f"File not found: {filepath}", message="Failed to edit file.")
        if not path.is_file():
            return fail(f"Not a file: {filepath}", message="Failed to edit file.")
        if old_string == new_string:
            return fail(
                "old_string and new_string are identical.",
                message="Failed to edit file."
            )

        content = path.read_text(encoding="utf-8")
        occurrences = content.count(old_string)

        if occurrences == 0:
            return fail("Target string not found.", message="Failed to edit file.")

        if not replace_all and occurrences > 1:
            return fail(
                "Target string appears multiple times. Use a more specific old_string or set replace_all=True.",
                message="Failed to edit file."
            )

        if replace_all:
            new_content = content.replace(old_string, new_string)
            replaced = occurrences
        else:
            new_content = content.replace(old_string, new_string, 1)
            replaced = 1

        path.write_text(new_content, encoding="utf-8")

        return ok(
            {
                "path": str(path.relative_to(WORKSPACE_ROOT)),
                "replacements_made": replaced,
            },
            message=f"Replaced {replaced} occurrence(s).",
        )
    except Exception as e:
        return fail(str(e), message="Failed to edit file.")


@mcp.tool()
def ls(
    dirpath: str = ".",
    recursive: bool = False,
    max_entries: int = 200,
) -> Dict[str, Any]:
    """
    List files and folders inside the workspace.

    Use this tool to inspect project structure. Recursive mode is useful
    for small directories, but should be avoided on very large trees.
    Results are capped to avoid overwhelming the model context.

    Args:
        dirpath (str): Target directory path inside the workspace.
        recursive (bool): If True, list all descendants recursively.
            If False, list only immediate children.
        max_entries (int): Maximum number of entries to return.

    Returns:
        Dict[str, Any]: A unified result object:
            {
                "success": bool,
                "data": {
                    "path": str,
                    "entries": List[str],
                    "returned_count": int,
                    "total_count": int,
                    "truncated": bool
                } | None,
                "error": str | None,
                "message": str
            }
    """
    try:
        target = _resolve_in_workspace(dirpath)

        if not target.exists():
            return fail(f"Directory not found: {dirpath}", message="Failed to list directory.")
        if not target.is_dir():
            return fail(f"Not a directory: {dirpath}", message="Failed to list directory.")
        if max_entries <= 0:
            return fail("max_entries must be greater than 0.", message="Failed to list directory.")

        all_entries: List[str] = []

        if recursive:
            for p in sorted(target.rglob("*")):
                rel = p.relative_to(target)
                all_entries.append(str(rel) + ("/" if p.is_dir() else ""))
        else:
            for p in sorted(target.iterdir()):
                all_entries.append(p.name + ("/" if p.is_dir() else ""))

        total_count = len(all_entries)
        entries = all_entries[:max_entries]
        truncated = total_count > max_entries

        return ok(
            {
                "path": str(target.relative_to(WORKSPACE_ROOT)),
                "entries": entries,
                "returned_count": len(entries),
                "total_count": total_count,
                "truncated": truncated,
            },
            message="Directory listed successfully.",
        )
    except Exception as e:
        return fail(str(e), message="Failed to list directory.")


@mcp.tool()
def file_glob_search(pattern: str, max_matches: int = 200) -> Dict[str, Any]:
    """
    Find files in the workspace using a glob pattern.

    This tool searches by file path, not by file content. Results are
    capped to avoid overwhelming the model context.

    Args:
        pattern (str): Glob pattern such as '**/*.py', 'src/**/*.ts',
            or '*.md'.
        max_matches (int): Maximum number of matches to return.

    Returns:
        Dict[str, Any]: A unified result object:
            {
                "success": bool,
                "data": {
                    "matches": List[str],
                    "returned_count": int,
                    "total_count": int,
                    "truncated": bool
                } | None,
                "error": str | None,
                "message": str
            }
    """
    excluded = {".git", "__pycache__", "node_modules", "dist", "build", ".venv", "venv"}

    try:
        if max_matches <= 0:
            return fail("max_matches must be greater than 0.", message="Glob search failed.")

        all_matches: List[str] = []

        for path in WORKSPACE_ROOT.glob(pattern):
            if not path.is_file():
                continue
            if any(part in excluded for part in path.parts):
                continue
            all_matches.append(str(path.relative_to(WORKSPACE_ROOT)))

        all_matches.sort()
        total_count = len(all_matches)
        matches = all_matches[:max_matches]
        truncated = total_count > max_matches

        return ok(
            {
                "matches": matches,
                "returned_count": len(matches),
                "total_count": total_count,
                "truncated": truncated,
            },
            message="Glob search completed.",
        )
    except Exception as e:
        return fail(str(e), message="Glob search failed.")


@mcp.tool()
def run_terminal_command(
    command: str,
    cwd: str = ".",
    wait_for_completion: bool = True,
    timeout: int = DEFAULT_CMD_TIMEOUT,
    max_output_chars: int = 20000,
) -> Dict[str, Any]:
    """
    Run a shell command inside the workspace.

    The command runs in a fresh subprocess. Use `cwd` to control the
    working directory for this execution. This tool is intended for
    setup, build, run, and test tasks.

    Args:
        command (str): Shell command to execute.
        cwd (str): Working directory inside the workspace.
        wait_for_completion (bool): If True, wait for the command and
            return stdout/stderr. If False, start it in background and
            return its PID.
        timeout (int): Timeout in seconds for synchronous execution.
        max_output_chars (int): Maximum number of characters returned for
            stdout and stderr separately.

    Returns:
        Dict[str, Any]: A unified result object.

        For synchronous execution:
            {
                "success": bool,
                "data": {
                    "cwd": str,
                    "stdout": str,
                    "stderr": str,
                    "stdout_truncated": bool,
                    "stderr_truncated": bool,
                    "stdout_length": int,
                    "stderr_length": int,
                    "returncode": int
                } | None,
                "error": str | None,
                "message": str
            }

        For background execution:
            {
                "success": bool,
                "data": {
                    "cwd": str,
                    "pid": int
                } | None,
                "error": str | None,
                "message": str
            }
    """
    shell_path = "/bin/bash"

    def _truncate_output(text: str, limit: int) -> tuple[str, bool, int]:
        if text is None:
            return "", False, 0
        original_length = len(text)
        if original_length <= limit:
            return text, False, original_length
        return text[:limit], True, original_length

    try:
        working_dir = _resolve_in_workspace(cwd)
        if not working_dir.exists():
            return fail(
                f"Working directory does not exist: {cwd}",
                message="Failed to run command.",
            )
        if not working_dir.is_dir():
            return fail(
                f"cwd is not a directory: {cwd}",
                message="Failed to run command.",
            )
        if max_output_chars <= 0:
            return fail(
                "max_output_chars must be greater than 0.",
                message="Failed to run command.",
            )

        if wait_for_completion:
            completed = subprocess.run(
                [shell_path, "-c", command],
                cwd=str(working_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

            stdout, stdout_truncated, stdout_length = _truncate_output(
                completed.stdout, max_output_chars
            )
            stderr, stderr_truncated, stderr_length = _truncate_output(
                completed.stderr, max_output_chars
            )

            return ok(
                {
                    "cwd": str(working_dir.relative_to(WORKSPACE_ROOT)),
                    "stdout": stdout,
                    "stderr": stderr,
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                    "stdout_length": stdout_length,
                    "stderr_length": stderr_length,
                    "returncode": completed.returncode,
                },
                message=(
                    "Command completed successfully."
                    if completed.returncode == 0
                    else f"Command finished with return code {completed.returncode}."
                ),
            )

        proc = subprocess.Popen(
            [shell_path, "-c", command],
            cwd=str(working_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

        return ok(
            {
                "cwd": str(working_dir.relative_to(WORKSPACE_ROOT)),
                "pid": proc.pid,
            },
            message=f"Command started in background with PID {proc.pid}.",
        )

    except subprocess.TimeoutExpired:
        return fail("Command timed out.", message="Command execution timed out.")
    except Exception as e:
        return fail(str(e), message="Failed to run command.")


@mcp.tool()
def fetch_url_content(url: str) -> Dict[str, Any]:
    """
    Fetch text or binary content from a web URL.

    This tool is useful for reading documentation, README pages, and
    lightweight web resources. Only HTTP and HTTPS URLs are allowed.

    Args:
        url (str): URL to fetch. Must start with http:// or https://.

    Returns:
        Dict[str, Any]: A unified result object:
            {
                "success": bool,
                "data": {
                    "url": str,
                    "status_code": int,
                    "content": str,
                    "truncated": bool
                } | None,
                "error": str | None,
                "message": str
            }
    """
    if not (url.startswith("http://") or url.startswith("https://")):
        return fail(
            "Only http:// and https:// URLs are allowed.",
            message="Failed to fetch URL."
        )

    try:
        import requests

        resp = requests.get(url, timeout=10, allow_redirects=True)
        content_type = resp.headers.get("content-type", "").lower()
        raw = resp.content[:MAX_FETCH_BYTES]

        if (
            "text" in content_type
            or "json" in content_type
            or "xml" in content_type
            or "javascript" in content_type
        ):
            text = raw.decode(resp.encoding or "utf-8", errors="replace")
            return ok(
                {
                    "url": url,
                    "status_code": resp.status_code,
                    "content": text,
                    "truncated": len(resp.content) > MAX_FETCH_BYTES,
                },
                message="URL fetched successfully.",
            )

        return ok(
            {
                "url": url,
                "status_code": resp.status_code,
                "content": "base64:" + base64.b64encode(raw).decode("ascii"),
                "truncated": len(resp.content) > MAX_FETCH_BYTES,
            },
            message="Binary content fetched successfully.",
        )
    except Exception as e:
        return fail(str(e), message="Failed to fetch URL.")


@mcp.tool()
def clone_repo(
    repo_url: str,
    dest_path: str = "",
    branch: str = "",
    depth: int = 1,
) -> Dict[str, Any]:
    """
    Clone a Git repository into the workspace.

    If `dest_path` is empty, the repository name is used as the target
    folder under the workspace root.

    Args:
        repo_url (str): Repository URL. Supported prefixes are http://,
            https://, git@, and ssh://.
        dest_path (str): Destination path inside the workspace. If empty,
            a folder name is derived from the repository name.
        branch (str): Optional branch or tag to check out.
        depth (int): Clone depth for shallow clone.

    Returns:
        Dict[str, Any]: A unified result object:
            {
                "success": bool,
                "data": {
                    "repo_path": str,
                    "stdout": str
                } | None,
                "error": str | None,
                "message": str
            }
    """
    allowed_prefixes = ("http://", "https://", "git@", "ssh://")
    if not repo_url.startswith(allowed_prefixes):
        return fail("Invalid repository URL.")

    try:
        if dest_path.strip():
            target_dir = _resolve_in_workspace(dest_path)
        else:
            repo_name = Path(repo_url.rstrip("/").split("/")[-1])
            if repo_name.suffix == ".git":
                repo_name = repo_name.with_suffix("")
            target_dir = _resolve_in_workspace(str(repo_name))

        if target_dir.exists():
            return fail(f"Destination already exists: {target_dir}", message="Failed to clone repository.")

        target_dir.parent.mkdir(parents=True, exist_ok=True)

        cmd = ["git", "clone", f"--depth={depth}"]
        if branch:
            cmd.extend(["--branch", branch])
        cmd.extend([repo_url, str(target_dir)])

        result = subprocess.run(
            cmd,
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        if result.returncode != 0:
            return fail(
                result.stderr.strip() or f"git clone failed with code {result.returncode}",
                message="Clone failed.",
            )

        return ok(
            {
                "repo_path": str(target_dir.relative_to(WORKSPACE_ROOT)),
                "stdout": result.stdout,
            },
            message="Repository cloned successfully.",
        )
    except Exception as e:
        return fail(str(e), message="Failed to clone repository.")


@mcp.tool()
def search_in_files(
    query: str,
    glob_pattern: str = "**/*",
    max_results: int = 50,
    context_chars: int = 120,
) -> Dict[str, Any]:
    """
    Search for a text query inside files in the workspace.

    This tool searches file contents, not file paths. It is useful for
    finding function names, configuration keys, imports, TODO markers,
    and entry points. Results are capped to avoid overwhelming the model.

    Args:
        query (str): Text to search for.
        glob_pattern (str): File path glob pattern used to restrict search scope.
        max_results (int): Maximum number of matched snippets to return.
        context_chars (int): Number of surrounding characters to include
            around each match.

    Returns:
        Dict[str, Any]: A unified result object:
            {
                "success": bool,
                "data": {
                    "results": List[{
                        "path": str,
                        "line": int,
                        "match": str
                    }],
                    "returned_count": int,
                    "total_count": int,
                    "truncated": bool
                } | None,
                "error": str | None,
                "message": str
            }
    """
    excluded = {".git", "__pycache__", "node_modules", "dist", "build", ".venv", "venv"}

    try:
        if not query.strip():
            return fail("query must not be empty.", message="Search failed.")
        if max_results <= 0:
            return fail("max_results must be greater than 0.", message="Search failed.")
        if context_chars < 0:
            return fail("context_chars must be non-negative.", message="Search failed.")

        all_results: List[Dict[str, Any]] = []

        for path in WORKSPACE_ROOT.glob(glob_pattern):
            if not path.is_file():
                continue
            if any(part in excluded for part in path.parts):
                continue

            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue

            start = 0
            while True:
                idx = text.find(query, start)
                if idx == -1:
                    break

                line_no = text.count("\n", 0, idx) + 1
                snippet_start = max(0, idx - context_chars)
                snippet_end = min(len(text), idx + len(query) + context_chars)
                snippet = text[snippet_start:snippet_end].replace("\n", "\\n")

                all_results.append(
                    {
                        "path": str(path.relative_to(WORKSPACE_ROOT)),
                        "line": line_no,
                        "match": snippet,
                    }
                )
                start = idx + len(query)

        total_count = len(all_results)
        results = all_results[:max_results]
        truncated = total_count > max_results

        return ok(
            {
                "results": results,
                "returned_count": len(results),
                "total_count": total_count,
                "truncated": truncated,
            },
            message="Content search completed.",
        )
    except Exception as e:
        return fail(str(e), message="Search failed.")


@mcp.tool()
def inspect_project(dirpath: str = ".") -> Dict[str, Any]:
    """
    Inspect a project directory and infer its structure and likely workflow.

    This tool helps the agent quickly understand what kind of project it is,
    which files are important, and how it may be run or tested.

    Args:
        dirpath (str): Target project directory inside the workspace.

    Returns:
        Dict[str, Any]: A unified result object:
            {
                "success": bool,
                "data": {
                    "path": str,
                    "project_types": List[str],
                    "detected_files": List[str],
                    "entry_candidates": List[str],
                    "test_candidates": List[str],
                    "dependency_files": List[str]
                } | None,
                "error": str | None,
                "message": str
            }
    """
    try:
        target = _resolve_in_workspace(dirpath)

        if not target.exists():
            return fail(f"Directory not found: {dirpath}", message="Project inspection failed.")
        if not target.is_dir():
            return fail(f"Not a directory: {dirpath}", message="Project inspection failed.")

        markers = {
            "pyproject.toml": "python",
            "requirements.txt": "python",
            "setup.py": "python",
            "package.json": "node",
            "Cargo.toml": "rust",
            "go.mod": "go",
            "pom.xml": "java",
            "build.gradle": "java",
            "Makefile": "make",
            "CMakeLists.txt": "cpp",
        }

        detected_files: List[str] = []
        dependency_files: List[str] = []
        project_types = set()

        for name, kind in markers.items():
            p = target / name
            if p.exists():
                detected_files.append(name)
                project_types.add(kind)
                if name in {"pyproject.toml", "requirements.txt", "package.json", "Cargo.toml", "go.mod"}:
                    dependency_files.append(name)

        entry_candidates: List[str] = []
        test_candidates: List[str] = []

        candidate_patterns = [
            "main.py", "app.py", "manage.py", "server.py",
            "src/main.py", "src/app.py", "index.js", "src/index.js"
        ]
        for rel in candidate_patterns:
            p = target / rel
            if p.exists():
                entry_candidates.append(rel)

        for pattern in ["tests/**/*", "test/**/*", "pytest.ini", "tox.ini", "package.json"]:
            for p in target.glob(pattern):
                if p.is_file():
                    rel = str(p.relative_to(target))
                    if rel not in test_candidates:
                        test_candidates.append(rel)

        if not project_types:
            project_types.add("unknown")

        return ok(
            {
                "path": str(target.relative_to(WORKSPACE_ROOT)),
                "project_types": sorted(project_types),
                "detected_files": sorted(detected_files),
                "entry_candidates": sorted(entry_candidates),
                "test_candidates": sorted(test_candidates),
                "dependency_files": sorted(dependency_files),
            },
            message="Project inspection completed.",
        )
    except Exception as e:
        return fail(str(e), message="Project inspection failed.")


@mcp.tool()
def get_file_info(filepath: str) -> Dict[str, Any]:
    """
    Get metadata about a file or directory inside the workspace.

    This tool is useful when the agent wants to know whether a path is a
    file or directory, whether it is executable, how large it is, or when
    it was last modified, without reading its full content.

    Args:
        filepath (str): Target path inside the workspace.

    Returns:
        Dict[str, Any]: A unified result object:
            {
                "success": bool,
                "data": {
                    "path": str,
                    "exists": bool,
                    "is_file": bool,
                    "is_dir": bool,
                    "size_bytes": int,
                    "suffix": str,
                    "is_executable": bool,
                    "modified_time": float
                } | None,
                "error": str | None,
                "message": str
            }
    """
    try:
        path = _resolve_in_workspace(filepath)

        if not path.exists():
            return fail(f"Path not found: {filepath}", message="Failed to get file info.")

        st = path.stat()

        return ok(
            {
                "path": str(path.relative_to(WORKSPACE_ROOT)),
                "exists": True,
                "is_file": path.is_file(),
                "is_dir": path.is_dir(),
                "size_bytes": st.st_size,
                "suffix": path.suffix,
                "is_executable": bool(st.st_mode & stat.S_IXUSR),
                "modified_time": st.st_mtime,
            },
            message="File info retrieved successfully.",
        )
    except Exception as e:
        return fail(str(e), message="Failed to get file info.")


@mcp.tool()
def setup_python_env(
    project_dir: str = ".",
    python_executable: str = "python3",
    venv_name: str = ".venv",
    install: bool = True,
    requirements_file: str = "",
    install_args: str = "",
) -> Dict[str, Any]:
    """
    Create a Python virtual environment for a project and install dependencies.

    This tool prefers uv when available. It falls back to the standard
    library venv module and pip when uv is not installed. It supports
    pyproject.toml projects, requirements-based projects, and custom pip
    install arguments such as "-r requirements.txt".

    Args:
        project_dir (str): Target project directory inside the workspace.
        python_executable (str): Python interpreter used when falling back
            to python -m venv.
        venv_name (str): Name of the virtual environment directory.
        install (bool): If True, install project dependencies after creating
            the environment.
        requirements_file (str): Optional requirements file path relative to
            the project directory, such as "requirements.txt" or
            "requirements-dev.txt".
        install_args (str): Optional raw install arguments, such as
            "-r requirements.txt".

    Returns:
        Dict[str, Any]: A unified result object:
            {
                "success": bool,
                "data": {
                    "project_dir": str,
                    "venv_path": str,
                    "manager": str,
                    "install_mode": str | None,
                    "commands": List[str],
                    "stdout": str,
                    "stderr": str
                } | None,
                "error": str | None,
                "message": str
            }
    """
    import shlex

    try:
        target = _resolve_in_workspace(project_dir)

        if not target.exists():
            return fail(f"Directory not found: {project_dir}", message="Environment setup failed.")
        if not target.is_dir():
            return fail(f"Not a directory: {project_dir}", message="Environment setup failed.")

        venv_path = target / venv_name
        pyproject = target / "pyproject.toml"

        # Resolve requirements file if provided
        resolved_requirements = None
        if requirements_file.strip():
            resolved_requirements = _resolve_in_workspace(str(Path(project_dir) / requirements_file))
            if not resolved_requirements.exists():
                return fail(
                    f"Requirements file not found: {requirements_file}",
                    message="Environment setup failed.",
                )
            if not resolved_requirements.is_file():
                return fail(
                    f"Requirements path is not a file: {requirements_file}",
                    message="Environment setup failed.",
                )

        # Default requirements.txt fallback
        default_requirements = target / "requirements.txt"
        if resolved_requirements is None and default_requirements.exists():
            resolved_requirements = default_requirements

        commands: List[str] = []
        stdout_parts: List[str] = []
        stderr_parts: List[str] = []

        uv_bin = shutil.which("uv")
        manager = "uv" if uv_bin else "pip"

        def _run(cmd: List[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
            commands.append(" ".join(cmd))
            result = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            stdout_parts.append(result.stdout)
            stderr_parts.append(result.stderr)
            return result

        if uv_bin:
            result = _run([uv_bin, "venv", str(venv_path)], cwd=target, timeout=120)
            if result.returncode != 0:
                return fail(
                    result.stderr.strip() or f"uv venv failed with code {result.returncode}",
                    message="Environment setup failed.",
                )

            install_mode = None
            if install:
                cmd = None
                python_in_venv = venv_path / "bin" / "python"

                if install_args.strip():
                    cmd = [uv_bin, "pip", "install", "--python", str(python_in_venv)] + shlex.split(install_args)
                    install_mode = f"uv pip install --python {python_in_venv} {install_args}"
                elif resolved_requirements is not None:
                    rel_req = resolved_requirements.relative_to(target)
                    cmd = [
                        uv_bin, "pip", "install",
                        "--python", str(python_in_venv),
                        "-r", str(rel_req)
                    ]
                    install_mode = f"uv pip install --python {python_in_venv} -r {rel_req}"
                elif pyproject.exists():
                    # uv sync 本身是 project-oriented，通常在 project cwd 下执行
                    cmd = [uv_bin, "sync"]
                    install_mode = "uv sync"

                if cmd is not None:
                    result = _run(cmd, cwd=target, timeout=300)
                    if result.returncode != 0:
                        return fail(
                            result.stderr.strip() or f"{install_mode} failed with code {result.returncode}",
                            message="Environment setup failed.",
                        )

        # Fallback: stdlib venv + pip
        result = _run([python_executable, "-m", "venv", str(venv_path)], cwd=target, timeout=120)
        if result.returncode != 0:
            return fail(
                result.stderr.strip() or f"python -m venv failed with code {result.returncode}",
                message="Environment setup failed.",
            )

        pip_path = venv_path / "bin" / "pip"
        install_mode = None
        if install:
            cmd = None

            if install_args.strip():
                cmd = [str(pip_path), "install"] + shlex.split(install_args)
                install_mode = f"pip install {install_args}"
            elif resolved_requirements is not None:
                rel_req = resolved_requirements.relative_to(target)
                cmd = [str(pip_path), "install", "-r", str(rel_req)]
                install_mode = f"pip install -r {rel_req}"
            elif pyproject.exists():
                install_mode = None

            if cmd is not None:
                result = _run(cmd, cwd=target, timeout=300)
                if result.returncode != 0:
                    return fail(
                        result.stderr.strip() or f"{install_mode} failed with code {result.returncode}",
                        message="Environment setup failed.",
                    )

        return ok(
            {
                "project_dir": str(target.relative_to(WORKSPACE_ROOT)),
                "venv_path": str(venv_path.relative_to(WORKSPACE_ROOT)),
                "manager": manager,
                "install_mode": install_mode,
                "commands": commands,
                "stdout": "\n".join(s for s in stdout_parts if s).strip(),
                "stderr": "\n".join(s for s in stderr_parts if s).strip(),
            },
            message="Python environment prepared successfully.",
        )
    except Exception as e:
        return fail(str(e), message="Environment setup failed.")


if __name__ == "__main__":
    print(f"Workspace root: {WORKSPACE_ROOT}")
    mcp.run(transport="http", host="0.0.0.0", port=8000)
