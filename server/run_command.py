import subprocess
import os
from typing import Any, Dict
from fastmcp import FastMCP

mcp = FastMCP(name="CommandExecutorMCP")


@mcp.tool()
def run_command(cmd: str, privileged: bool = False, timeout: int = 10) -> Dict[str, Any]:
    """
    Safely execute a system command with privilege control.

    Args:
        cmd (str): The command to execute, e.g., "ls -la".
        privileged (bool): Whether to run with elevated privileges.
        timeout (int): Timeout in seconds (default: 10).

    Returns:
        dict: A JSON-compatible dictionary containing:
            - success (bool): True if the command executed successfully.
            - return_code (int, optional): The process exit code.
            - stdout (str, optional): Captured standard output.
            - stderr (str, optional): Captured standard error.
            - error (str, optional): Error message if execution failed or was blocked.
            - privileged (bool): Whether privileged execution was requested.
            - cmd (str): The original command string.
    """
    if not isinstance(cmd, str):
        return {"success": False, "error": "Invalid command type", "cmd": str(cmd)}

    cmd = cmd.strip()
    is_sudo_in_cmd = cmd.startswith("sudo")

    # Reject unauthorized sudo
    if is_sudo_in_cmd and not privileged:
        return {
            "success": False,
            "error": "Command rejected: 'sudo' not allowed without privileged=True",
            "cmd": cmd,
            "privileged": False
        }

    # Use /bin/bash -c for proper shell support
    if privileged:
        if is_sudo_in_cmd:
            cmd = cmd[len("sudo"):].strip()
        exec_cmd = ["sudo", "/bin/bash", "-c", cmd]
    else:
        exec_cmd = ["/bin/bash", "-c", cmd]

    try:
        result = subprocess.run(
            exec_cmd,
            cwd=os.path.expanduser("~"),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )

        return {
            "success": result.returncode == 0,
            "return_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "privileged": privileged,
            "cmd": cmd
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Command timed out",
            "cmd": cmd,
            "privileged": privileged
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "cmd": cmd,
            "privileged": privileged
        }


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
