"""
SSH MCP Server for AutoDL remote control.
Lets Claude Code execute commands on a remote Linux machine.
Usage: python mcp_ssh_server.py
Config via env: SSH_HOST, SSH_PORT, SSH_USER, SSH_PASSWORD
"""

import os
import json
import sys
import io


def get_client():
    """Create paramiko SSH client from env vars."""
    import paramiko

    host = os.environ.get("SSH_HOST", "")
    port = int(os.environ.get("SSH_PORT", "22"))
    user = os.environ.get("SSH_USER", "root")
    password = os.environ.get("SSH_PASSWORD", "")
    key_file = os.environ.get("SSH_KEY_FILE", "")

    if not host:
        raise ValueError("SSH_HOST not set")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    if key_file:
        client.connect(host, port=port, username=user, key_filename=key_file, timeout=10)
    elif password:
        client.connect(host, port=port, username=user, password=password, timeout=10)
    else:
        # Try default key
        client.connect(host, port=port, username=user, timeout=10)

    return client


def run_ssh(command, timeout=300):
    """Execute command on remote host, return output."""
    try:
        client = get_client()
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout)

        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")

        # Get exit code
        exit_code = stdout.channel.recv_exit_status()

        result = out
        if err:
            if len(result) > 2000:
                result = result[:2000] + "\n...[truncated]"
            result += f"\n[STDERR]\n{err[:2000]}"
        client.close()
        return f"[exit={exit_code}]\n{result[:8000]}"
    except Exception as e:
        return f"ERROR: {e}"


def run_scp_put(local_path, remote_path):
    """Upload a local file to remote machine."""
    import paramiko

    try:
        client = get_client()
        sftp = client.open_sftp()
        sftp.put(local_path, remote_path)
        sftp.close()
        client.close()
        return f"OK: Uploaded {local_path} -> {remote_path}"
    except Exception as e:
        return f"ERROR: {e}"


def run_scp_get(remote_path, local_path):
    """Download a remote file to local machine."""
    import paramiko

    try:
        client = get_client()
        sftp = client.open_sftp()
        sftp.get(remote_path, local_path)
        sftp.close()
        client.close()
        return f"OK: Downloaded {remote_path} -> {local_path}"
    except Exception as e:
        return f"ERROR: {e}"


def main():
    captive_stdout = io.StringIO()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = request.get("method", "")
        req_id = request.get("id")

        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "ssh-remote-executor",
                        "version": "1.0.0",
                    },
                },
            }
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        elif method == "tools/list":
            host_info = os.environ.get("SSH_HOST", "not set")
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "ssh_exec",
                            "description": f"Execute a shell command on remote AutoDL machine ({host_info}). Run ANY shell command and get stdout+stderr+exit code back. Use for: checking GPU (nvidia-smi), listing files (ls), viewing logs (tail/cat), running Python scripts, pip install, monitoring training, etc.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "command": {
                                        "type": "string",
                                        "description": "Shell command to run on the remote Linux machine. Can use && to chain commands.",
                                    },
                                },
                                "required": ["command"],
                            },
                        },
                        {
                            "name": "scp_upload",
                            "description": f"Upload a local file to the remote AutoDL machine ({host_info}).",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "local_path": {
                                        "type": "string",
                                        "description": "Absolute path to the file on local Windows machine",
                                    },
                                    "remote_path": {
                                        "type": "string",
                                        "description": "Destination path on the remote machine, e.g. /root/autodl-tmp/myfile.tar.gz",
                                    },
                                },
                                "required": ["local_path", "remote_path"],
                            },
                        },
                        {
                            "name": "scp_download",
                            "description": f"Download a file from the remote AutoDL machine ({host_info}) to local.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "remote_path": {
                                        "type": "string",
                                        "description": "Path to the file on the remote machine",
                                    },
                                    "local_path": {
                                        "type": "string",
                                        "description": "Destination path on local Windows machine",
                                    },
                                },
                                "required": ["remote_path", "local_path"],
                            },
                        },
                    ]
                },
            }
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        elif method == "tools/call":
            tool_name = request.get("params", {}).get("name", "")
            args = request.get("params", {}).get("arguments", {})

            if tool_name == "ssh_exec":
                result_text = run_ssh(args.get("command", "echo hello"))
            elif tool_name == "scp_upload":
                result_text = run_scp_put(
                    args.get("local_path", ""),
                    args.get("remote_path", ""),
                )
            elif tool_name == "scp_download":
                result_text = run_scp_get(
                    args.get("remote_path", ""),
                    args.get("local_path", ""),
                )
            else:
                result_text = f"Unknown tool: {tool_name}"

            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": result_text}],
                },
            }
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        elif method == "notifications/initialized":
            pass


if __name__ == "__main__":
    main()
