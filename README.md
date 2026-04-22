# 🛡️ SafeAgent Runtime Controller Demo

SafeAgent Runtime Controller Demo is a system-level prototype for runtime safety control of LLM agents.

It demonstrates how to enforce real-time monitoring, control, and human intervention over an agent’s behavior without modifying the underlying model.

This repository also includes a functional Vibe Shell demo, showcasing how an agent can safely perform real-world tasks such as file operations, command execution, and project setup.


## 🚀 Overview

Modern LLM agents can execute complex tasks but lack runtime safety guarantees.

This project introduces a Runtime Controller Layer that:
- Intercepts agent execution at key points
- Evaluates behavior using a safety core
- Enforces decisions such as Allow / Block / Override / HITL


## 🧠 Architecture

The system has two main components:

### 1. SafeAgent Runtime Controller

Responsible for:
- Receiving execution context (hook + observation)
- Calling SafeAgent Core
- Returning control signals

Capabilities:
- Runtime behavior auditing
- Tool-call control
- Call budget enforcement
- Long-term memory control
- Policy-driven decisions


### 2. Vibe Shell Demo (Application Layer)

A real agent system that can:
- Manage files
- Run shell commands
- Clone repositories
- Set up environments
- Execute development workflows

Features:
- MCP-based tools
- AI Dev Shell style interaction
- Integrated with controller
- HITL (human approval UI)


## 🔁 Execution Flow

User Input  
→ Controller (before_agent)  
→ SafeAgent Decision  
→ (ALLOW / BLOCK / HITL)  
→ Agent Execution  
→ Tool Call (tool_wrapper)  
→ Controller Enforcement  


## 🧑‍⚖️ Human-in-the-Loop (HITL)

When risky behavior is detected:

- APPROVE → execute
- REJECT → block

This enables real-time human supervision.


## 🧰 Tooling

### File System
- `read_file`
- `create_new_file`
- `single_find_and_replace`
- `ls`
- `file_glob_search`

### System
- `run_terminal_command`

### Network
- `fetch_url_content`
- `clone_repo`

### Development
- `setup_python_env` (supports uv / pip / requirements.txt)

### Project Understanding
- `inspect_project`
- `search_in_files`
- `get_file_info`


## ⚙️ Deployment

### Docker

Build:

    docker build -t vibeshell-mcp .

Run:

    docker compose up -d


## 🔍 Design Principles

### Runtime Safety
- No retraining required
- No prompt tricks
- No agent modification
- Control at execution time

### Hook-based Control
- before_agent
- after_model
- tool_wrapper

### Policy-driven
- runtime config
- dev config
- policy engine

### Practical System
- Real command execution
- Real project handling
- Real environment setup


## 📦 Use Cases

- Safe AI agents
- DevOps automation
- Agent sandboxing
- LLM red teaming
- Runtime policy research


## 📌 Future Work

- Multi-agent control
- Fine-grained policies
- Audit logging
- Visual debugging tools
- Learned policy systems
