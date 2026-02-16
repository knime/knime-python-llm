# MCP Tool Provider Integration Specification

## Overview

Implement prototype support for Model Context Protocol (MCP) servers in KNIME agents. Creates a new "MCP Tool Provider" node that connects to MCP servers via SSE, retrieves available tools, and outputs them in a table. Adds a new `MCPTool` type alongside `WorkflowTool`, with agent execution adapting to dispatch tool calls to either KNIME workflows or MCP servers. Prototype skips complex type conversions, supporting only string/int/float parameters and results.

## Implementation Steps

### 1. Define MCPTool data structure

**Location:** [org.knime.python3.arrow.types/src/main/python/knime/types/tool.py](../../org.knime.python3.arrow.types/src/main/python/knime/types/tool.py)

- Add `MCPTool` dataclass with: `name`, `description`, `parameter_schema` (JSON schema), `server_uri`, `tool_name`, `input_schema`, `output_type`
- Add `MCPToolValueFactory` class similar to `WorkflowToolValueFactory` for Arrow serialization
- Keep simple types only (string/int/float) in parameter schema

### 2. Create MCP client library

**Location:** `knime-python-llm/src/tools/mcp_client.py`

- Implement SSE transport client for MCP protocol
- Add `MCPClient` class with methods: `connect(server_uri)`, `list_tools()`, `call_tool(tool_name, arguments)`
- Handle JSON-RPC communication over SSE
- Use `httpx` for HTTP/SSE connections
- Basic error handling for connection failures and tool execution errors

### 3. Build MCP Tool Provider node

**Location:** `knime-python-llm/src/tools/mcp_provider.py`

- Create `@knext.node` with name "MCP Tool Provider", icon, category placement
- **Parameters**: `server_url` (StringParameter for SSE endpoint URL)
- **Output**: `@knext.output_table` with single column of type `knext.logical(MCPTool)`
- `configure()`: Return schema with MCPTool column (no validation per user choice)
- `execute()`: Connect to MCP server, call `list_tools()`, convert each MCP tool to `MCPTool` object, build table with one tool per row
- Filter parameter schemas to only include string/int/float types

### 4. Extend agent tool converter

**Location:** [src/agents/_tool.py](../../src/agents/_tool.py)

- Import `MCPTool` from `knime.types.tool`
- Add `LangchainToolConverter.to_langchain_tool_from_mcp()` method
- Convert `MCPTool` to LangChain `StructuredTool`:
  - Map `parameter_schema` to `args_schema`
  - Create `tool_function` that calls `ctx._execute_mcp_tool(mcp_tool, params)`
  - Simple string return value (no data tables for prototype)
- Modify existing `to_langchain_tool()` to detect tool type and dispatch accordingly

### 5. Add MCP execution to agent context

**Location:** [src/agents/base.py](../../src/agents/base.py)

- Add `_execute_mcp_tool(tool: MCPTool, parameters: dict)` method to agent classes
- Instantiate `MCPClient` with `tool.server_uri`
- Call `client.call_tool(tool.tool_name, parameters)`
- Return JSON result as string message
- Handle errors and return error messages

### 6. Update agent input handling

**Location:** [src/agents/base.py](../../src/agents/base.py)

- Modify `_extract_tools_from_table()` to handle mixed `WorkflowTool` and `MCPTool` columns
- Update tool conversion loop in `execute()` to check tool type before conversion
- Pass both `WorkflowTool` and `MCPTool` objects to toolset

## Verification

- Create test workflow: MCP Tool Provider → Agent Prompter → output
- Configure MCP server URL to a test SSE endpoint
- Execute and verify:
  - Tools table contains MCPTool objects from server
  - Agent can list and call MCP tools
  - Tool results appear in conversation
  - Error messages shown for invalid tool calls
- Check for any compile issues
- Run existing agent tests to ensure no regressions

## Design Decisions

- **SSE transport**: Chosen for remote MCP server support (vs stdio for local processes)
- **MCPTool type**: New type alongside WorkflowTool requires agent dispatch logic but keeps clean separation
- **Execute-time validation**: No server connection during configure for faster node setup
- **Simple types only**: String/int/float parameters/results for prototype simplicity