# MCP Tool Provider Integration Specification

## Overview

Implement prototype support for Model Context Protocol (MCP) servers in KNIME agents. Creates a new "MCP Tool Provider" node that connects to MCP servers via SSE, retrieves available tools, and outputs them in a table. Adds a new `MCPTool` type alongside `WorkflowTool`, with agent execution adapting to dispatch tool calls to either KNIME workflows or MCP servers. Prototype skips complex type conversions, supporting only string/int/float parameters and results.

## Implementation Steps

### 1. Define MCPTool data structure

**Location:** [org.knime.python3.arrow.types/src/main/python/knime/types/tool.py](../../org.knime.python3.arrow.types/src/main/python/knime/types/tool.py)

- Add `ToolType` IntEnum with values: `WORKFLOW = 0`, `MCP = 1` (matching Java enum indices)
- Add `MCPTool` dataclass with: `name`, `description`, `parameter_schema` (JSON schema), `server_uri`, `tool_name`
- Add `tool_type` property returning `ToolType.MCP`
- Add `MCPToolValueFactory` class similar to `WorkflowToolValueFactory` for Arrow serialization
- Keep simple types only (string/int/float) in parameter schema
- Simplified from unified design: Fields removed (`input_schema`, `output_type`) since outputs are always strings for prototype

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

## Unified Value Factory Architecture

### Problem Statement

The initial implementation (steps 1-6 above) uses separate value factories for `WorkflowTool` and `MCPTool`:
- `WorkflowToolValueFactory` (Java + Python) for workflow-based tools
- `MCPToolValueFactory` (Python only) for MCP server tools

This creates several issues:
1. **Separate columns required**: Cannot mix tool types in same column
2. **Agent complexity**: Requires multiple tool columns or complex union handling
3. **Plugin registration**: Need to register both value factories in `plugin.xml`
4. **Type system fragmentation**: Different logical types for conceptually similar data

### Goal: Unified Tool Value Factory

Create a single value factory that can represent both workflow and MCP tools, allowing them to be mixed freely in the same table column. This enables:
- Seamless mixing of tool sources (e.g., output of Workflow to Tool + MCP Tool Provider → concatenate → Agent)
- Simplified agent node configuration (single tool column parameter)
- Future extensibility for additional tool types (e.g., HTTP API tools, Python function tools)

### Architecture Design

#### Discriminator Pattern

Use a **byte-indexed enum** as discriminator to determine which tool variant is stored, enabling seamless mixing of workflow and MCP tools in a single table column.

#### Java Side Implementation

**Location:** `org.knime.core` (knime-core repository)

- Define `ToolType` enum with byte indices: `WORKFLOW = 0`, `MCP = 1`
- Create unified `ToolValueFactory` with 9-field struct (see Arrow Schema below)
- Implement `ToolValue` interface for polymorphic tool execution
- Deprecate `WorkflowToolValueFactory` with migration support

#### Python Side Implementation

**Location:** `org.knime.python3.arrow.types/src/main/python/knime/types/tool.py`

- Define `ToolType(IntEnum)` matching Java indices
- Add `tool_type` property to `WorkflowTool` and `MCPTool` 
- Create unified `ToolValueFactory` with encode/decode dispatch
- Keep deprecated factories as aliases for backward compatibility

### Implementation Plan

#### Arrow Schema Structure

The unified value factory uses a 9-field struct representation:

```
Field 0: tool_type (BYTE)        - Enum index: 0=WORKFLOW, 1=MCP
Field 1: name (STRING)           - Display name of the tool
Field 2: description (STRING)    - Tool description for LLM
Field 3: parameter_schema (STRING) - JSON configuration schema (both types)
Field 4: input_spec (STRING)     - JSON of input ports (Workflow) or null (MCP)
Field 5: output_spec (STRING)    - JSON of output ports (Workflow) or null (MCP)
Field 6: message_output_port_index (INT) - Port index (Workflow) or -1 (MCP)
Field 7: workflow_filestore (FILESTORE) - Workflow bytes (Workflow) or null (MCP)
Field 8: server_uri (STRING)     - MCP server URI (MCP) or null (Workflow)
```

**Field Consolidation Rationale:**

Original design had 12 fields with duplication:
- `input_ports` (Workflow) + `input_schema` (MCP) → consolidated to `input_spec`
- `output_ports` (Workflow) + `output_type` (MCP) → consolidated to `output_spec`

Benefits of consolidation:
- **Reduced redundancy**: `input_schema` was redundant with `parameter_schema` for MCP tools
- **Clearer semantics**: `input_spec`/`output_spec` clearly indicate data port specifications
- **Smaller schema**: 9 fields vs 12 fields (25% reduction)
- **Type inference**: MCP output type always "string" for prototype, can be inferred

Field usage by tool type:
- **Common** (0-3): tool_type, name, description, parameter_schema
- **Workflow-only** (4-7): input_spec, output_spec, message_output_port_index, workflow_filestore
- **MCP-only** (8): server_uri

#### Implementation Phases

**Phase 1: Java Side (knime-core)** - Requires KNIME Core team coordination
- Create `ToolType` enum and `ToolValue` interface
- Implement unified `ToolValueFactory`
- Deprecate `WorkflowToolValueFactory`

**Phase 2: Python Side (knime-python)** - Can implement independently
- Add `ToolType` enum to `knime/types/tool.py`
- Create unified `ToolValueFactory` with encode/decode
- Update `plugin.xml` registration
- Maintain deprecated factories for compatibility

**Phase 3: Agent Updates (knime-python-llm)** - Depends on Phase 2
- Use `tool.tool_type` property for dispatch
- Simplify tool extraction (single column, mixed types)
- Update MCP Tool Provider to output unified type

### Benefits

1. **Seamless Tool Mixing**: Single column can contain workflow + MCP tools
2. **Simplified Agent Nodes**: One tool column parameter instead of multiple
3. **Future Extensibility**: Easy to add HTTP API tools, Python function tools, etc.
4. **Type System Consistency**: One logical type for all tools
5. **Backward Compatibility**: Old workflows continue working via deprecated factories
6. **Reduced Code Complexity**: Single conversion path in agents

### Migration Path

1. **Immediate (v1.0)**: Current implementation with separate factories works
2. **Next Release (v1.1)**: Add unified factory alongside deprecated ones
3. **Migration Period (v1.1-v2.0)**: Both systems coexist, users encouraged to migrate
4. **Future (v2.0)**: Remove deprecated factories, unified only

### Risks & Mitigation

- **Java coordination**: Implement Python proof-of-concept first, then coordinate with core team
- **Filestore handling**: Use nullable field (7) for workflow-specific data
- **Breaking changes**: Maintain deprecated factories during migration period
- **Schema versioning**: Use byte enum + nullable fields for efficient type discrimination

### Next Steps

1. Implement Phase 2 (Python unified factory) as proof of concept
2. Coordinate with KNIME Core team for Phase 1 (Java implementation)
3. After Phase 1-2 complete, implement Phase 3 (Agent updates)