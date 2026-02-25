# MCP Tool Provider Integration Specification

## Overview

Implement prototype support for Model Context Protocol (MCP) servers in KNIME agents. Creates a new "MCP Tool Provider" node that connects to MCP servers via SSE, retrieves available tools, and outputs them in a table. Adds a new `MCPTool` type alongside `WorkflowTool`, with agent execution adapting to dispatch tool calls to either KNIME workflows or MCP servers. Prototype skips complex type conversions, supporting only string/int/float parameters and results.

## Implementation Steps

### 1. Define MCPTool data structure

**Location:** [org.knime.python3.arrow.types/src/main/python/knime/types/tool.py](../../org.knime.python3.arrow.types/src/main/python/knime/types/tool.py)

- Add `ToolType` IntEnum with values: `WORKFLOW = 0`, `MCP = 1` (matching Java enum indices)
- Add `MCPTool` dataclass with: `name`, `description`, `parameter_schema` (JSON schema), `server_uri`, `tool_name`
- Add `tool_type` property returning `ToolType.MCP`
- Add unified `ToolValueFactory` class similar to `WorkflowToolValueFactory` for Arrow serialization
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
- `ToolValueFactory` for unified tools across MCP server and workflow  tools

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

1. ✅ **Completed**: Phase 2 (Python unified factory) 
2. ✅ **Completed**: Phase 1 (Java ToolType enum and ToolValueFactory)
3. ✅ **Completed**: Unified ToolCell implementation
4. ⏳ **In Progress**: MCP tool execution logic
5. **Pending**: Agent updates (Phase 3)

## Cell and Value Factory Architecture

### KNIME Type System Constraints

KNIME tables have a fundamental constraint: each column has exactly ONE `DataType`, and all cells in that column must be instances of that specific cell class. This means you cannot mix `WorkflowToolCell` and `MCPToolCell` in the same column—they're different classes with different types.

**Solution**: Create a unified `ToolCell` that uses an internal discriminator to represent both tool types within a single cell class.

### Cell Inheritance Hierarchy

```
DataCell (abstract)
  └── FileStoreCell (abstract)
        └── ToolCell (concrete, implements WorkflowToolValue)
```

**Design Rationale:**
- **FileStoreCell**: Enables persistent storage of workflow bytes via KNIME's file store mechanism (needed for workflow tools)
- **WorkflowToolValue**: Interface that defines tool execution and metadata access methods
- **ToolCell**: Single concrete implementation handling both workflow and MCP tools internally

### Discriminator Pattern

ToolCell uses a `ToolType` enum field to determine behavior at runtime:

```
ToolCell
├── m_toolType: WORKFLOW | MCP
├── Common fields: name, description, parameter_schema
├── Workflow-only fields: inputs, outputs, workflow_bytes, file_store_path
└── MCP-only fields: server_uri, tool_name
```

Type-specific fields are null/empty when not applicable. The discriminator drives all type-dependent behavior.

### Type-Aware Behavior

ToolCell methods check the `m_toolType` field and branch accordingly:

**Persistence Lifecycle:**
- `postConstruct()`: Only workflow tools load their serialized workflow bytes from the file store; MCP tools are stateless
- `flushToFileStore()`: Only workflow tools save their workflow data; MCP tools have nothing to persist

**Execution:**
- `execute()`: Dispatches to `executeWorkflowTool()` or `executeMCPTool()` based on type
- Workflow execution loads the workflow bytes and runs via `CombinedExecutor`
- MCP execution will make HTTP JSON-RPC calls to the server URI (currently a stub)

**Serialization:**
- First byte written/read is the type discriminator (0=WORKFLOW, 1=MCP)
- Common fields (name, description, parameter_schema) written for both types
- Type-specific fields only written for the appropriate type
- Deserialization reads discriminator first, then constructs the appropriate cell variant

### Value Factory Integration

The `ToolValueFactory` bridges between Python's Arrow representation and Java's `ToolCell`:

**Arrow Schema (9 fields):**
```
[0] tool_type (BYTE)        - Discriminator
[1-3] name, description, parameter_schema (STRING) - Common
[4-7] input_spec, output_spec, msg_port, workflow_filestore - Workflow-only
[8] server_uri (STRING)     - MCP-only
```

**Arrow → Java (Python to KNIME):**
1. Read discriminator byte from field 0
2. Read common fields (1-3)
3. If WORKFLOW: read fields 4-7, construct workflow ToolCell
4. If MCP: read field 8, construct MCP ToolCell

**Java → Arrow (KNIME to Python):**
1. Detect type from ToolCell instance
2. Write discriminator to field 0
3. Write common fields (1-3)
4. If WORKFLOW: write fields 4-7, mark field 8 as missing
5. If MCP: write field 8, mark fields 4-7 as missing

### Complete Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Python (knime-python-llm)                                    │
│   MCPTool / WorkflowTool objects                             │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ ToolValueFactory.encode() (Python)                           │
│   → 9-field Arrow struct with discriminator                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼ Arrow IPC (memory-mapped file)
                            │
┌─────────────────────────────────────────────────────────────┐
│ ToolValueFactory.createCell() (Java)                         │
│   → ToolCell with correct m_toolType                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ ToolCell.execute()                                           │
│   ├─ WORKFLOW → executeWorkflowTool()                        │
│   └─ MCP → executeMCPTool()                                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
                    WorkflowToolResult
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ ToolValueFactory.setTableData() (Java)                       │
│   → 9-field Arrow struct                                     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼ Arrow IPC
                            │
┌─────────────────────────────────────────────────────────────┐
│ ToolValueFactory.decode() (Python)                           │
│   → MCPTool / WorkflowTool objects                           │
└─────────────────────────────────────────────────────────────┘
```

### Key Benefits

1. **Single Column Type**: Use `ToolCell.TYPE` for entire column, enabling mixed tool sources
2. **Seamless Concatenation**: Workflow Tool Provider + MCP Tool Provider outputs can be concatenated
3. **Type Safety**: Discriminator ensures correct execution path
4. **Efficient Storage**: Only stores fields relevant to each type
5. **KNIME Compliance**: Follows standard DataCell/FileStoreCell patterns

### Implementation Status

✅ ToolType enum (Java + Python)  
✅ ToolCell with discriminator and type-specific fields  
✅ ToolCell constructors for both types  
✅ Type-aware lifecycle methods (postConstruct/flushToFileStore)  
✅ execute() dispatch to type-specific methods  
✅ Binary serialization with discriminator byte  
✅ ToolValueFactory Arrow conversion with type branching  
⏳ executeMCPTool() implementation (HTTP client stub)  
⏳ MCP Tool Caller node (Java)  
⏳ Audit workflow node creation dispatch

## MCP Tool Caller Node Design

### Purpose & Use Cases

The **MCP Tool Caller** node enables users to execute tools from MCP servers, either manually or as part of an agent audit workflow.

**Primary use cases:**
1. **Manual tool testing**: Users can test MCP tools directly without running a full agent
2. **Workflow integration**: Incorporate MCP tool calls into regular KNIME workflows
3. **Agent audit trail**: Automatically created by agents to record MCP tool executions for reproducibility

**Parallel to Workflow Segment Executor**: Just as workflow tools execute via Workflow Segment Executor nodes, MCP tools execute via MCP Tool Caller nodes. This symmetry makes audit workflows intuitive.

### Node Configuration

**Input Ports:**
- Port 0: Tool table (from MCP Tool Provider or Workflow to Tool nodes)
- Port 1 (optional): Parameters table for dynamic parameter values

**Output Ports:**
- Port 0: Result table containing tool execution output

**Configuration Dialog:**

The dialog displays human-readable information extracted from the selected tool:

```
Tool Selection:
  └─ Row ID: [dropdown of tool rows]

Selected Tool Details (read-only):
  ├─ Type: MCP Tool
  ├─ Name: search_web
  ├─ Description: Searches the web for information
  ├─ Server: http://localhost:8080/mcp
  └─ Parameter Schema:
      {
        "query": {"type": "string", "description": "Search query"},
        "max_results": {"type": "integer", "default": 10}
      }

Parameters (JSON):
  {
    "query": "KNIME tutorials",
    "max_results": 5
  }
```

**Key features:**
- Tool metadata displayed clearly for user understanding
- Parameter schema shown to guide manual parameter entry
- JSON editor for parameter input with validation against schema
- Can use parameter port or manual JSON configuration

### Implementation Language: Java

**Rationale:**
- **Consistency**: Matches Workflow Segment Executor architecture
- **Audit workflow creation**: Easier integration with agent code that builds audit workflows
- **HTTP client**: Native `java.net.http.HttpClient` (Java 11+) handles MCP server communication
- **Error handling**: Better integration with KNIME node lifecycle
- **No Python bridge**: Direct execution without Py4J overhead

**Location:** `org.knime.core/src/eclipse/org/knime/core/node/agentic/tool/` (alongside WorkflowToolCell)

### Execution Flow

1. **Configure phase**: Validate tool table schema, check parameter format
2. **Execute phase**:
   - Extract selected ToolCell from input table
   - Verify it's an MCP tool (check `ToolType.MCP`)
   - Parse parameters (from JSON config or parameter port)
   - Construct JSON-RPC 2.0 request
   - HTTP POST to server URI from ToolCell
   - Parse JSON-RPC response
   - Convert result to KNIME table
   - Handle errors and return as node warnings/errors

## Type Dispatch Points (Critical for Maintainability)

When adding new tool types in the future, these locations MUST be updated. Use exhaustive switch statements to ensure compiler/interpreter catches missing cases.

### 1. Tool Execution Dispatch

**Location:** `ToolCell.execute()` methods

**Status:** Already implemented with discriminator

**Pattern:** Use switch statement rather than if-else to ensure exhaustiveness

**Purpose:** Route execution to appropriate handler (workflow engine vs HTTP client)

### 2. Audit Workflow Node Creation ⚠️ **MOST CRITICAL**

**Location:** Agent execution code (likely `ToolExecutor.java` or agent Python code)

**Purpose:** When agent creates audit workflow after tool execution, it must create the appropriate node type

**Required logic:**
- Inspect `ToolCell.getToolType()`
- For `WORKFLOW`: Create Workflow Segment Executor node, configure with workflow segment
- For `MCP`: Create MCP Tool Caller node, configure with server URI, tool name, parameters
- Set node positions, connections, and metadata for audit trail

**Why critical:** This is the main integration point for the new MCP Tool Caller node. Missing this means MCP tools won't appear in audit workflows.

**Search hints:** Look for code that instantiates `WorkflowSegmentExecutor` or creates nodes in audit workflows. Agent execution likely has a method that builds the trace workflow after each tool call.

### 3. Serialization/Deserialization

**Location:** `ToolCell.WorkflowToolCellSerializer`

**Status:** Already implemented with type byte discriminator

**Purpose:** Binary serialization for workflow persistence

**Pattern:** Write type byte first, then branch to type-specific fields

### 4. Arrow Schema Conversion

**Location:** `ToolValueFactory` (Java and Python)

**Status:** Already implemented

**Purpose:** Convert between KNIME tables (Arrow format) and Java ToolCell objects

**Pattern:** Read field 0 discriminator, dispatch to appropriate constructor

### 5. LangChain Tool Conversion

**Location:** `knime-python-llm/src/agents/_tool.py`

**Status:** Already implemented

**Purpose:** Convert KNIME ToolCell to LangChain StructuredTool for agent use

**Pattern:** Python match statement on `tool.tool_type`

### 6. Tool Execution Context Setup

**Location:** Agent initialization code

**Purpose:** Set up execution environment before tool runs (file stores for workflows, HTTP clients for MCP)

**Pattern:** Branch based on tool type to prepare appropriate resources

### Non-Critical Dispatch Points

These improve clarity and performance but aren't strictly required:

- **Debug display strings**: Format tool names for logging/UI
- **Resource cleanup**: Close connections or file stores after execution
- **Icon/visualization**: Show different icons for different tool types

### Ensuring Exhaustiveness

**Java approach:**
Use switch expressions (Java 14+) without default clause. Compiler enforces all enum values handled:

```java
// @ToolTypeDispatch - Update when adding new ToolType enum values
var result = switch (tool.getToolType()) {
    case WORKFLOW -> handleWorkflow(tool);
    case MCP -> handleMCP(tool);
    // No default - compiler error if case missing
};
```

**Python approach:**
Use match statements (Python 3.10+) with explicit error case:

```python
# @ToolTypeDispatch - Update when adding new ToolType values
match tool.tool_type:
    case ToolType.WORKFLOW:
        return handle_workflow(tool)
    case ToolType.MCP:
        return handle_mcp(tool)
    case _:
        raise ValueError(f"Unhandled tool type: {tool.tool_type}")
```

**Code review marker:**
Add `@ToolTypeDispatch` comments at all critical dispatch points. When adding new tool types, search for this marker to find all locations requiring updates.

## Next Steps

1. **Implement MCP Tool Caller node** (Java)
   - Node model with tool table input and configuration
   - HTTP client for JSON-RPC communication
   - Error handling and result conversion

2. **Find and update audit workflow creation code**
   - Search for Workflow Segment Executor instantiation
   - Add type dispatch to create MCP Tool Caller nodes
   - Ensure audit workflows show both tool types correctly

3. **Convert dispatch points to exhaustive switches**
   - Replace if-else with switch statements in Java
   - Use match statements in Python
   - Add `@ToolTypeDispatch` markers

4. **Testing**
   - Manual: Use MCP Tool Caller node with tool table
   - Agent: Verify MCP tools appear in audit workflow
   - Mixed: Test workflows with both tool types