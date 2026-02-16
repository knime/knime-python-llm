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

Use a **discriminator field** (tool type enum) to determine which tool variant is stored:

```
ToolValue (interface/base)
├── WorkflowToolValue (type="WORKFLOW")
│   ├── workflow bytes (filestore)
│   ├── parameter schema
│   └── input/output ports
└── MCPToolValue (type="MCP")
    ├── server URI
    ├── tool name
    └── parameter schema
```

#### Java Side Implementation

**Location:** `org.knime.core` (knime-core repository - not in our workspace)

Create new unified value factory:

```java
// New unified factory
public class ToolValueFactory implements ValueFactory<StructReadAccess, StructWriteAccess> {
    
    @Override
    public DataSpec getSpec() {
        return new StructDataSpec(
            StringDataSpec.INSTANCE,        // 0: tool_type ("WORKFLOW" | "MCP")
            StringDataSpec.INSTANCE,        // 1: name
            StringDataSpec.INSTANCE,        // 2: description
            StringDataSpec.INSTANCE,        // 3: parameter_schema (JSON)
            // WorkflowTool-specific fields
            FileStoreDataSpec.INSTANCE,     // 4: workflow_filestore (null for MCP)
            StringDataSpec.INSTANCE,        // 5: input_ports (JSON, null for MCP)
            StringDataSpec.INSTANCE,        // 6: output_ports (JSON, null for MCP)
            IntDataSpec.INSTANCE,           // 7: message_output_port_index (-1 for MCP)
            // MCPTool-specific fields
            StringDataSpec.INSTANCE,        // 8: server_uri (null for Workflow)
            StringDataSpec.INSTANCE,        // 9: tool_name (null for Workflow)
            StringDataSpec.INSTANCE,        // 10: input_schema (JSON, null if not needed)
            StringDataSpec.INSTANCE         // 11: output_type (null if not needed)
        );
    }
    
    // ReadValue/WriteValue implementations check tool_type field
    // and populate appropriate fields
}

// Shared interface
public interface ToolValue extends DataValue {
    String getToolType();
    String getName();
    String getDescription();
    Map<String, Object> getParameterSchema();
    
    // Execution method with type dispatch
    ToolExecutionResult execute(Map<String, Object> parameters, ...);
}

// Concrete implementations
public class WorkflowToolValue implements ToolValue {
    // Existing WorkflowTool implementation
}

public class MCPToolValue implements ToolValue {
    // New MCP tool implementation
}
```

**Deprecation Strategy:**
- Mark `WorkflowToolValueFactory` as `@Deprecated`
- Add `@since` annotation to `ToolValueFactory` 
- Migration period: both value factories coexist, old workflows auto-convert on load
- Remove deprecated factory in future major version

#### Python Side Implementation

**Location:** `org.knime.python3.arrow.types/src/main/python/knime/types/tool.py`

Create unified Python value factory:

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Union

class ToolType(Enum):
    WORKFLOW = "WORKFLOW"
    MCP = "MCP"

@dataclass
class WorkflowTool:
    """Workflow-based tool (unchanged)"""
    name: str
    description: str
    parameter_schema: dict
    _filestore_keys: bytes = None
    input_ports: list[ToolPort] = None
    output_ports: list[ToolPort] = None
    message_output_port_index: Optional[int] = -1
    
    @property
    def tool_type(self) -> ToolType:
        return ToolType.WORKFLOW

@dataclass
class MCPTool:
    """MCP server tool (unchanged)"""
    name: str
    description: str
    parameter_schema: dict
    server_uri: str
    tool_name: str
    input_schema: Optional[dict] = None
    output_type: Optional[str] = None
    
    @property
    def tool_type(self) -> ToolType:
        return ToolType.MCP

# Type alias for mixed tool types
Tool = Union[WorkflowTool, MCPTool]

class ToolValueFactory(kt.PythonValueFactory):
    """Unified value factory handling both workflow and MCP tools."""
    
    def __init__(self):
        # Register for Union type (duck typing compatibility)
        kt.PythonValueFactory.__init__(self, Tool)
    
    def decode(self, storage):
        import json
        
        if storage is None:
            return None
        
        tool_type = storage.get("0")
        
        if tool_type == "WORKFLOW":
            # Decode WorkflowTool fields (1-7)
            return WorkflowTool(
                name=storage["1"],
                description=storage["2"],
                parameter_schema=json.loads(storage["3"]) if storage["3"] else {},
                _filestore_keys=storage["4"],  # filestore bytes
                input_ports=[ToolPort._from_arrow_dict(p) for p in json.loads(storage["5"])] if storage["5"] else [],
                output_ports=[ToolPort._from_arrow_dict(p) for p in json.loads(storage["6"])] if storage["6"] else [],
                message_output_port_index=storage.get("7", -1),
            )
        elif tool_type == "MCP":
            # Decode MCPTool fields (1-3, 8-11)
            return MCPTool(
                name=storage["1"],
                description=storage["2"],
                parameter_schema=json.loads(storage["3"]) if storage["3"] else {},
                server_uri=storage["8"],
                tool_name=storage["9"],
                input_schema=json.loads(storage["10"]) if storage["10"] else None,
                output_type=storage.get("11"),
            )
        else:
            raise ValueError(f"Unknown tool type: {tool_type}")
    
    def encode(self, value: Tool):
        import json
        
        if value is None:
            return None
        
        # Common fields
        encoded = {
            "0": value.tool_type.value,
            "1": value.name,
            "2": value.description,
            "3": json.dumps(value.parameter_schema),
        }
        
        # Type-specific fields
        if isinstance(value, WorkflowTool):
            encoded.update({
                "4": value._filestore_keys,
                "5": json.dumps([p._to_arrow_dict() for p in value.input_ports]) if value.input_ports else None,
                "6": json.dumps([p._to_arrow_dict() for p in value.output_ports]) if value.output_ports else None,
                "7": value.message_output_port_index,
                "8": None,  # server_uri
                "9": None,  # tool_name
                "10": None, # input_schema
                "11": None, # output_type
            })
        elif isinstance(value, MCPTool):
            encoded.update({
                "4": None,  # workflow_filestore
                "5": None,  # input_ports
                "6": None,  # output_ports
                "7": -1,    # message_output_port_index
                "8": value.server_uri,
                "9": value.tool_name,
                "10": json.dumps(value.input_schema) if value.input_schema else None,
                "11": value.output_type,
            })
        else:
            raise TypeError(f"Unknown tool type: {type(value)}")
        
        return encoded

# Backward compatibility: keep old factories as deprecated aliases
class WorkflowToolValueFactory(ToolValueFactory):
    """@deprecated Use ToolValueFactory instead"""
    def __init__(self):
        super().__init__()
        warnings.warn(
            "WorkflowToolValueFactory is deprecated, use ToolValueFactory instead",
            DeprecationWarning,
            stacklevel=2
        )

class MCPToolValueFactory(ToolValueFactory):
    """@deprecated Use ToolValueFactory instead"""
    def __init__(self):
        super().__init__()
        warnings.warn(
            "MCPToolValueFactory is deprecated, use ToolValueFactory instead",
            DeprecationWarning,
            stacklevel=2
        )
```

#### Plugin Registration

**Location:** `org.knime.python3.arrow.types/plugin.xml`

Update registration to use unified factory:

```xml
<Module modulePath="src/main/python" moduleName="knime.types.tool">
    <!-- New unified factory -->
    <PythonValueFactory
        PythonClassName="ToolValueFactory"
        ValueFactory="org.knime.core.node.agentic.tool.ToolValueFactory"
        ValueTypeName="knime.types.tool.Tool"
        isDefaultPythonRepresentation="true">
    </PythonValueFactory>
    
    <!-- Deprecated: keep for backward compatibility -->
    <PythonValueFactory
        PythonClassName="WorkflowToolValueFactory"
        ValueFactory="org.knime.core.node.agentic.tool.WorkflowToolValueFactory"
        ValueTypeName="knime.types.tool.WorkflowTool"
        isDefaultPythonRepresentation="false">
    </PythonValueFactory>
</Module>
```

### Implementation Plan

#### Phase 1: Java Side (knime-core repository)
**Owner:** KNIME Core team (requires coordination)

1. Create `ToolValue` interface with common methods
2. Refactor `WorkflowToolValue` to implement `ToolValue`
3. Create `MCPToolValue` implementing `ToolValue`
4. Create unified `ToolValueFactory` with discriminator pattern
5. Mark `WorkflowToolValueFactory` as deprecated
6. Update `ToolExecutor` to handle both tool types via polymorphism

#### Phase 2: Python Side (knime-python repository)
**Owner:** Can be implemented by us

1. Add `ToolType` enum to `knime/types/tool.py`
2. Add `tool_type` property to existing `WorkflowTool` and `MCPTool`
3. Create unified `ToolValueFactory` with encode/decode for both types
4. Keep `WorkflowToolValueFactory` and `MCPToolValueFactory` as deprecated aliases
5. Update `plugin.xml` to register unified factory
6. Add migration tests (old format → new format)

#### Phase 3: Agent Updates (knime-python-llm repository)
**Owner:** Can be implemented by us

1. Update `src/agents/_tool.py`:
   - Remove separate tool type checks
   - Use `tool.tool_type` property for dispatch
   - Simplify `to_langchain_tool()` with type-based dispatch

2. Update `src/agents/base.py`:
   - Simplify `_extract_tools_from_table()` (single column, mixed types)
   - Remove separate MCPTool column handling

3. Update nodes:
   - MCP Tool Provider outputs unified tool type
   - Workflow to Tool outputs unified tool type
   - Concatenate tool tables without type conflicts

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

**Risk**: Java side changes require knime-core coordination
- **Mitigation**: Implement Python side first as proof of concept, coordinate with core team

**Risk**: Filestore handling for WorkflowTool in unified factory
- **Mitigation**: Use nullable filestore field (field 4), only populated for workflow tools

**Risk**: Breaking changes for existing Python extensions
- **Mitigation**: Keep deprecated factories working, provide clear migration guide

**Risk**: Arrow schema versioning complexity
- **Mitigation**: Use struct fields with null values, schema version field if needed

### Open Questions

1. Should we introduce an abstract base class for Tool in Python? (Currently using Union type)
2. How to handle tool execution polymorphism in Java? (Interface method vs dispatch)
3. Migration timeline - when to deprecate old factories?
4. Should knime-python-llm initially use unified factory or wait for knime-core?