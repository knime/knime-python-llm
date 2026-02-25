#!/usr/bin/env python3
"""
Simple MCP test server with example tools.

Implements JSON-RPC 2.0 over HTTP for testing the MCP Tool Provider node.

Usage:
    pixi run python test_mcp_server.py                    # No auth
    pixi run python test_mcp_server.py --auth              # Basic auth (user/testpass)

The server will start at http://localhost:8080/mcp
"""

import argparse
import base64
import json
import logging
import secrets
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MCP Test Server")

# --- Authentication -----------------------------------------------------------

# Credentials for basic auth (only enforced when --auth is passed)
AUTH_USERNAME = "user"
AUTH_PASSWORD = "testpass"

# Global flag set from CLI args
_require_auth = False

security = HTTPBasic(auto_error=False)


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Dependency that enforces HTTP Basic Auth when ``_require_auth`` is set."""
    if not _require_auth:
        return  # auth disabled – allow all requests

    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing credentials", headers={"WWW-Authenticate": "Basic"})

    correct_username = secrets.compare_digest(credentials.username, AUTH_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, AUTH_PASSWORD)

    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="Invalid credentials", headers={"WWW-Authenticate": "Basic"})


# Example tools
TOOLS = [
    {
        "name": "add",
        "description": "Add two numbers together",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "greet",
        "description": "Generate a greeting message",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name to greet"},
                "formal": {
                    "type": "boolean",
                    "description": "Use formal greeting",
                    "default": False,
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "reverse_string",
        "description": "Reverse a string",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to reverse"}
            },
            "required": ["text"],
        },
    },
]


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Execute a tool and return the result."""
    if tool_name == "add":
        return {"result": arguments["a"] + arguments["b"]}
    elif tool_name == "greet":
        name = arguments["name"]
        formal = arguments.get("formal", False)
        if formal:
            greeting = f"Good day, {name}. How do you do?"
        else:
            greeting = f"Hi {name}!"
        return {"greeting": greeting}
    elif tool_name == "reverse_string":
        text = arguments["text"]
        return {"reversed": text[::-1]}
    else:
        raise ValueError(f"Unknown tool: {tool_name}")


@app.post("/mcp")
async def handle_jsonrpc(request: Request, _=Depends(verify_credentials)):
    """Handle JSON-RPC 2.0 requests."""
    try:
        data = await request.json()
        logger.info(f"Received request: {data}")

        # Validate JSON-RPC 2.0 format
        if data.get("jsonrpc") != "2.0":
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": data.get("id"),
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request: jsonrpc field must be '2.0'",
                    },
                },
                status_code=400,
            )

        method = data.get("method")
        request_id = data.get("id")
        params = data.get("params", {})

        # Handle methods
        if method == "tools/list":
            logger.info("Listing tools")
            result = {"tools": TOOLS}
            return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": result})

        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            logger.info(f"Calling tool '{tool_name}' with arguments: {arguments}")

            try:
                result = execute_tool(tool_name, arguments)
                logger.info(f"Tool result: {result}")
                return JSONResponse(
                    {"jsonrpc": "2.0", "id": request_id, "result": result}
                )
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32603,
                            "message": f"Tool execution failed: {str(e)}",
                        },
                    },
                    status_code=500,
                )

        else:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                },
                status_code=404,
            )

    except json.JSONDecodeError:
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error: Invalid JSON"},
            },
            status_code=400,
        )
    except Exception as e:
        logger.error(f"Server error: {e}")
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
            },
            status_code=500,
        )


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "MCP Test Server",
        "endpoint": "/mcp",
        "tools": len(TOOLS),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Test Server")
    parser.add_argument(
        "--auth",
        action="store_true",
        help=f"Enable HTTP Basic Auth (username={AUTH_USERNAME}, password={AUTH_PASSWORD})",
    )
    args = parser.parse_args()

    _require_auth = args.auth

    print("\n" + "=" * 60)
    print("Starting MCP Test Server")
    print("=" * 60)
    print(f"Server URL: http://localhost:8080/mcp")
    if _require_auth:
        print(f"Auth:       Basic (user={AUTH_USERNAME}, pass={AUTH_PASSWORD})")
    else:
        print("Auth:       NONE")
    print(f"Available tools: {', '.join(t['name'] for t in TOOLS)}")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8080)
