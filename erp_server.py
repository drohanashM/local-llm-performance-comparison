"""MCP server exposing the purchase-order tools over stdio."""
try:
    from mcp.server.mcpserver import MCPServer          # mcp 2.x
except ImportError:
    from mcp.server.fastmcp import FastMCP as MCPServer  # mcp 1.x (the same class under its old name)

import erp_tools

mcp = MCPServer("erp-audit")

# Equivalent to putting @mcp.tool() above each function in erp_tools.py.
# The function name, docstring and type hints become the tool's name, description and input schema.
for fn in erp_tools.TOOLS:
    mcp.tool()(fn)

if __name__ == "__main__":
    # NOTE: stdout is the protocol channel. Never print() in this file; log to stderr if needed.
    mcp.run(transport="stdio")
