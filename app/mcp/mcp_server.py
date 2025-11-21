# mcp_add_server.py
from mcp.server.fastmcp import FastMCP

# Cria o servidor MCP
mcp = FastMCP("AddServer")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Soma dois números inteiros."""
    return a + b


if __name__ == "__main__":
    # Transport "stdio" = comunicação via stdin/stdout
    # (é o que o MCP client em Python vai usar)
    mcp.run(transport="stdio")
