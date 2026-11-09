import asyncio, json, sys, time
from pathlib import Path

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from erp_tools import TOOLS, TOOL_MAP

MODEL = "llama3.2:3b"
OPTS = {"temperature": 0, "num_ctx": 4096}
SYSTEM = ("You answer questions about purchase orders (POs) using the provided tools. "
          "Always call a tool to get facts; never guess. Use PO IDs and user IDs exactly as written. "
          "If a tool returns an error, tell the user plainly. Answer briefly.")

def compact(obj):
    return json.dumps(obj, separators=(",", ":"))

# ---------- Part A: the agent loop with plain Python functions ----------
def run_agent(question, model=MODEL, max_steps=5):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    trace, t0 = [], time.time()
    for _ in range(max_steps):
        resp = ollama.chat(model=model, messages=messages, tools=TOOLS, options=OPTS)
        messages.append(resp.message)
        if not resp.message.tool_calls:
            return {"answer": resp.message.content, "trace": trace, "secs": round(time.time() - t0, 1)}
        for call in resp.message.tool_calls:
            name, args = call.function.name, dict(call.function.arguments)
            trace.append((name, args))
            try:
                result = TOOL_MAP[name](**args)
            except Exception as e:  # unknown tool name, missing or wrong arguments, ...
                result = {"error": f"{type(e).__name__}: {e}"}
            messages.append({"role": "tool", "tool_name": name, "content": compact(result)})
    return {"answer": None, "trace": trace, "secs": round(time.time() - t0, 1), "note": "stopped after max_steps"}

# ---------- Part B: the same loop, but tools live in an MCP server ----------
SERVER = StdioServerParameters(command=sys.executable, args=[str(Path("erp_server.py").resolve())])

# Jupyter replaces sys.stderr with an object that has no file descriptor, and the MCP client
# tries to hand stderr to the server subprocess. Sending the server's log output to a real
# file avoids that (and also keeps the notebook output clean).
Path("results").mkdir(exist_ok=True)
ERRLOG = open("results/mcp_server.log", "a", encoding="utf-8")

def connect():
    """Start the MCP server as a subprocess and return the connection to use with `async with`."""
    return stdio_client(SERVER, errlog=ERRLOG)

# mcp 2.x uses snake_case field names, mcp 1.x camelCase; these helpers accept both.
def tool_schema(t):
    return t.input_schema if hasattr(t, "input_schema") else t.inputSchema

def result_is_error(r):
    return r.is_error if hasattr(r, "is_error") else r.isError

def result_text(r):
    text = "".join(c.text for c in r.content if getattr(c, "type", "") == "text")
    try:
        return compact(json.loads(text))   # servers pretty-print JSON; compact it to save tokens
    except ValueError:
        return text

def to_ollama_tools(mcp_tools):
    """Translate MCP tool descriptions into the format Ollama's chat API expects."""
    return [{"type": "function",
             "function": {"name": t.name, "description": t.description or "", "parameters": tool_schema(t)}}
            for t in mcp_tools]

async def run_agent_mcp(question, session, ollama_tools, model=MODEL, max_steps=5):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    trace, t0 = [], time.time()
    for _ in range(max_steps):
        resp = ollama.chat(model=model, messages=messages, tools=ollama_tools, options=OPTS)
        messages.append(resp.message)
        if not resp.message.tool_calls:
            return {"answer": resp.message.content, "trace": trace, "secs": round(time.time() - t0, 1)}
        for call in resp.message.tool_calls:
            name, args = call.function.name, dict(call.function.arguments)
            trace.append((name, args))
            try:
                out = await session.call_tool(name, args)          # <- this goes through MCP
                text = result_text(out)
                if result_is_error(out):
                    text = "ERROR: " + text
            except Exception as e:
                text = f"ERROR: {type(e).__name__}: {e}"
            messages.append({"role": "tool", "tool_name": name, "content": text})
    return {"answer": None, "trace": trace, "secs": round(time.time() - t0, 1), "note": "stopped after max_steps"}

async def ask_mcp(questions, model=MODEL):
    async with connect() as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = to_ollama_tools((await session.list_tools()).tools)
            return [await run_agent_mcp(q, session, tools, model) for q in questions]
