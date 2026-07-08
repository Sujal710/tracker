"""
FastAPI server — Expense Tracker AI
Connects to one.py (MCP stdio) on startup, exposes POST /api/chat,
and serves the frontend from static/index.html.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from workflow import ExpenseWorkflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── global state ──────────────────────────────────────────────────────────────
_workflow: ExpenseWorkflow = None
_mcp_lock = asyncio.Lock()   # stdio MCP is sequential — serialize calls


# ── lifespan: connect MCP on startup, disconnect on shutdown ──────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _workflow

    logger.info("🚀 [STARTUP] Connecting to MCP server (one.py)…")
    params = StdioServerParameters(command=sys.executable, args=["one.py"])

    streams_ctx = stdio_client(params)
    streams = await streams_ctx.__aenter__()

    session_ctx = ClientSession(*streams)
    session = await session_ctx.__aenter__()
    await session.initialize()

    tools = (await session.list_tools()).tools
    logger.info(f"✅ [STARTUP] MCP connected — tools: {[t.name for t in tools]}")

    _workflow = ExpenseWorkflow(session)
    logger.info("✅ [STARTUP] LangGraph workflow ready")

    yield  # ── server runs ──────────────────────────────────────────────────

    logger.info("🛑 [SHUTDOWN] Disconnecting MCP…")
    await session_ctx.__aexit__(None, None, None)
    await streams_ctx.__aexit__(None, None, None)
    logger.info("✅ [SHUTDOWN] Done")


# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Expense Tracker AI", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (CSS/JS assets if any)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str


class StepInfo(BaseModel):
    name:        str
    icon:        str
    model:       str
    description: str
    input:       str
    output:      str
    status:      str   # "done" | "error" | "skipped"


class ChatResponse(BaseModel):
    answer:     str
    query:      str
    ctx_query:  str
    tool_name:  str
    parameters: dict
    steps:      list[StepInfo]
    error:      str | None


# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html = static_dir / "index.html"
    if not html.exists():
        raise HTTPException(status_code=404, detail="Frontend not built — static/index.html missing")
    return HTMLResponse(html.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health():
    return {
        "status":          "ok",
        "workflow_ready":  _workflow is not None,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not _workflow:
        raise HTTPException(status_code=503, detail="Workflow not initialised yet")

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty")

    logger.info(f"[API] POST /api/chat  query='{query}'")

    async with _mcp_lock:          # one MCP call at a time (stdio constraint)
        result = await _workflow.run(query)

    logger.info(f"[API] response ready — tool={result['tool_name']} error={result['error']}")
    return result
