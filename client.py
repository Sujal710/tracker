"""
Expense Tracker CLI Client

Connects to one.py over stdio (MCP), then hands every query to
ExpenseWorkflow (LangGraph) which runs:
  contextual (minimax-m3:cloud  )
    → router (gpt-oss:20b-cloud)
      → tool (MCP / one.py)
        → summary (minimax-m3:cloud)
"""

import asyncio
import logging
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from workflow import ExpenseWorkflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ExpenseClient:

    def __init__(self, model: str = "gpt-oss:20b-cloud"):
        self.model    = model
        self.session  = None
        self.workflow = None

    async def connect(self, server_script: str = "one.py"):
        logger.info(f"[CLIENT] connecting to MCP server: {server_script}")
        params = StdioServerParameters(command=sys.executable, args=[server_script])

        self._streams_ctx = stdio_client(params)
        streams = await self._streams_ctx.__aenter__()

        self._session_ctx = ClientSession(*streams)
        self.session = await self._session_ctx.__aenter__()
        await self.session.initialize()

        tools = (await self.session.list_tools()).tools
        logger.info(f"[CLIENT] connected — tools: {[t.name for t in tools]}")

        self.workflow = ExpenseWorkflow(self.session, model=self.model)

        print(f"\n{'='*60}")
        print("✅ EXPENSE TRACKER CONNECTED")
        print(f"   Server : {server_script}")
        print(f"   Tools  : {[t.name for t in tools]}")
        print(f"   Router : {self.model}")
        print(f"{'='*60}\n")

    async def cleanup(self):
        logger.info("[CLIENT] shutting down")
        if hasattr(self, "_session_ctx"):
            await self._session_ctx.__aexit__(None, None, None)
        if hasattr(self, "_streams_ctx"):
            await self._streams_ctx.__aexit__(None, None, None)

    async def run(self, query: str) -> str:
        return await self.workflow.run(query)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    client = ExpenseClient()
    try:
        await client.connect("one.py")

        print("💰 EXPENSE TRACKER CHATBOT")
        print("Try:")
        print("  'I spent 250 on groceries'")
        print("  'show my expenses today'")
        print("  'how much did I spend this month?'")
        print("  'total food expenses this week'")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                query = input("💬 > ").strip()
                if not query:
                    continue
                if query.lower() in ("quit", "exit", "q"):
                    print("👋 Goodbye!")
                    break
                result = await client.run(query)
                print(result["answer"])
                print("-" * 60)
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                break
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
