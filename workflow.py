"""
LangGraph workflow for the Expense Tracker.

Pipeline:
  contextual  →  router  →  tool  →  summary
  (minimax-m3:cloud  )  (gpt-oss:20b-cloud)  (MCP/one.py)  (minimax-m3:cloud)
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from ollama import chat

from contextual import ContextualAgent
from summary import ExpenseSummarizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VALID_PARAMS: Dict[str, list] = {
    "add_expense":   ["date", "amount", "category", "subcategory", "note"],
    "list_expenses": ["start_date", "end_date"],
    "summarize":     ["start_date", "end_date", "category"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH STATE
# ═══════════════════════════════════════════════════════════════════════════════

class ExpenseState(TypedDict):
    query:        str            # raw user input
    ctx_query:    str            # after contextual resolution
    tool_name:    str            # chosen MCP tool
    parameters:   Dict[str, Any] # tool parameters
    tool_result:  Any            # raw JSON from MCP
    final_answer: str            # natural language answer
    error:        str            # non-empty when any node fails


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER  (gpt-oss:20b-cloud)
# ═══════════════════════════════════════════════════════════════════════════════

class ExpenseQueryRouter:

    def __init__(self, model: str = "gpt-oss:20b-cloud"):
        self.model = model
        logger.info(f"✅ Router initialized: {model}")

    def _system_prompt(self) -> str:
        today       = datetime.now().strftime("%Y-%m-%d")
        yesterday   = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        month_start = datetime.now().strftime("%Y-%m-01")

        return f"""You are a routing assistant for a personal expense tracker.
Pick the correct tool and extract its parameters from the user query.

TODAY = {today}
YESTERDAY = {yesterday}
MONTH_START = {month_start}

════════════════════════
TOOLS
════════════════════════

TOOL 1 — add_expense
  When : User wants to record a new expense.
  Params:
    date        (YYYY-MM-DD, REQUIRED — default {today})
    amount      (number, REQUIRED)
    category    (string, REQUIRED — e.g. food, transport, shopping)
    subcategory (string, optional)
    note        (string, optional)

TOOL 2 — list_expenses
  When : User wants to see individual expense entries.
  Params:
    start_date  (YYYY-MM-DD, REQUIRED)
    end_date    (YYYY-MM-DD, REQUIRED)

TOOL 3 — summarize
  When : User wants totals or aggregates, not a raw list.
  Params:
    start_date  (YYYY-MM-DD, REQUIRED)
    end_date    (YYYY-MM-DD, REQUIRED)
    category    (string, optional)

════════════════════════
DATE RULES
════════════════════════
• "today"      → {today}
• "yesterday"  → {yesterday}
• "this month" → start_date={month_start}, end_date={today}
• No date for add_expense → use {today}

════════════════════════
RESPONSE FORMAT — strict JSON only
════════════════════════
{{"tool_call":{{"name":"<tool_name>","parameters":{{...}}}},"reasoning":"<one sentence>"}}

════════════════════════
FEW-SHOT EXAMPLES
════════════════════════
Q: "I spent 150 on auto today"
{{"tool_call":{{"name":"add_expense","parameters":{{"date":"{today}","amount":150,"category":"transport","subcategory":"auto"}}}},"reasoning":"Recording new expense."}}

Q: "add 800 for groceries at DMart yesterday"
{{"tool_call":{{"name":"add_expense","parameters":{{"date":"{yesterday}","amount":800,"category":"food","subcategory":"groceries","note":"DMart"}}}},"reasoning":"Past expense with note."}}

Q: "show my expenses today"
{{"tool_call":{{"name":"list_expenses","parameters":{{"start_date":"{today}","end_date":"{today}"}}}},"reasoning":"List today's entries."}}

Q: "list what I spent this month"
{{"tool_call":{{"name":"list_expenses","parameters":{{"start_date":"{month_start}","end_date":"{today}"}}}},"reasoning":"List monthly entries."}}

Q: "how much did I spend this month?"
{{"tool_call":{{"name":"summarize","parameters":{{"start_date":"{month_start}","end_date":"{today}"}}}},"reasoning":"Aggregate totals."}}

Q: "total food expenses today"
{{"tool_call":{{"name":"summarize","parameters":{{"start_date":"{today}","end_date":"{today}","category":"food"}}}},"reasoning":"Category-specific total."}}

RESPOND WITH ONLY THE JSON OBJECT. NO MARKDOWN. NO EXTRA TEXT."""

    def _parse(self, text: str) -> Dict[str, Any]:
        text = re.sub(r"```json|```", "", text).strip()
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON found in router response")
        brace, end, in_str, esc = 0, -1, False, False
        for i in range(start, len(text)):
            c = text[i]
            if esc:       esc = False; continue
            if c == "\\": esc = True;  continue
            if c == '"':  in_str = not in_str; continue
            if not in_str:
                if c == "{":  brace += 1
                elif c == "}":
                    brace -= 1
                    if brace == 0:
                        end = i + 1; break
        if end == -1:
            raise ValueError("Unbalanced braces in router response")
        return json.loads(text[start:end])

    def route(self, query: str) -> Optional[Dict[str, Any]]:
        try:
            resp = chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user",   "content": f"Query: {query}"},
                ],
                options={"temperature": 0.0, "num_predict": 200, "seed": 42},
            )
            raw = resp.message.content.strip()
            logger.info(f"[ROUTER] raw output: {raw}")
            parsed = self._parse(raw)
            if "tool_call" not in parsed:
                raise ValueError("Missing tool_call key")
            return parsed
        except Exception as e:
            logger.error(f"[ROUTER] error: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════════

class ExpenseWorkflow:

    def __init__(self, session, model: str = "gpt-oss:20b-cloud"):
        self.session    = session                  # MCP ClientSession (stdio)
        self.contextual = ContextualAgent()        # minimax-m3:cloud  
        self.router     = ExpenseQueryRouter(model)# gpt-oss:20b-cloud
        self.summarizer = ExpenseSummarizer()      # minimax-m3:cloud
        self.app        = self._build()
        logger.info("✅ LangGraph workflow compiled")

    # ── graph construction ────────────────────────────────────────────────────

    def _build(self):
        g = StateGraph(ExpenseState)

        g.add_node("contextual", self._contextual_node)
        g.add_node("router",     self._router_node)
        g.add_node("tool",       self._tool_node)
        g.add_node("summary",    self._summary_node)

        g.set_entry_point("contextual")
        g.add_edge("contextual", "router")
        g.add_conditional_edges("router", self._gate, {"ok": "tool",    "error": END})
        g.add_conditional_edges("tool",   self._gate, {"ok": "summary", "error": END})
        g.add_edge("summary", END)

        return g.compile()

    @staticmethod
    def _gate(state: ExpenseState) -> str:
        return "error" if state.get("error") else "ok"

    # ── NODE 1 — contextual ───────────────────────────────────────────────────

    async def _contextual_node(self, state: ExpenseState) -> dict:
        query = state["query"]
        print("\n" + "="*60)
        print("📅 [STEP 1 — CONTEXTUAL LLM]  minimax-m3:cloud  ")
        print(f"   Input : {query}")

        ctx_query = self.contextual.process_query(query)

        print(f"   Output: {ctx_query}")
        logger.info(f"[CONTEXTUAL] '{query}' → '{ctx_query}'")
        return {"ctx_query": ctx_query}

    # ── NODE 2 — router ───────────────────────────────────────────────────────

    async def _router_node(self, state: ExpenseState) -> dict:
        ctx_query = state["ctx_query"]
        print("\n" + "-"*60)
        print("🔀 [STEP 2 — ROUTER LLM]  gpt-oss:20b-cloud")
        print(f"   Input : {ctx_query}")

        routed = self.router.route(ctx_query)

        if not routed:
            msg = "Router could not select a tool."
            print(f"   ❌ Error: {msg}")
            logger.error(f"[ROUTER] {msg}")
            return {"error": msg, "final_answer": f"❌ {msg}"}

        tool_name  = routed["tool_call"]["name"]
        parameters = routed["tool_call"].get("parameters", {})
        reasoning  = routed.get("reasoning", "")

        # Validate + strip unknown params
        if tool_name not in VALID_PARAMS:
            msg = f"Unknown tool returned by router: '{tool_name}'"
            print(f"   ❌ Error: {msg}")
            logger.error(f"[ROUTER] {msg}")
            return {"error": msg, "final_answer": f"❌ {msg}"}

        allowed    = VALID_PARAMS[tool_name]
        stripped   = [k for k in parameters if k not in allowed]
        parameters = {k: v for k, v in parameters.items() if k in allowed}

        if stripped:
            logger.warning(f"[ROUTER] stripped unexpected params: {stripped}")
            print(f"   ⚠️  Stripped params: {stripped}")

        print(f"   Tool      : {tool_name}")
        print(f"   Parameters: {json.dumps(parameters, indent=6)}")
        print(f"   Reasoning : {reasoning}")
        logger.info(f"[ROUTER] tool={tool_name} params={parameters}")

        return {"tool_name": tool_name, "parameters": parameters}

    # ── NODE 3 — MCP tool call ────────────────────────────────────────────────

    async def _tool_node(self, state: ExpenseState) -> dict:
        tool_name  = state["tool_name"]
        parameters = state["parameters"]

        print("\n" + "-"*60)
        print("🔧 [STEP 3 — MCP TOOL CALL]  one.py (stdio)")
        print(f"   Tool  : {tool_name}")
        print(f"   Params: {json.dumps(parameters, indent=10)}")
        logger.info(f"[TOOL] calling {tool_name} with {parameters}")

        try:
            result = await self.session.call_tool(tool_name, parameters)
        except Exception as e:
            msg = f"MCP call failed: {e}"
            print(f"   ❌ Error: {msg}")
            logger.error(f"[TOOL] {msg}")
            return {"error": msg, "final_answer": f"❌ {msg}"}

        if not result or not result.content:
            msg = "Empty response from MCP server"
            print(f"   ❌ Error: {msg}")
            logger.error(f"[TOOL] {msg}")
            return {"error": msg, "final_answer": f"❌ {msg}"}

        raw = result.content[0].text
        try:
            tool_result = json.loads(raw)
        except json.JSONDecodeError:
            msg = f"MCP returned non-JSON: {raw[:200]}"
            print(f"   ❌ Error: {msg}")
            logger.error(f"[TOOL] {msg}")
            return {"error": msg, "final_answer": f"❌ {msg}"}

        print(f"   ✅ Result: {json.dumps(tool_result, indent=10)[:400]}")
        logger.info(f"[TOOL] result preview: {str(tool_result)[:200]}")
        return {"tool_result": tool_result}

    # ── NODE 4 — summary ──────────────────────────────────────────────────────

    async def _summary_node(self, state: ExpenseState) -> dict:
        query       = state["query"]
        ctx_query   = state["ctx_query"]
        tool_name   = state["tool_name"]
        parameters  = state["parameters"]
        tool_result = state["tool_result"]

        print("\n" + "-"*60)
        print("💬 [STEP 4 — SUMMARY LLM]  minimax-m3:cloud")
        print(f"   Tool result type : {type(tool_result).__name__}")
        print(f"   User query       : {query}")
        print(f"   Contextual query : {ctx_query}")
        logger.info(f"[SUMMARY] dispatching tool_name={tool_name} query='{query}'")

        if tool_name == "add_expense":
            answer = await self.summarizer.summarize_add(query, tool_result, parameters, ctx_query)
        elif tool_name == "list_expenses":
            answer = await self.summarizer.summarize_list(query, tool_result, parameters, ctx_query)
        elif tool_name == "summarize":
            answer = await self.summarizer.summarize_totals(query, tool_result, parameters, ctx_query)
        else:
            answer = json.dumps(tool_result, indent=2)

        print(f"\n{'='*60}")
        print("📝 FINAL ANSWER")
        print(f"{'='*60}")
        print(answer)
        print(f"{'='*60}\n")
        logger.info(f"[SUMMARY] answer length={len(answer)} chars")

        return {"final_answer": answer}

    # ── public entry point ────────────────────────────────────────────────────

    async def run(self, query: str) -> dict:
        print(f"\n{'#'*60}")
        print(f"  NEW QUERY: {query}")
        print(f"{'#'*60}")
        logger.info(f"[WORKFLOW] starting — query='{query}'")

        initial: ExpenseState = {
            "query":        query,
            "ctx_query":    "",
            "tool_name":    "",
            "parameters":   {},
            "tool_result":  None,
            "final_answer": "",
            "error":        "",
        }

        state = await self.app.ainvoke(initial)

        ctx_query   = state.get("ctx_query", query)
        tool_name   = state.get("tool_name", "")
        parameters  = state.get("parameters", {})
        tool_result = state.get("tool_result")
        answer      = state.get("final_answer") or state.get("error") or "No answer generated."
        error       = state.get("error") or None

        steps = [
            {
                "name":        "Contextual",
                "icon":        "📅",
                "model":       "minimax-m3:cloud  ",
                "description": "Date & time resolution",
                "input":       query,
                "output":      ctx_query,
                "status":      "done",
            },
            {
                "name":        "Router",
                "icon":        "🔀",
                "model":       "gpt-oss:20b-cloud",
                "description": "Tool selection",
                "input":       ctx_query,
                "output":      f"{tool_name}  {json.dumps(parameters)}",
                "status":      "done" if tool_name else "error",
            },
            {
                "name":        "Tool",
                "icon":        "🔧",
                "model":       "MCP / one.py",
                "description": "Database query",
                "input":       f"{tool_name}({json.dumps(parameters)})",
                "output":      "Data retrieved" if tool_result is not None else "No data / error",
                "status":      "done" if tool_result is not None else "error",
            },
            {
                "name":        "Summary",
                "icon":        "💬",
                "model":       "minimax-m3:cloud",
                "description": "Natural language answer",
                "input":       "Tool result + user query",
                "output":      (answer[:120] + "…") if len(answer) > 120 else answer,
                "status":      "done" if not error else "error",
            },
        ]

        logger.info(f"[WORKFLOW] done — tool={tool_name} error={error} answer_len={len(answer)}")

        return {
            "answer":      answer,
            "query":       query,
            "ctx_query":   ctx_query,
            "tool_name":   tool_name,
            "parameters":  parameters,
            "steps":       steps,
            "error":       error,
        }
