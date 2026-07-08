"""
Expense Summarizer — converts raw MCP tool results into intelligent analysis
with recommendations using minimax-m3:cloud via Ollama AsyncClient.
"""

import logging
from typing import List, Dict, Any, Optional

from ollama import AsyncClient

logger = logging.getLogger(__name__)


class ExpenseSummarizer:

    def __init__(self, model: str = "minimax-m3:cloud"):
        self.model = model
        logger.info(f"ExpenseSummarizer ready: {model}")

    # ── shared async call ─────────────────────────────────────────────────────

    async def _call(self, system: str, user: str, max_tokens: int = 800) -> str:
        try:
            client = AsyncClient()
            resp = await client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                options={"temperature": 0.5, "num_predict": max_tokens},
            )
            return (resp.message.content or "").strip()
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return f"(summary unavailable: {e})"

    # ── data formatters ───────────────────────────────────────────────────────

    @staticmethod
    def _fmt_list(rows: List[Dict[str, Any]]) -> str:
        lines = []
        for e in rows:
            sub  = f" / {e['subcategory']}" if e.get("subcategory") else ""
            note = f" [{e['note']}]"        if e.get("note")        else ""
            lines.append(f"  {e['date']}  ₹{e['amount']}  {e['category']}{sub}{note}")
        return "\n".join(lines)

    @staticmethod
    def _fmt_totals(rows: List[Dict[str, Any]]) -> str:
        lines = []
        for r in rows:
            lines.append(f"  {r['category']}: ₹{r['total_amount']}")
        return "\n".join(lines)

    @staticmethod
    def _compute_list_stats(rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return ""
        total      = sum(r["amount"] for r in rows)
        count      = len(rows)
        avg        = total / count
        max_e      = max(rows, key=lambda r: r["amount"])
        min_e      = min(rows, key=lambda r: r["amount"])
        by_cat: Dict[str, float] = {}
        for r in rows:
            by_cat[r["category"]] = by_cat.get(r["category"], 0) + r["amount"]
        top_cat    = max(by_cat, key=lambda k: by_cat[k])
        return (
            f"  Total entries : {count}\n"
            f"  Grand total   : ₹{total:.0f}\n"
            f"  Average/entry : ₹{avg:.0f}\n"
            f"  Highest entry : ₹{max_e['amount']} on {max_e['date']} ({max_e['category']})\n"
            f"  Lowest entry  : ₹{min_e['amount']} on {min_e['date']} ({min_e['category']})\n"
            f"  Top category  : {top_cat} (₹{by_cat[top_cat]:.0f})\n"
            f"  Category breakdown:\n" +
            "\n".join(f"    {k}: ₹{v:.0f}" for k, v in sorted(by_cat.items(), key=lambda x: -x[1]))
        )

    @staticmethod
    def _compute_totals_stats(rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return ""
        grand      = sum(r["total_amount"] for r in rows)
        top        = max(rows, key=lambda r: r["total_amount"])
        pct_top    = (top["total_amount"] / grand * 100) if grand else 0
        return (
            f"  Grand total   : ₹{grand:.0f}\n"
            f"  Top category  : {top['category']} (₹{top['total_amount']:.0f}, {pct_top:.0f}% of total)\n"
            f"  Categories    : {len(rows)}"
        )

    # ── public methods ────────────────────────────────────────────────────────

    async def summarize_add(
        self,
        query: str,
        result: Dict[str, Any],
        params: Dict[str, Any],
        ctx_query: Optional[str] = None,
    ) -> str:
        if not (isinstance(result, dict) and result.get("status") == "ok"):
            return f"The expense could not be saved. Server response: {result}"

        amount   = params.get("amount", 0)
        category = params.get("category", "")
        date     = params.get("date", "")
        sub      = params.get("subcategory", "")
        note     = params.get("note", "")

        system = """You are an intelligent personal finance assistant.

When a user records an expense:
1. Confirm what was saved in one natural sentence.
2. If the amount seems notably high for that category (e.g. ₹2000+ on food, ₹5000+ on shopping), gently flag it.
3. Offer one brief, relevant saving tip or alternative for that category if applicable.
4. Keep the tone friendly and non-judgmental — like a helpful friend, not a lecture.
5. Total response: 2–4 sentences max."""

        user = (
            f"User said: {query}\n"
            f"Contextual query: {ctx_query or query}\n\n"
            f"Expense saved:\n"
            f"  ID       : {result['id']}\n"
            f"  Amount   : ₹{amount}\n"
            f"  Category : {category}\n"
            f"  Subcategory: {sub or 'N/A'}\n"
            f"  Date     : {date}\n"
            f"  Note     : {note or 'N/A'}\n\n"
            "Confirm the save, then add any relevant tip or observation."
        )
        return await self._call(system, user, max_tokens=200)

    async def summarize_list(
        self,
        query: str,
        rows: List[Dict[str, Any]],
        params: Dict[str, Any],
        ctx_query: Optional[str] = None,
    ) -> str:
        start  = params.get("start_date", "")
        end    = params.get("end_date", "")
        period = f"{start} to {end}" if start != end else start

        if not rows:
            return (
                f"No expenses were recorded for {period}. "
                "If you expected to see entries, check that you've added them or try a wider date range."
            )

        stats = self._compute_list_stats(rows)
        data  = self._fmt_list(rows)

        system = """You are an intelligent personal finance assistant with analytical skills.

When presenting an expense list:
1. Start by directly answering what the user asked — e.g. "Here's what you spent on [period]."
2. List the expenses clearly (don't skip any).
3. After the list, provide a short analysis section:
   - Total and average spend
   - Which category consumed the most
   - Any unusual or high single expenses worth flagging
4. End with 1–2 concrete, actionable recommendations based on the actual spending pattern you see.
   Examples: suggest switching from Zomato to cooking, carpooling if transport is high, etc.
5. Tone: conversational, helpful, like a smart friend reviewing your bank statement."""

        user = (
            f"User asked: {query}\n"
            f"Contextual query: {ctx_query or query}\n\n"
            f"Period: {period}\n\n"
            f"Raw expenses:\n{data}\n\n"
            f"Computed statistics:\n{stats}\n\n"
            "Present the list, analyse the spending, and give actionable recommendations."
        )
        return await self._call(system, user, max_tokens=600)

    async def summarize_totals(
        self,
        query: str,
        rows: List[Dict[str, Any]],
        params: Dict[str, Any],
        ctx_query: Optional[str] = None,
    ) -> str:
        start    = params.get("start_date", "")
        end      = params.get("end_date", "")
        period   = f"{start} to {end}" if start != end else start
        category = params.get("category")

        if not rows:
            scope = f"in '{category}'" if category else ""
            return (
                f"No expenses found {scope} for {period}. "
                "Try a wider date range or check if expenses have been added for this period."
            )

        stats = self._compute_totals_stats(rows)
        data  = self._fmt_totals(rows)
        grand = sum(r["total_amount"] for r in rows)

        system = """You are an intelligent personal finance assistant.

When presenting a spending summary:
1. Open with a direct answer to what the user asked (e.g. total amount for the period).
2. Break down by category — mention which is highest and which is lowest.
3. Provide percentage share for each category so the user understands where their money goes.
4. Analyse the pattern:
   - Is spending balanced or skewed heavily to one area?
   - Flag any category that seems disproportionately high.
5. Give 2–3 specific, actionable recommendations to reduce spend or rebalance the budget.
   Be specific — mention the actual category names and realistic alternatives.
6. If the user mentioned a specific category in their question, focus extra attention on that.
7. Tone: insightful, warm, non-preachy. Like a financial advisor who knows the user well."""

        user = (
            f"User asked: {query}\n"
            f"Contextual query: {ctx_query or query}\n\n"
            f"Period: {period}\n"
            f"Category filter: {category or 'all categories'}\n\n"
            f"Spending by category:\n{data}\n"
            f"Grand total: ₹{grand:.0f}\n\n"
            f"Key statistics:\n{stats}\n\n"
            "Analyse the spending and give specific recommendations."
        )
        return await self._call(system, user, max_tokens=600)
