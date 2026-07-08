"""
Contextual date/time resolution using minimax-m3:cloud  .
Converts relative/ambiguous date-time references in a query to absolute values
before the query reaches the tool router.
"""

import datetime
from ollama import chat


def build_prompts(user_query: str):
    current_datetime = datetime.datetime.now()
    today                = current_datetime.date()
    current_time         = current_datetime.strftime("%H:%M:%S")
    yesterday            = today - datetime.timedelta(days=1)
    day_before_yesterday = today - datetime.timedelta(days=2)
    tomorrow             = today + datetime.timedelta(days=1)
    current_year         = current_datetime.year

    now_full     = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    last_30m     = (current_datetime - datetime.timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
    last_1h      = (current_datetime - datetime.timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    last_2h      = (current_datetime - datetime.timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    last_3h      = (current_datetime - datetime.timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
    last_4h      = (current_datetime - datetime.timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S")
    last_6h      = (current_datetime - datetime.timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")
    last_12h     = (current_datetime - datetime.timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S")
    last_24h     = (current_datetime - datetime.timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")

    system_prompt = (
        "You are a minimal context resolution assistant. Your job is to ONLY add missing date/time context while preserving the user's exact words.\n\n"
        "CORE RULES:\n"
        "1. PRESERVE the user's exact words, grammar, and sentence structure\n"
        "2. ONLY add missing date/time information when needed\n"
        "3. DO NOT rewrite or change the user's question style\n"
        "4. Output ONLY the minimally modified query\n"
        "5. CONVERT relative times (last 1 hour, last 30 mins) to absolute 'from ... to ...' using the formulas provided below\n\n"

        "═══════════════════════════════════════\n"
        "DATE CONVERSION RULES\n"
        "═══════════════════════════════════════\n"
        f"• 'today'     → {today.strftime('%Y-%m-%d')}\n"
        f"• 'yesterday' → {yesterday.strftime('%Y-%m-%d')}\n"
        f"• 'tomorrow'  → {tomorrow.strftime('%Y-%m-%d')}\n"
        f"• 'last two days' / 'last 2 days' → 'from {day_before_yesterday.strftime('%Y-%m-%d')} to {today.strftime('%Y-%m-%d')}'\n\n"

        "MONTH NAME → NUMBER (always use current year unless user specifies a year):\n"
        f"  january/jan   → {current_year}-01\n"
        f"  february/feb  → {current_year}-02\n"
        f"  march/mar     → {current_year}-03\n"
        f"  april/apr     → {current_year}-04\n"
        f"  may           → {current_year}-05\n"
        f"  june/jun      → {current_year}-06\n"
        f"  july/jul      → {current_year}-07\n"
        f"  august/aug    → {current_year}-08\n"
        f"  september/sep → {current_year}-09\n"
        f"  october/oct   → {current_year}-10\n"
        f"  november/nov  → {current_year}-11\n"
        f"  december/dec  → {current_year}-12\n\n"

        "DAY + MONTH PATTERNS → YYYY-MM-DD:\n"
        f"  '4th april'  → {current_year}-04-04\n"
        f"  'april 4'    → {current_year}-04-04\n"
        f"  '11th feb'   → {current_year}-02-11\n"
        f"  '15 march'   → {current_year}-03-15\n"
        "  RULE: Always YYYY-MM-DD. Zero-pad the day (4 → 04).\n"
        "  **CRITICAL: NEVER output partial dates like '05-04'. ALWAYS include the year: '2026-04-05'**\n\n"

        "═══════════════════════════════════════\n"
        "FULL DAY RULES (no time specified)\n"
        "═══════════════════════════════════════\n"
        "If the user mentions a date WITHOUT a specific time → use 00:00:00 to 23:59:59.\n"
        "NEVER use current time for both start and end.\n\n"
        f"  'today'           → 'from {today.strftime('%Y-%m-%d')} 00:00:00 to {today.strftime('%Y-%m-%d')} 23:59:59'\n"
        f"  'yesterday'       → 'from {yesterday.strftime('%Y-%m-%d')} 00:00:00 to {yesterday.strftime('%Y-%m-%d')} 23:59:59'\n"
        f"  '4th april'       → 'from {current_year}-04-04 00:00:00 to {current_year}-04-04 23:59:59'\n"
        f"  '15 march'        → 'from {current_year}-03-15 00:00:00 to {current_year}-03-15 23:59:59'\n"
        "  GOLDEN RULE: date only (no time) = full day = 00:00:00 to 23:59:59.\n\n"

        "═══════════════════════════════════════\n"
        "TIME CONVERSION RULES\n"
        "═══════════════════════════════════════\n"
        "AM: 12am=00:00:00  1am=01:00:00 ... 11am=11:00:00\n"
        "PM: 12pm=12:00:00  1pm=13:00:00 ... 11pm=23:00:00\n\n"
        "NO AM/PM: '9 to 10' → 'from 09:00:00 to 10:00:00' (zero-pad, keep as-is)\n"
        "MIXED:    '9am to 1pm' → 'from 09:00:00 to 13:00:00'\n\n"

        "BEFORE / AFTER / UNTIL / SINCE:\n"
        "'before X' / 'until X' → 'from 00:00:00 to HH:MM:SS'\n"
        "'after X'  / 'since X' → 'from HH:MM:SS to 23:59:59'\n\n"
        "  before 10am → from 00:00:00 to 10:00:00\n"
        "  before 2pm  → from 00:00:00 to 14:00:00\n"
        "  after 6pm   → from 18:00:00 to 23:59:59\n"
        "  since 8am   → from 08:00:00 to 23:59:59\n\n"

        "COMBINED WITH DATE:\n"
        f"  'before 10am today'        → 'from {today.strftime('%Y-%m-%d')} 00:00:00 to {today.strftime('%Y-%m-%d')} 10:00:00'\n"
        f"  'after 6pm yesterday'      → 'from {yesterday.strftime('%Y-%m-%d')} 18:00:00 to {yesterday.strftime('%Y-%m-%d')} 23:59:59'\n"
        f"  'before 10am on 4th april' → 'from {current_year}-04-04 00:00:00 to {current_year}-04-04 10:00:00'\n\n"

        "GOLDEN RULE: 'before X' start is ALWAYS 00:00:00. 'after X' end is ALWAYS 23:59:59.\n\n"

        "═══════════════════════════════════════\n"
        "RELATIVE TIME\n"
        "═══════════════════════════════════════\n"
        f"  'last 30 minutes' → 'from {last_30m} to {now_full}'\n"
        f"  'last 1 hour'     → 'from {last_1h} to {now_full}'\n"
        f"  'last 2 hours'    → 'from {last_2h} to {now_full}'\n"
        f"  'last 3 hours'    → 'from {last_3h} to {now_full}'\n"
        f"  'last 4 hours'    → 'from {last_4h} to {now_full}'\n"
        f"  'last 6 hours'    → 'from {last_6h} to {now_full}'\n"
        f"  'last 12 hours'   → 'from {last_12h} to {now_full}'\n"
        f"  'last 24 hours'   → 'from {last_24h} to {now_full}'\n"
        "  FORMULA: start = now MINUS N hours, end = now. NEVER add.\n\n"

        "═══════════════════════════════════════\n"
        "COMBINED DATE + TIME EXAMPLES\n"
        "═══════════════════════════════════════\n"
        f"  '8pm to 9pm on 4th april'      → 'from {current_year}-04-04 20:00:00 to {current_year}-04-04 21:00:00'\n"
        f"  'yesterday from 2pm to 4pm'    → 'from {yesterday.strftime('%Y-%m-%d')} 14:00:00 to {yesterday.strftime('%Y-%m-%d')} 16:00:00'\n"
        f"  'today 9 to 10'                → 'from {today.strftime('%Y-%m-%d')} 09:00:00 to {today.strftime('%Y-%m-%d')} 10:00:00'\n\n"

        "CRITICAL: When user mentions a specific date like '4th april', use THAT date, NOT today.\n\n"
        "• Keep everything else exactly the same.\n"
    )

    user_prompt = (
        f"Current Date: {today.strftime('%Y-%m-%d')}\n"
        f"Current Time: {current_time}\n"
        f"Current Year: {current_year}\n"
        f"Yesterday: {yesterday.strftime('%Y-%m-%d')}\n"
        f"Day Before Yesterday: {day_before_yesterday.strftime('%Y-%m-%d')}\n"
        f"Tomorrow: {tomorrow.strftime('%Y-%m-%d')}\n\n"
        f"User Query: {user_query}\n\n"
        "Task: Convert all dates and times to standard format, keep everything else EXACTLY the same.\n"
        "CRITICAL REMINDERS:\n"
        "  - Specific date (e.g. '4th april') → use THAT date, not today\n"
        "  - pm times: add 12  (8pm → 20:00:00)\n"
        "  - Month names → numbers (april → 04); zero-pad days (4 → 04)\n"
        "  - ALWAYS output full YYYY-MM-DD dates (NOT '05-04')\n"
        "  - 'last N hours' → 'from ... to ...' using subtraction\n"
        "  - date only (no time) → 00:00:00 to 23:59:59\n"
        "  - 'before X' → 00:00:00 to X;  'after X' → X to 23:59:59\n"
        "  - NEVER set start_time == end_time\n"
        "Just output the updated query — no explanations.\n"
    )

    return system_prompt, user_prompt


class ContextualAgent:
    """Resolves ambiguous date/time references in a query using minimax-m3:cloud  ."""

    def __init__(self, model: str = "minimax-m3:cloud  "):
        self.model = model

    def process_query(self, user_query: str) -> str:
        system_prompt, user_prompt = build_prompts(user_query)
        try:
            resp = chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                options={"temperature": 0.0, "num_predict": 150, "seed": 42},
            )
            resolved = resp.message.content.strip()
            return resolved if resolved else user_query
        except Exception as e:
            # If contextual LLM fails, pass original query through unchanged
            return user_query
