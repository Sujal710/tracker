from fastmcp import FastMCP
import os
import sqlite3

db_path = os.path.join(os.path.dirname(__file__), "expenses.db")
categories_path = os.path.join(os.path.dirname(__file__), "categories.json")

mcp= FastMCP("ExpenseTracker")

def init_db():
    with sqlite3.connect(db_path) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS expenses(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                subcategory TEXT DEFAULT '',
                note TEXT DEFAULT ''
            )
        """)
init_db()
@mcp.tool()
def add_expense(date, amount, category, subcategory="", note=""):
    '''Add a new expense entry to the database.'''
    with sqlite3.connect(db_path) as c:
        cur = c.execute(
            "INSERT INTO expenses(date, amount, category, subcategory, note) VALUES (?,?,?,?,?)",
            (date, amount, category, subcategory, note)
        )
        return {"status": "ok", "id": cur.lastrowid}
@mcp.tool()
def list_expenses(start_date, end_date):
    with sqlite3.connect(db_path) as c:
        cur=c.execute(
            """
            SELECT id, date, amount, category, subcategory, note
            FROM expenses
            WHERE date BETWEEN ? AND ?
            ORDER BY id ASC
            """,
            (start_date, end_date)
        )
        cols=[d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
@mcp.tool()
def summarize(start_date, end_date, category=None):
    query= ("""
            SELECT category, SUM(amount) AS total_amount
            FROM expenses
            WHERE date BETWEEN ? AND ?
            """
            )
    params = [start_date, end_date]

    if category:
        query += " AND category = ?"
        params.append(category)
    query += " GROUP BY category"
    cur = c.execute(query, params)
    cols= [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]
@mcp.resource("expense://categories",mime_type="application/json")
def categories():
    with open(categories_path, "r") as f:
        return f.read()
if __name__ == "__main__":
    mcp.run()