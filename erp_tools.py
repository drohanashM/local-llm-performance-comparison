"""Pandas-backed tools over the synthetic purchase-order data.

These are plain Python functions. Notebook 03 calls them directly, and
erp_server.py serves the same functions over MCP.
"""
from pathlib import Path

import pandas as pd

THR = 50_000          # approval threshold in INR
MAX_ROWS = 10         # keep tool results short: the model's context window is small
DATA = Path(__file__).parent / "data" / "pos.csv"

# The CSV also holds the true labels (violation_type, ...). Tools must never expose them.
PUBLIC = ["po_id", "date", "vendor", "amount", "created_by",
          "approved_by", "payment_processed_by", "description"]
_df = pd.read_csv(DATA, parse_dates=["date"])[PUBLIC]

ROLE_COL = {"created": "created_by", "approved": "approved_by", "paid": "payment_processed_by"}


def find_same_creator_and_payer() -> dict:
    """Find purchase orders where the same user both created the PO and processed its payment (a segregation-of-duties violation). Returns the total count and up to 10 examples."""
    d = _df[_df.created_by == _df.payment_processed_by]
    return {"count": int(len(d)),
            "examples": d[["po_id", "created_by"]].head(MAX_ROWS).to_dict("records")}


def find_self_approved() -> dict:
    """Find purchase orders where the same user both created the PO and approved it (a segregation-of-duties violation). Returns the total count and up to 10 examples."""
    d = _df[_df.created_by == _df.approved_by]
    return {"count": int(len(d)),
            "examples": d[["po_id", "created_by"]].head(MAX_ROWS).to_dict("records")}


def find_split_pos() -> dict:
    """Find possible split purchase orders: groups of 3 or more POs created by the same user for the same vendor within 7 days, each just below the 50,000 approval threshold. Returns the number of groups and the groups."""
    near = _df[(_df.amount >= 0.8 * THR) & (_df.amount < THR)]
    groups = []
    for (creator, vendor), g in near.groupby(["created_by", "vendor"]):
        if len(g) >= 3 and (g.date.max() - g.date.min()).days <= 7:
            groups.append({"created_by": creator, "vendor": vendor, "po_ids": g.po_id.tolist()})
    return {"count": len(groups), "groups": groups[:MAX_ROWS]}


def get_po(po_id: str) -> dict:
    """Look up one purchase order by its ID, for example PO-0104. Returns its date, vendor, amount, creator, approver and payment processor."""
    d = _df[_df.po_id == po_id.strip().upper()]
    if d.empty:
        return {"error": f"No purchase order with id {po_id}"}
    r = d.iloc[0]
    return {"po_id": r.po_id, "date": str(r.date.date()), "vendor": r.vendor,
            "amount": float(r.amount), "created_by": r.created_by,
            "approved_by": r.approved_by, "payment_processed_by": r.payment_processed_by,
            "description": r.description}


def list_pos_by_user(user_id: str, role: str) -> dict:
    """List purchase orders that a user created, approved or paid. role must be exactly one of: created, approved, paid. Returns the total count and up to 10 PO IDs."""
    col = ROLE_COL.get(role.strip().lower())
    if col is None:
        return {"error": "role must be one of: created, approved, paid"}
    uid = user_id.strip().upper()
    d = _df[_df[col] == uid]
    note = (f"list is partial: only the first {MAX_ROWS} of {len(d)} PO IDs are shown"
            if len(d) > MAX_ROWS else "list is complete")
    return {"user_id": uid, "role": role.strip().lower(), "count": int(len(d)),
            "po_ids": d.po_id.head(MAX_ROWS).tolist(), "note": note}


TOOLS = [find_same_creator_and_payer, find_self_approved, find_split_pos, get_po, list_pos_by_user]
TOOL_MAP = {f.__name__: f for f in TOOLS}
