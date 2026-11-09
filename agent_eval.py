"""Evaluation harness: 20 questions with known answers, scored on tool choice,
arguments and final answer, for the direct (plain-function) and MCP agents."""
import json, random, re, subprocess, time
from pathlib import Path

import pandas as pd

import agent_core as A
import erp_tools

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------- questions
def build_questions(path="data/pos.csv", seed=11):
    df = pd.read_csv(path)
    rng = random.Random(seed)
    dev_pos, dev_users = {"PO-0104", "PO-9999"}, {"U1488"}      # used in the dev questions: not reused here

    # ground truth comes from the label column, not from the tools' own logic
    n_cp = int((df.violation_type == "create_pay").sum())
    n_sa = int((df.violation_type == "self_approve").sum())
    n_split = int((df.violation_type == "split_po").sum()) // 3
    assert n_cp == erp_tools.find_same_creator_and_payer()["count"], "tools and labels disagree (create_pay)"
    assert n_sa == erp_tools.find_self_approved()["count"], "tools and labels disagree (self_approve)"
    assert n_split == erp_tools.find_split_pos()["count"], "tools and labels disagree (split_po)"

    clean = df[(df.violation_type == "none") & ~df.po_id.isin(dev_pos)]
    lookups = clean.sample(n=6, random_state=seed).reset_index(drop=True)
    cp = df[df.violation_type == "create_pay"].sample(n=1, random_state=seed).iloc[0]
    sa = df[df.violation_type == "self_approve"].sample(n=1, random_state=seed).iloc[0]
    neg = lookups.iloc[5]

    def users(col, k):
        pool = sorted(set(df[col]) - dev_users)
        return rng.sample(pool, k)
    u_created, u_approved, u_paid = users("created_by", 2), users("approved_by", 1), users("payment_processed_by", 1)
    count = lambda col, u: int((df[col] == u).sum())

    Q = []
    def add(cat, q, expect, check):
        Q.append({"id": f"q{len(Q)+1:02d}", "cat": cat, "q": q, "expect": expect, "check": check})

    # aggregate questions (different wording from the dev questions)
    add("aggregate", "How many purchase orders have the same person as both the creator and the payment processor?",
        [("find_same_creator_and_payer", {})], ("number", n_cp))
    add("aggregate", "Did anyone pay for a purchase order they raised themselves? How many such POs are there?",
        [("find_same_creator_and_payer", {})], ("number", n_cp))
    add("aggregate", "How many POs were approved by the same user who created them?",
        [("find_self_approved", {})], ("number", n_sa))
    add("aggregate", "Count the purchase orders where a user signed off on their own request.",
        [("find_self_approved", {})], ("number", n_sa))
    add("aggregate", "How many groups of possibly split purchase orders can you find?",
        [("find_split_pos", {})], ("number", n_split))
    add("aggregate", "Is anyone splitting orders to stay under the approval limit? Give the number of suspicious groups.",
        [("find_split_pos", {})], ("number", n_split))
    # single-PO lookups
    r = lookups
    add("lookup", f"Which user created {r.po_id[0]}?", [("get_po", {"po_id": r.po_id[0]})], ("contains", r.created_by[0]))
    add("lookup", f"Who processed the payment for {r.po_id[1]}?", [("get_po", {"po_id": r.po_id[1]})], ("contains", r.payment_processed_by[1]))
    add("lookup", f"What vendor supplied {r.po_id[2]}?", [("get_po", {"po_id": r.po_id[2]})], ("contains", r.vendor[2]))
    add("lookup", f"How much was {r.po_id[3]} for?", [("get_po", {"po_id": r.po_id[3]})], ("number", float(r.amount[3])))
    add("lookup", f"Which user approved {r.po_id[4]}?", [("get_po", {"po_id": r.po_id[4]})], ("contains", r.approved_by[4]))
    # per-user counts
    add("user_count", f"How many POs did {u_created[0]} create?", [("list_pos_by_user", {"user_id": u_created[0], "role": "created"})], ("number", count("created_by", u_created[0])))
    add("user_count", f"How many purchase orders has {u_created[1]} created in total?", [("list_pos_by_user", {"user_id": u_created[1], "role": "created"})], ("number", count("created_by", u_created[1])))
    add("user_count", f"How many POs has {u_approved[0]} approved?", [("list_pos_by_user", {"user_id": u_approved[0], "role": "approved"})], ("number", count("approved_by", u_approved[0])))
    add("user_count", f"Count the purchase orders whose payment {u_paid[0]} processed.", [("list_pos_by_user", {"user_id": u_paid[0], "role": "paid"})], ("number", count("payment_processed_by", u_paid[0])))
    # yes/no about one PO (the right route is get_po: the list tools only show 10 examples)
    add("yes_no", f"Was {cp.po_id} created and paid by the same user? Answer yes or no first.", [("get_po", {"po_id": cp.po_id})], ("yesno", "yes"))
    add("yes_no", f"Did the creator of {sa.po_id} also approve it? Answer yes or no first.", [("get_po", {"po_id": sa.po_id})], ("yesno", "yes"))
    add("yes_no", f"Was {neg.po_id} both created and paid by one user? Answer yes or no first.", [("get_po", {"po_id": neg.po_id})], ("yesno", "no"))
    # PO that does not exist
    add("not_found", "Show me the details of PO-0777.", [("get_po", {"po_id": "PO-0777"})], ("notfound", None))
    add("not_found", "What is the amount of PO-0000?", [("get_po", {"po_id": "PO-0000"})], ("notfound", None))
    return Q

# ---------------------------------------------------------------- scoring
_UNITS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}
WORDS = {w: i for i, w in enumerate(_UNITS)}
WORDS.update(_TENS)
WORDS.update({f"{t}-{u}": v + i for t, v in _TENS.items() for i, u in enumerate(_UNITS[1:10], start=1)})
_WORD_ORDER = sorted(WORDS, key=len, reverse=True)       # match "twenty-five" before "five"

def numbers_in(text):
    t = re.sub(r"\bPO-\d+\b|\bU\d+\b", " ", text, flags=re.I).replace(",", "")
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", t)]
    low = t.lower()
    for w in _WORD_ORDER:
        if re.search(rf"\b{w}\b", low):
            nums.append(float(WORDS[w]))
            low = re.sub(rf"\b{w}\b", " ", low)
    return nums

NOT_FOUND = ["not found", "no purchase order", "unable to find", "could not find", "couldn't find", "can't find",
             "cannot find", "does not exist", "doesn't exist", "no record", "no po ", "no information"]

def check_answer(answer, check):
    if not answer:
        return False
    kind, val = check
    low = answer.lower()
    if kind == "number":
        return any(abs(x - float(val)) < 0.5 for x in numbers_in(answer))
    if kind == "contains":
        return str(val).lower() in low
    if kind == "yesno":
        m = re.match(r"\W*(yes|no)\b", low)
        return bool(m) and m.group(1) == val
    if kind == "notfound":
        return any(p in low for p in NOT_FOUND)
    raise ValueError(kind)

_norm = lambda x: str(x).strip().upper()

def check_trace(trace, expect):
    names = {n for n, _ in expect}
    tool_ok = any(n in names for n, _ in trace)
    args_ok = any(n == en and all(_norm(a.get(k)) == _norm(v) for k, v in ea.items())
                  for n, a in trace for en, ea in expect)
    return tool_ok, args_ok

def check_trace_strict(trace, expect):
    """Like check_trace's args_ok, but a call that also passes arguments the tool does not have
    (e.g. user_id to a tool with no parameters) does not count as correct."""
    import inspect
    for n, a in trace:
        if n not in erp_tools.TOOL_MAP:
            continue
        allowed = set(inspect.signature(erp_tools.TOOL_MAP[n]).parameters)
        for en, ea in expect:
            if n == en and all(_norm(a.get(k)) == _norm(v) for k, v in ea.items()) and set(a) <= allowed:
                return True
    return False

def add_strict_args(df, questions):
    """Add an args_strict column to loaded results, using only the stored traces (no model calls)."""
    exp = {q["id"]: q["expect"] for q in questions}
    return df.assign(args_strict=[check_trace_strict(t, exp[i]) for t, i in zip(df.trace, df.id)])

# ---------------------------------------------------------------- running
def _cache_path(tag, mode, model):
    return RESULTS / f"{tag}_{mode}_{model.replace(':', '_')}.jsonl"

def _done_ids(path):
    if not path.exists():
        return set()
    return {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}

def _stop(model):
    try:
        subprocess.run(["ollama", "stop", model], capture_output=True)
    except FileNotFoundError:
        pass

def _record(q, model, mode, out, error=None):
    trace = out.get("trace", []) if out else []
    tool_ok, args_ok = check_trace(trace, q["expect"])
    ans = out.get("answer") if out else None
    rec = {"model": model, "mode": mode, "id": q["id"], "cat": q["cat"], "question": q["q"],
           "answer": ans, "trace": [[n, a] for n, a in trace], "secs": out.get("secs") if out else None,
           "tool_ok": tool_ok, "args_ok": args_ok, "answer_ok": check_answer(ans, q["check"]), "error": error}
    return rec

async def eval_model(model, questions, mode="mcp", tag="agent_eval"):
    """Run every question not yet in the cache file. mode is 'mcp' or 'direct'."""
    path = _cache_path(tag, mode, model)
    todo = [q for q in questions if q["id"] not in _done_ids(path)]
    if not todo:
        return path

    def save(rec):
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    async def run_all(ask, warm):
        try:
            warm()                                   # load the model and the shared prompt prefix first
        except Exception as e:
            for q in todo:
                save(_record(q, model, mode, None, error=f"{type(e).__name__}: {e}"))
            return
        for q in todo:
            try:
                out = await ask(q["q"])
                save(_record(q, model, mode, out))
            except Exception as e:
                save(_record(q, model, mode, None, error=f"{type(e).__name__}: {e}"))
            print(f"{model} {mode} {q['id']} done")

    msgs = [{"role": "system", "content": A.SYSTEM}, {"role": "user", "content": "Hello"}]
    if mode == "mcp":
        async with A.connect() as (r, w):
            async with A.ClientSession(r, w) as session:
                await session.initialize()
                tools = A.to_ollama_tools((await session.list_tools()).tools)
                warm = lambda: A.ollama.chat(model=model, messages=msgs, tools=tools, options={**A.OPTS, "num_predict": 5})
                async def ask(text): return await A.run_agent_mcp(text, session, tools, model)
                await run_all(ask, warm)
    else:
        warm = lambda: A.ollama.chat(model=model, messages=msgs, tools=A.TOOLS, options={**A.OPTS, "num_predict": 5})
        async def ask(text): return A.run_agent(text, model)
        await run_all(ask, warm)
    _stop(model)
    return path

# ---------------------------------------------------------------- reporting
def load_results(tag="agent_eval"):
    rows = [json.loads(l) for p in sorted(RESULTS.glob(f"{tag}_*.jsonl"))
            for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.DataFrame(rows)

def summarize(df):
    df = df.assign(no_tool_call=df.trace.map(len) == 0, errored=df.error.notna())
    g = df.groupby(["mode", "model"])
    return pd.DataFrame({
        "n": g.size(),
        "tool_ok_%": g.tool_ok.mean() * 100,
        "args_ok_%": g.args_ok.mean() * 100,
        **({"args_strict_%": g.args_strict.mean() * 100} if "args_strict" in df else {}),
        "answer_ok_%": g.answer_ok.mean() * 100,
        "no_tool_call": g.no_tool_call.sum(),
        "errors": g.errored.sum(),
        "secs_per_q": g.secs.mean(),
    }).round(1)

def by_category(df, col="answer_ok"):
    return (df.pivot_table(index=["mode", "model"], columns="cat", values=col, aggfunc="mean") * 100).round(0)
