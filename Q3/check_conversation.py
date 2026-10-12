#!/usr/bin/env python3
"""
Check whether a generic_followup agent took the correct branch at each question.

How it works
------------
For every question the agent asks in a thread:
  1. The caller's reply is classified into one of that question's scenarios
     (e.g. "active" / "inactive") using Claude, because replies are free-form
     ("I use it for my doctor visits all the time" means active).
  2. The config tells us where that scenario should lead: another question,
     an outcome (disposition), or _complete.
  3. The agent's next message is matched to a question/outcome in the config.
     This is done by text matching first (the agent mostly reads scripted text);
     Claude is only asked when the agent paraphrased and no match is found.
  4. Expected vs. actual are compared.

Only branching is checked. Wording, tone, re-asking things the caller
volunteered earlier, etc. are out of scope.

Usage
-----
  pip install -r requirements.txt
  echo "ANTHROPIC_API_KEY=..." > .env   # .env lives next to this script
  python check_paths.py conversations.json
  python check_conversation.py conversations.json --threads thread_02 thread_04 --json report.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

MATCH_THRESHOLD = 0.75  # fraction of a scripted line's words that must appear in the agent message
DEFAULT_MODEL = "claude-haiku-5-5"


# ---------------------------------------------------------------- text matching

def _tokens(text):
    text = text.lower().replace("\u2019", "'")
    return re.findall(r"[a-z0-9']+", text)


def _coverage(target, message):
    """(fraction of target words present in message, number of words present)."""
    t = _tokens(target)
    if not t:
        return 0.0, 0
    m = set(_tokens(message))
    hits = sum(1 for w in t if w in m)
    return hits / len(t), hits


# ---------------------------------------------------------------- flow config

class Flow:
    """The question graph from agent_config.engine_config."""

    def __init__(self, engine_config):
        self.questions = {}  # "q2_active.a" -> question dict
        self.labels = {}     # "q2_active.a" -> "Q2 Active Medi-Cal"
        self.order = []
        for item in engine_config["items"]:
            for q in item["questions"]:
                key = f'{item["id"]}.{q["id"]}'
                self.questions[key] = q
                self.labels[key] = item.get("label", item["id"])
                self.order.append(key)
        self.outcomes = {o["id"]: o for o in engine_config.get("outcomes", [])}
        # _complete routes that speak a line we can recognise
        self.completions = {}
        for key, q in self.questions.items():
            for s in q.get("scenarios", []):
                if s.get("goto") == "_complete" and s.get("say"):
                    self.completions[f"{key}/{s['id']}"] = s["say"]

    # where a scenario should lead
    def expected_next(self, qkey, scenario_id):
        q = self.questions[qkey]
        sc = next((s for s in q.get("scenarios", []) if s["id"] == scenario_id), None)
        if sc is None:
            return self._route(q.get("default_route"), qkey, None)
        if sc.get("disposition"):
            return ("outcome", sc["disposition"])
        return self._route(sc.get("goto"), qkey, sc)

    def _route(self, goto, qkey, sc):
        if goto is None:
            return ("none", None)
        if goto == "_complete":
            return ("complete", f"{qkey}/{sc['id']}" if sc and sc.get("say") else None)
        m = re.fullmatch(r"item_(.+):question_(.+)", goto)
        if m:
            return ("question", f"{m.group(1)}.{m.group(2)}")
        return ("unknown", goto)

    # which node an agent message corresponds to (deterministic)
    def identify(self, message):
        # A question in the message wins: "say" lines are often spoken right
        # before the next question (e.g. "Got it ... Which plan is that?").
        best = None
        for key, q in self.questions.items():
            score, hits = _coverage(q["text"], message)
            if score >= MATCH_THRESHOLD and (best is None or (score, hits) > best[0]):
                best = ((score, hits), ("question", key))
        if best:
            return best[1]
        # Otherwise a closing line. Rank by words matched so that a long line
        # isn't beaten by a short one it happens to contain ("Thanks for letting me know").
        candidates = [("outcome", k, o["say"]) for k, o in self.outcomes.items() if o.get("say")]
        candidates += [("complete", k, say) for k, say in self.completions.items()]
        for kind, key, say in candidates:
            score, hits = _coverage(say, message)
            if score >= MATCH_THRESHOLD and (best is None or hits > best[0]):
                best = (hits, (kind, key))
        return best[1] if best else None

    def describe(self, node):
        if node is None:
            return "unrecognised"
        kind, key = node
        if kind == "question":
            return f"ask {self.labels.get(key, key)}"
        if kind == "outcome":
            return f"outcome {key}"
        if kind == "complete":
            return "_complete (start renewal)" if key else "_complete"
        return f"{kind} {key}"


def same_node(expected, actual):
    if actual is None:
        return False
    if expected[0] == "complete" and actual[0] == "complete":
        return expected[1] is None or expected[1] == actual[1]
    return expected == actual


# ---------------------------------------------------------------- LLM calls

class Judge:
    def __init__(self, model):
        import anthropic  # imported here so --help works without it
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / ".env")  # sets ANTHROPIC_API_KEY
        self.client = anthropic.Anthropic()
        self.model = model

    def _call(self, prompt, enum, tool_name):
        tool = {
            "name": tool_name,
            "description": "Record your answer.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "One short sentence."},
                    "answer": {"type": "string", "enum": enum},
                },
                "required": ["reason", "answer"],
            },
        }
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": prompt}],
        )
        block = next(b for b in resp.content if b.type == "tool_use")
        return block.input["answer"], block.input["reason"]

    def classify_reply(self, question, transcript, reply):
        scenarios = question.get("scenarios", [])
        lines = "\n".join(f'- "{s["id"]}": {s.get("means", "")}' for s in scenarios)
        prompt = f"""You are grading a phone screening call. The agent asked this question:

"{question["text"]}"

Possible meanings of the caller's answer:
{lines}
- "unclear": the reply does not answer this question clearly enough to pick one of the above.

Conversation so far:
{transcript}

Caller's reply to that question:
"{reply}"

Which meaning does the caller's reply to THIS question have? Use earlier turns only to interpret
references like "like I said". Ignore extra information the caller volunteers about other topics."""
        return self._call(prompt, [s["id"] for s in scenarios] + ["unclear"], "classify_reply")

    def identify_message(self, flow, message):
        options = {}
        for key, q in flow.questions.items():
            options[f"question:{key}"] = f'asks: "{q["text"]}"'
        for key, o in flow.outcomes.items():
            options[f"outcome:{key}"] = f'closes with: "{o.get("say", o.get("label", key))}"'
        for key, say in flow.completions.items():
            options[f"complete:{key}"] = f'says: "{say}" (starts the renewal)'
        listing = "\n".join(f'- "{k}": {v}' for k, v in options.items())
        prompt = f"""A scripted phone agent said:

"{message}"

Which scripted step is this, possibly paraphrased? If it is none of them, answer "none".
{listing}"""
        answer, reason = self._call(prompt, list(options) + ["none"], "identify_step")
        if answer == "none":
            return None, reason
        kind, key = answer.split(":", 1)
        return (kind, key), reason


# ---------------------------------------------------------------- per-thread check

def check_thread(thread, flow, judge):
    msgs = thread["messages"]
    result = {"thread_id": thread["thread_id"], "steps": [], "notes": []}

    def identify(text):
        node = flow.identify(text)
        if node is not None:
            return node, "text match"
        node, reason = judge.identify_message(flow, text)
        return node, f"LLM: {reason}"

    if not msgs or msgs[0]["role"] != "agent":
        result["status"] = "ERROR"
        result["notes"].append("thread does not start with an agent message")
        return result

    current, how = identify(msgs[0]["text"])
    first = ("question", flow.order[0])
    if current != first:
        result["notes"].append(f"opener was {flow.describe(current)}, expected {flow.describe(first)}")

    transcript = [f'Agent: {msgs[0]["text"]}']
    i = 1
    while i < len(msgs):
        if msgs[i]["role"] != "caller":
            transcript.append(f'Agent: {msgs[i]["text"]}')
            i += 1
            continue
        caller = msgs[i]["text"]
        agent = msgs[i + 1]["text"] if i + 1 < len(msgs) and msgs[i + 1]["role"] == "agent" else None
        transcript.append(f"Caller: {caller}")

        if current is None:
            result["notes"].append("lost track of the flow (unrecognised agent message); remaining turns not checked")
            break
        if current[0] != "question":
            remaining = sum(1 for m in msgs[i:] if m["role"] == "caller")
            result["notes"].append(f"{remaining} caller turn(s) after the flow ended were not checked")
            break

        qkey = current[1]
        question = flow.questions[qkey]
        scenarios = question.get("scenarios", [])
        if len(scenarios) == 1:
            scenario, reason = scenarios[0]["id"], "only one branch"
        else:
            scenario, reason = judge.classify_reply(question, "\n".join(transcript[:-1]), caller)

        step = {
            "question": flow.labels[qkey],
            "caller": caller,
            "caller_meaning": scenario,
            "meaning_reason": reason,
        }

        if agent is None:
            step.update(expected=None, actual=None, ok=None,
                        verdict="not checked: thread ended before the agent replied")
            result["steps"].append(step)
            break

        actual, how = identify(agent)
        step["agent"] = agent
        step["actual"] = flow.describe(actual)
        step["matched_by"] = how

        if scenario == "unclear":
            step["expected"] = f"re-ask {flow.labels[qkey]}"
            step["ok"] = actual == current
            step["verdict"] = ("ok: re-asked an unclear answer" if step["ok"]
                               else "moved on although the answer was unclear")
        else:
            expected = flow.expected_next(qkey, scenario)
            step["expected"] = flow.describe(expected)
            if actual == current:
                step["ok"] = False
                step["verdict"] = "re-asked although the answer was clear"
            else:
                step["ok"] = same_node(expected, actual)
                step["verdict"] = "ok" if step["ok"] else "wrong branch"

        result["steps"].append(step)
        transcript.append(f"Agent: {agent}")
        current = actual
        i += 2

    checked = [s for s in result["steps"] if s["ok"] is not None]
    if any(s["ok"] is False for s in checked):
        result["status"] = "FAIL"
    elif current is not None and current[0] == "question":
        result["status"] = "INCOMPLETE"  # ended mid-flow; every step so far was right
    else:
        result["status"] = "PASS"
    return result


# ---------------------------------------------------------------- CLI

def print_report(results):
    for r in results:
        print(f'\n{r["thread_id"]}  {r["status"]}')
        for s in r["steps"]:
            mark = {True: "✓", False: "✗", None: "·"}[s["ok"]]
            caller = s["caller"] if len(s["caller"]) <= 70 else s["caller"][:67] + "..."
            print(f'  {mark} {s["question"]}: caller "{caller}" -> {s["caller_meaning"]}')
            if s.get("expected"):
                print(f'      expected: {s["expected"]} | agent: {s["actual"]} | {s["verdict"]}')
            else:
                print(f'      {s["verdict"]}')
        for n in r["notes"]:
            print(f"  note: {n}")
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\nSummary: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threads", nargs="*", help="only check these thread_ids")
    ap.add_argument("--json", help="also write the full results to this file")
    args = ap.parse_args()

    with open(args.file) as f:
        data = json.load(f)
    flow = Flow(data["agent_config"]["engine_config"])
    judge = Judge(args.model)

    threads = data["threads"]
    if args.threads:
        threads = [t for t in threads if t["thread_id"] in set(args.threads)]

    results = [check_thread(t, flow, judge) for t in threads]
    print_report(results)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
    sys.exit(1 if any(r["status"] == "FAIL" for r in results) else 0)


if __name__ == "__main__":
    main()
