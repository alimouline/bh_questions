#!/usr/bin/env python3
"""Browse call transcripts thread by thread.

Reads the format of logs/gym_agent_conversations.json: a top-level "threads"
list, each with a "messages" list of {turn, role, text}.

Usage: python utils/browse_calls.py path/to/conversations.json
Press Enter to advance to the next thread, 'q' to quit.
"""
import json
import sys
from itertools import groupby
from pathlib import Path


def load_threads(path):
    with open(path) as f:
        data = json.load(f)
    return data["threads"]


def print_thread(thread, index, total):
    print("=" * 80)
    print(f"[{index}/{total}] thread_id: {thread.get('thread_id')}")
    print("=" * 80)
    for turn, messages in groupby(thread.get("messages", []), key=lambda m: m.get("turn")):
        print(f"\n--- Turn {turn} ---")
        for msg in messages:
            role = msg.get("role", "?").upper()
            print(f"{role + ':':8}{msg.get('text', '')}")
    print()


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <conversations.json>")
    path = Path(sys.argv[1])
    threads = load_threads(path)
    total = len(threads)
    for i, thread in enumerate(threads, 1):
        print_thread(thread, i, total)
        if i < total:
            try:
                resp = input("-- Enter for next thread, 'q' to quit -- ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if resp.strip().lower() == "q":
                break


if __name__ == "__main__":
    main()
