"""Summarize oracle collection logs."""
import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime

LOG_RE = re.compile(r"\[oracle (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (.+)")
MEASURING_RE = re.compile(r"measuring event (\d+)/(\d+).*subsets=(\d+)")
BATCH_DONE_RE = re.compile(r"batch (\d+)/(\d+) done.*total_events=(\d+)")
SUMMARY_RE = re.compile(r"Shard summary: (.+)")


def summarize_log(log_path):
    events_started = 0
    total_subsets = 0
    batches_done = 0
    total_events = 0
    first_ts = None
    last_ts = None
    summary_data = None
    for line in Path(log_path).read_text().splitlines():
        m = LOG_RE.match(line)
        if not m:
            continue
        ts_str, msg = m.groups()
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        if first_ts is None:
            first_ts = ts
        last_ts = ts
        m2 = MEASURING_RE.search(msg)
        if m2:
            events_started += 1
            total_subsets += int(m2.group(3))
        m3 = BATCH_DONE_RE.search(msg)
        if m3:
            batches_done += 1
            total_events = max(total_events, int(m3.group(3)))
        m4 = SUMMARY_RE.search(line)
        if m4:
            try:
                summary_data = json.loads(m4.group(1))
            except json.JSONDecodeError:
                pass
    elapsed_sec = (last_ts - first_ts).total_seconds() if first_ts and last_ts else 0
    result = {
        "log_path": str(log_path),
        "elapsed_sec": elapsed_sec,
        "events_seen": events_started,
        "avg_subsets_per_event": total_subsets / max(events_started, 1),
        "batches_done": batches_done,
        "total_events_at_end": total_events,
        "events_per_hour": events_started / max(elapsed_sec / 3600, 1e-6),
    }
    if summary_data:
        result["shard_summary"] = summary_data
    return result


def main():
    parser = argparse.ArgumentParser(description="Summarize oracle collection logs")
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    total_events = 0
    total_elapsed = 0.0
    for log in args.logs:
        s = summarize_log(log)
        total_events += s["events_seen"]
        total_elapsed += s["elapsed_sec"]
        print(f"{log.name}: {s['events_seen']} events, "
              f"avg {s['avg_subsets_per_event']:.1f} subsets/event, "
              f"{s['events_per_hour']:.1f} events/hour, "
              f"{s['elapsed_sec']/60:.1f} min")
        if "shard_summary" in s:
            ss = s["shard_summary"]
            print(f"  event types: {ss.get('event_type_counts', {})}")
            print(f"  sequences: {ss.get('num_sequences', '?')}, datasets: {ss.get('dataset_counts', {})}")
            print(f"  subsets/event: {ss.get('subsets_per_event', {})}")
    if len(args.logs) > 1:
        print(f"\n--- Total: {total_events} events, {total_elapsed/3600:.1f}h ---")


if __name__ == "__main__":
    main()
