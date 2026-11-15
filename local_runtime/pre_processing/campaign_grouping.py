from __future__ import annotations

import csv
from collections import defaultdict, deque
from typing import Any

LANL_CAMPAIGN_WINDOW_SECONDS = 30 * 60
OPTC_CAMPAIGN_WINDOW_SECONDS = 5 * 60

def _to_int(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None

def assign_temporal_entity_campaigns(
    events: list[dict[str, Any]],
    *,
    dataset: str,
    entity_keys: tuple[str, ...],
    time_window_seconds: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    if not events:
        return [], []

    prepared: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        timestamp = _to_int(event.get("timestamp", event.get("ts")))
        if timestamp is None:
            continue
        normalized = dict(event)
        normalized["_orig_index"] = index
        normalized["timestamp"] = timestamp
        prepared.append(normalized)

    if not prepared:
        return [""] * len(events), []

    prepared.sort(key=lambda event: (int(event["timestamp"]), int(event["_orig_index"])))
    parent = list(range(len(prepared)))
    rank = [0] * len(prepared)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            parent[left_root] = right_root
        elif rank[left_root] > rank[right_root]:
            parent[right_root] = left_root
        else:
            parent[right_root] = left_root
            rank[left_root] += 1

    recent_by_entity = {
        entity_key: defaultdict(deque)
        for entity_key in entity_keys
    }

    for event_index, event in enumerate(prepared):
        timestamp = int(event["timestamp"])
        for entity_key in entity_keys:
            entity_value = str(event.get(entity_key, "")).strip()
            if entity_value == "":
                continue
            recent = recent_by_entity[entity_key][entity_value]
            while recent and timestamp - int(prepared[recent[0]]["timestamp"]) > time_window_seconds:
                recent.popleft()
            for previous_index in recent:
                union(event_index, previous_index)
            recent.append(event_index)

    groups: dict[int, list[int]] = defaultdict(list)
    for event_index in range(len(prepared)):
        groups[find(event_index)].append(event_index)

    ordered_roots = sorted(
        groups,
        key=lambda root: (
            min(int(prepared[event_index]["timestamp"]) for event_index in groups[root]),
            -len(groups[root]),
        ),
    )

    labels = [""] * len(events)
    stats: list[dict[str, Any]] = []
    for ordinal, root in enumerate(ordered_roots, start=1):
        group_indices = groups[root]
        campaign = f"{dataset}_campaign_{ordinal:03d}"
        timestamps = [int(prepared[event_index]["timestamp"]) for event_index in group_indices]
        entity_summary = {
            entity_key: sorted(
                {
                    str(prepared[event_index].get(entity_key, "")).strip()
                    for event_index in group_indices
                    if str(prepared[event_index].get(entity_key, "")).strip()
                }
            )
            for entity_key in entity_keys
        }
        for event_index in group_indices:
            labels[int(prepared[event_index]["_orig_index"])] = campaign
        stats.append(
            {
                "campaign": campaign,
                "event_count": len(group_indices),
                "start_time": min(timestamps),
                "end_time": max(timestamps),
                "duration_seconds": max(timestamps) - min(timestamps),
                "entities": entity_summary,
            }
        )
    return labels, stats

def build_lanl_campaign_lookup(
    redteam_path: str,
    *,
    time_window_seconds: int = LANL_CAMPAIGN_WINDOW_SECONDS,
) -> tuple[dict[tuple[int, str, str, str], str], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    with open(redteam_path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 4:
                continue
            timestamp = _to_int(row[0])
            if timestamp is None:
                continue
            src_user = str(row[1]).strip()
            src_computer = str(row[2]).strip()
            dst_computer = str(row[3]).strip()
            if src_user == "" or src_computer == "" or dst_computer == "":
                continue
            events.append(
                {
                    "timestamp": timestamp,
                    "src_user": src_user,
                    "src_computer": src_computer,
                    "dst_computer": dst_computer,
                }
            )
    campaigns, stats = assign_temporal_entity_campaigns(
        events,
        dataset="lanl",
        entity_keys=("src_computer",),
        time_window_seconds=time_window_seconds,
    )
    lookup = {
        (
            int(event["timestamp"]),
            str(event["src_user"]),
            str(event["src_computer"]),
            str(event["dst_computer"]),
        ): campaign
        for event, campaign in zip(events, campaigns)
    }
    return lookup, stats

def build_optc_campaign_lookup(
    events: list[dict[str, Any]],
    *,
    time_window_seconds: int = OPTC_CAMPAIGN_WINDOW_SECONDS,
) -> tuple[dict[tuple[int, str, str], str], list[dict[str, Any]]]:
    normalized: list[dict[str, Any]] = []
    for event in events:
        timestamp = _to_int(event.get("timestamp", event.get("ts")))
        src = str(event.get("src", event.get("src_computer", ""))).strip()
        dst = str(event.get("dst", event.get("dst_computer", ""))).strip()
        if timestamp is None or src == "" or dst == "":
            continue
        normalized.append(
            {
                "timestamp": timestamp,
                "src_computer": src,
                "dst_computer": dst,
            }
        )
    campaigns, stats = assign_temporal_entity_campaigns(
        normalized,
        dataset="optc",
        entity_keys=("src_computer",),
        time_window_seconds=time_window_seconds,
    )
    lookup = {
        (
            int(event["timestamp"]),
            str(event["src_computer"]),
            str(event["dst_computer"]),
        ): campaign
        for event, campaign in zip(normalized, campaigns)
    }
    return lookup, stats
