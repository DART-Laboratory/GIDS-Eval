from __future__ import annotations

import csv
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path
from typing import Any

from gids_eval.paths import get_paths

RAW_DATA_ROOT = get_paths().raw_data_root
LANL_CAMPAIGN_WINDOW_SECONDS = 30 * 60
OPTC_CAMPAIGN_WINDOW_SECONDS = 5 * 60

def _to_int(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None

def _assign_temporal_entity_campaigns(
    events: list[dict[str, Any]],
    *,
    dataset: str,
    entity_keys: tuple[str, ...],
    time_window_seconds: int,
) -> list[str]:
    if not events:
        return []

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
        return []

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
    return [f"{dataset}_campaign_{ordinal:03d}" for ordinal, _ in enumerate(ordered_roots, start=1)]

def _read_optc_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3:
                continue
            ts = _to_int(row[0])
            src = str(row[1]).strip()
            dst = str(row[2]).strip()
            if ts is None or src == "" or dst == "":
                continue
            events.append({"ts": ts, "src": src, "dst": dst})
    return events

def _select_optc_component_nodes(events: list[dict[str, Any]], *, target_nodes: int = 8) -> set[str]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    nodes: set[str] = set()
    for event in events:
        src = str(event["src"])
        dst = str(event["dst"])
        nodes.add(src)
        nodes.add(dst)
        adjacency[src].add(dst)
        adjacency[dst].add(src)

    components: list[set[str]] = []
    seen: set[str] = set()
    for start in sorted(nodes):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: set[str] = set()
        while stack:
            node = stack.pop()
            component.add(node)
            for nxt in adjacency.get(node, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(component)

    candidates = [component for component in components if len(component) == int(target_nodes)]
    if not candidates:
        raise ValueError("No OpTC component matched the expected EG1 8-node subgraph")

    def edge_count(node_set: set[str]) -> int:
        allowed = set(node_set)
        return len({(event["src"], event["dst"]) for event in events if event["src"] in allowed and event["dst"] in allowed})

    candidates.sort(key=lambda node_set: (-edge_count(node_set), sorted(node_set)))
    return set(candidates[0])

@lru_cache(maxsize=16)
def reference_campaigns_for_dataset(dataset: str) -> tuple[str, ...]:
    dataset_key = str(dataset).strip().lower()
    if dataset_key == "0501":
        return ("oilrig", "sandworm", "wizard_spider")
    if dataset_key == "cic":
        return _reference_campaigns_for_cic()
    if dataset_key == "lanl":
        return _reference_campaigns_for_lanl()
    if dataset_key == "optc":
        return _reference_campaigns_for_optc()
    return ()

def _reference_campaigns_for_cic() -> tuple[str, ...]:
    values: set[str] = set()
    root = RAW_DATA_ROOT / "CIC-new"
    for day in ("monday", "tuesday", "wednesday", "thursday", "friday"):
        with (root / f"{day}.csv").open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label = str(row.get("Label", "")).strip()
                if label and label != "BENIGN":
                    values.add(label)
    return tuple(sorted(values))

def _reference_campaigns_for_lanl() -> tuple[str, ...]:
    events: list[dict[str, Any]] = []
    path = RAW_DATA_ROOT / "LANL" / "redteam.txt"
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
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
    return tuple(
        _assign_temporal_entity_campaigns(
            events,
            dataset="lanl",
            entity_keys=("src_computer",),
            time_window_seconds=LANL_CAMPAIGN_WINDOW_SECONDS,
        )
    )

def _reference_campaigns_for_optc() -> tuple[str, ...]:
    all_events = _read_optc_events(RAW_DATA_ROOT / "OpTC" / "redteam_optc.txt")
    nodes = _select_optc_component_nodes(all_events, target_nodes=8)
    selected = [event for event in all_events if event["src"] in nodes and event["dst"] in nodes]
    normalized = [
        {
            "timestamp": int(event["ts"]),
            "src_computer": str(event["src"]),
            "dst_computer": str(event["dst"]),
        }
        for event in selected
    ]
    return tuple(
        _assign_temporal_entity_campaigns(
            normalized,
            dataset="optc",
            entity_keys=("src_computer",),
            time_window_seconds=OPTC_CAMPAIGN_WINDOW_SECONDS,
        )
    )
