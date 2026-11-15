import csv
from collections import defaultdict

def _to_int(value):
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None

def read_optc_events(path):
    events = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            ts = _to_int(row[0])
            src = str(row[1]).strip()
            dst = str(row[2]).strip()
            if ts is None or src == "" or dst == "":
                continue
            events.append({"ts": ts, "src": src, "dst": dst, "row": row})
    return events

def _components(events):
    adj = defaultdict(set)
    nodes = set()
    for event in events:
        src = event["src"]
        dst = event["dst"]
        nodes.add(src)
        nodes.add(dst)
        adj[src].add(dst)
        adj[dst].add(src)
    comps = []
    seen = set()
    for start in sorted(nodes):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = set()
        while stack:
            node = stack.pop()
            comp.add(node)
            for nxt in adj.get(node, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    comps.sort(key=lambda c: (-len(c), sorted(c)))
    return comps

def _component_edge_count(events, node_set):
    node_set = set(node_set)
    edges = {(e["src"], e["dst"]) for e in events if e["src"] in node_set and e["dst"] in node_set}
    return len(edges)

def component_stats(events):
    stats = []
    for comp in _components(events):
        stats.append(
            {
                "nodes": len(comp),
                "edges": _component_edge_count(events, comp),
                "members": sorted(comp),
            }
        )
    return stats

def select_component_nodes(events, target_nodes=8):
    comps = _components(events)
    target = int(target_nodes)
    candidates = [comp for comp in comps if len(comp) == target]
    if not candidates:
        stats = [{"nodes": len(comp), "edges": _component_edge_count(events, comp)} for comp in comps]
        raise ValueError(f"no component with {target} nodes, components={stats}")
    candidates.sort(key=lambda c: (-_component_edge_count(events, c), sorted(c)))
    return set(candidates[0])

def filter_events_by_nodes(events, nodes):
    keep = set(nodes)
    return [event for event in events if event["src"] in keep and event["dst"] in keep]

def build_anomaly_index(events):
    out = {}
    for event in events:
        key = (event["src"], event["dst"])
        if key not in out:
            out[key] = set()
        out[key].add(int(event["ts"]))
    return out

def write_events(path, events):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for event in events:
            writer.writerow(event["row"])

def load_small_optc_component(redteam_path, target_nodes=8):
    all_events = read_optc_events(redteam_path)
    nodes = select_component_nodes(all_events, target_nodes=target_nodes)
    selected = filter_events_by_nodes(all_events, nodes)
    unique_edges = {(event["src"], event["dst"]) for event in selected}
    return {
        "all_events": all_events,
        "selected_nodes": sorted(nodes),
        "selected_events": selected,
        "selected_unique_edges": sorted(unique_edges),
        "component_stats": component_stats(all_events),
    }
