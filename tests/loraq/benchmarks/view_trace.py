"""
Offline trace viewer for Triton Proton .hatchet files.

Works entirely locally — no network access, no upload.

Usage:
    python -m benchmarks.view_trace traces/loraq_128x4096x4096.hatchet
    python -m benchmarks.view_trace traces/loraq_128x4096x4096.hatchet --plot
    python -m benchmarks.view_trace traces/loraq_128x4096x4096.hatchet --csv out.csv
    python -m benchmarks.view_trace traces/loraq_128x4096x4096.hatchet --top 20
"""

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# Triton Proton JSON parser (no hatchet dependency)
# ---------------------------------------------------------------------------

def _walk(node: dict, path: str, rows: list, depth: int = 0) -> None:
    """
    Recursively walk a Triton Proton JSON tree node.

    Each node looks like:
        {
            "frame":    {"name": "...", "type": "function"},
            "metrics":  {"count": N, "time (ns)": T, ...},
            "children": [...]
        }
    """
    frame = node.get("frame", {})
    name  = frame.get("name", "<root>")
    short = name[:80] + "…" if len(name) > 80 else name  # truncate mangled names

    metrics = node.get("metrics", {})
    time_ns = metrics.get("time (ns)", 0)
    count   = metrics.get("count", 0)
    device  = metrics.get("device_type", "")

    rows.append({
        "depth":    depth,
        "name":     short,
        "full_name": name,
        "time_ns":  time_ns,
        "time_ms":  time_ns / 1e6,
        "count":    count,
        "device":   device,
        "path":     path,
    })

    for child in node.get("children", []):
        child_path = path + " > " + name if path else name
        _walk(child, child_path, rows, depth + 1)


def load_trace(path: str) -> list:
    """
    Parse a Triton Proton .hatchet file into a flat list of row dicts.
    Returns a list sorted by time_ns descending.
    """
    import json
    with open(path, "r") as f:
        data = json.load(f)

    rows: list = []
    # Top-level is a list of root nodes
    if isinstance(data, list):
        for root_node in data:
            _walk(root_node, "", rows)
    elif isinstance(data, dict):
        _walk(data, "", rows)
    else:
        print(f"ERROR: unexpected top-level type {type(data)} in {path}")
        sys.exit(1)

    return rows


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def cmd_tree(rows: list, top: int) -> None:
    """Print an ASCII indented tree (depth-first order, as stored)."""
    print(f"\n{'='*100}")
    print("  Call tree  (time in ms)")
    print(f"{'='*100}")
    print(f"  {'name':<82} {'time_ms':>10} {'count':>8}")
    print(f"  {'-'*82} {'-'*10} {'-'*8}")
    for r in rows:
        indent = "  " * r["depth"]
        label  = (indent + r["name"])[:82]
        t = f"{r['time_ms']:.3f}" if r["time_ms"] else "—"
        c = str(r["count"]) if r["count"] else "—"
        print(f"  {label:<82} {t:>10} {c:>8}")


def cmd_table(rows: list, top: int) -> None:
    """Print flat table of top N kernels by time (leaf nodes only)."""
    # Leaf nodes = kernels (no children contribution, actual GPU work)
    leaves = [r for r in rows if r["time_ns"] > 0]
    leaves.sort(key=lambda r: r["time_ns"], reverse=True)
    leaves = leaves[:top]

    total_ms = sum(r["time_ms"] for r in leaves)

    print(f"\n{'='*100}")
    print(f"  Top {top} nodes by time")
    print(f"{'='*100}")
    print(f"  {'name':<60} {'time_ms':>10} {'count':>8} {'% total':>8} {'device':>6}")
    print(f"  {'-'*60} {'-'*10} {'-'*8} {'-'*8} {'-'*6}")
    for r in leaves:
        pct = 100.0 * r["time_ms"] / total_ms if total_ms > 0 else 0.0
        print(
            f"  {r['name']:<60} {r['time_ms']:>10.3f} {r['count']:>8} "
            f"{pct:>7.1f}% {r['device']:>6}"
        )
    print(f"\n  Total time shown: {total_ms:.3f} ms")


def cmd_plot(rows: list, path: str) -> None:
    """Save a horizontal bar chart of kernel times as PNG (no display needed)."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive: no display required
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not installed. Run:  pip install matplotlib")
        sys.exit(1)

    leaves = [r for r in rows if r["time_ns"] > 0]
    leaves.sort(key=lambda r: r["time_ns"], reverse=True)
    leaves = leaves[:20]

    labels = [r["name"] for r in leaves]
    times  = [r["time_ms"] for r in leaves]

    fig, ax = plt.subplots(figsize=(14, max(4, len(leaves) * 0.45)))
    bars = ax.barh(range(len(leaves)), times, color="steelblue")
    ax.set_yticks(range(len(leaves)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("time (ms)")
    ax.set_title(f"Kernel time breakdown — {path}", fontsize=10)

    for bar, val in zip(bars, times):
        ax.text(
            bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f} ms", va="center", fontsize=7,
        )

    out_png = (
        os.path.basename(path).replace(".hatchet", "_flamegraph.png")
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"\nPlot saved: {out_png}")
    print(f"Copy to local machine:  scp server:$(pwd)/{out_png} .")


def cmd_csv(rows: list, csv_path: str) -> None:
    """Export all rows to CSV."""
    import csv as _csv
    fields = ["depth", "name", "time_ms", "count", "device", "path", "full_name"]
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"CSV saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline viewer for Triton Proton .hatchet traces"
    )
    parser.add_argument("hatchet", help="Path to .hatchet trace file")
    parser.add_argument("--top", type=int, default=30,
                        help="Number of top kernels to show in table (default: 30)")
    parser.add_argument("--plot", action="store_true",
                        help="Save a bar-chart PNG (no display needed, uses Agg backend)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Export flat dataframe to this CSV path")
    parser.add_argument("--tree", action="store_true", default=True,
                        help="Print ASCII tree (default: on)")
    parser.add_argument("--no-tree", dest="tree", action="store_false",
                        help="Skip ASCII tree output")
    args = parser.parse_args()

    print(f"\nLoading: {args.hatchet}")
    rows = load_trace(args.hatchet)
    print(f"Total nodes: {len(rows)}\n")

    if args.tree:
        cmd_tree(rows, args.top)

    cmd_table(rows, args.top)

    if args.plot:
        cmd_plot(rows, args.hatchet)

    if args.csv:
        cmd_csv(rows, args.csv)

    print("\nDone.  To view in a local browser (no upload):")
    print("  1. scp the .hatchet file to your local machine")
    print("  2. Open Chrome → chrome://tracing → Load → select the file")
    print("  (chrome://tracing reads files locally — no network request is made)")


if __name__ == "__main__":
    main()
