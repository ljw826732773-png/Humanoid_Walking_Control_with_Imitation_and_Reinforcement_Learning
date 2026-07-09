import argparse
import csv
import html
import os


FIGURES = [
    (
        "Long-horizon walking keyframes",
        "portfolio_retrain_bc_improved/keyframes/bc_improved_keyframes.png",
        "Eight frames sampled from the final demo video, covering 14.0s to 265.7s.",
    ),
    (
        "Closed-loop performance comparison",
        "portfolio_retrain_bc_improved/report_steps_comparison.svg",
        "Average episode length improves from the original BC retrain to the rollout-selected enhanced BC.",
    ),
    (
        "Enhanced BC training loss",
        "portfolio_retrain_bc_improved/report_training_loss.svg",
        "Training and validation MSE curves for the normalized behavior cloning policy.",
    ),
    (
        "Rollout-based checkpoint selection",
        "portfolio_retrain_bc_improved/report_rollout_selection.svg",
        "MuJoCo closed-loop rollout is used to select the checkpoint, instead of relying only on validation MSE.",
    ),
    (
        "Episode stability diagnostics",
        "portfolio_retrain_bc_improved/report_stability_diagnostics.svg",
        "Episode steps and action-delta diagnostics help identify whether the controller is stable or only lucky.",
    ),
    (
        "Robustness and action smoothing sweep",
        "portfolio_retrain_bc_improved/report_robustness_sweep.svg",
        "Naive deployment-time action smoothing reduces performance for this learned gait.",
    ),
]


def read_first_row(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def svg_text(x, y, text, klass="label", anchor="start"):
    return f'<text class="{klass}" x="{x}" y="{y}" text-anchor="{anchor}">{html.escape(text)}</text>'


def write_overview_dashboard(output_dir):
    main = read_first_row("portfolio_retrain_bc_improved/bc_improved_summary.csv")
    stability = read_first_row("portfolio_retrain_bc_improved/stability_summary.csv")
    robustness_rows = read_rows("portfolio_retrain_bc_improved/robustness_summary.csv")
    raw = next((row for row in robustness_rows if row.get("smoothing") == "0.0"), {})

    metrics = [
        ("Original BC avg steps", "97.5"),
        ("Enhanced BC avg steps", f"{as_float(main, 'avg_steps'):.1f}"),
        ("Enhanced BC best steps", f"{as_float(main, 'best_steps'):.0f}"),
        ("Stability avg steps", f"{as_float(stability, 'avg_steps'):.1f}"),
        ("Robustness avg steps", f"{as_float(raw, 'avg_steps'):.1f}"),
        ("Demo video", "279.7s"),
    ]
    pipeline = [
        "Expert Dataset",
        "State/Action Normalization",
        "Behavior Cloning",
        "MuJoCo Rollout Selection",
        "Diagnostics + Report",
    ]

    width, height = 1160, 620
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}",
        ".title{font-size:28px;font-weight:700}",
        ".subtitle{font-size:15px;fill:#52606d}",
        ".metric{font-size:30px;font-weight:700;fill:#102a43}",
        ".label{font-size:14px;fill:#52606d}",
        ".step{font-size:13px;font-weight:700;fill:#102a43}",
        ".note{font-size:12px;fill:#627d98}",
        "</style>",
        '<rect width="1160" height="620" fill="#f8fafc"/>',
        '<rect x="24" y="24" width="1112" height="572" rx="18" fill="#ffffff" stroke="#d9e2ec"/>',
        svg_text(52, 70, "Humanoid Walking Control Portfolio Dashboard", "title"),
        svg_text(52, 96, "Enhanced behavior cloning with normalization, closed-loop checkpoint selection, diagnostics, and reproducible artifacts.", "subtitle"),
    ]

    card_w, card_h = 330, 92
    start_x, start_y = 52, 132
    gap_x, gap_y = 28, 24
    for idx, (label, value) in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        x = start_x + col * (card_w + gap_x)
        y = start_y + row * (card_h + gap_y)
        accent = "#2f80ed" if idx in (1, 2, 4) else "#8a8f98"
        lines.append(f'<rect x="{x}" y="{y}" width="{card_w}" height="{card_h}" rx="12" fill="#f0f4f8" stroke="#d9e2ec"/>')
        lines.append(f'<rect x="{x}" y="{y}" width="6" height="{card_h}" rx="3" fill="{accent}"/>')
        lines.append(svg_text(x + 22, y + 34, label, "label"))
        lines.append(svg_text(x + 22, y + 70, value, "metric"))

    lines.append(svg_text(52, 392, "Experiment Pipeline", "title"))
    step_w, step_h = 188, 72
    step_y = 430
    for idx, label in enumerate(pipeline):
        x = 52 + idx * 214
        lines.append(f'<rect x="{x}" y="{step_y}" width="{step_w}" height="{step_h}" rx="14" fill="#e6f6ff" stroke="#9ed9f7"/>')
        lines.append(svg_text(x + step_w / 2, step_y + 42, label, "step", "middle"))
        if idx < len(pipeline) - 1:
            ax = x + step_w + 8
            ay = step_y + step_h / 2
            lines.append(f'<line x1="{ax}" y1="{ay}" x2="{ax + 34}" y2="{ay}" stroke="#627d98" stroke-width="2"/>')
            lines.append(f'<polygon points="{ax + 34},{ay} {ax + 25},{ay - 6} {ax + 25},{ay + 6}" fill="#627d98"/>')

    lines.append(svg_text(52, 545, "Boundary: this is an improved imitation-learning controller for simulated humanoid walking, not a real-robot safety controller.", "note"))
    lines.append("</svg>")

    path = os.path.join(output_dir, "report_overview_dashboard.svg")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def write_gallery(output):
    lines = [
        "# Visual Gallery",
        "",
        "This gallery collects the visual evidence used to explain the humanoid walking control project.",
        "",
        "## Overview Dashboard",
        "",
        "![Overview dashboard](portfolio_retrain_bc_improved/report_overview_dashboard.svg)",
        "",
    ]
    for title, path, note in FIGURES:
        lines.extend(
            [
                f"## {title}",
                "",
                f"![{title}]({path})",
                "",
                note,
                "",
            ]
        )
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="VISUAL_GALLERY.md")
    parser.add_argument("--output-dir", default="portfolio_retrain_bc_improved")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dashboard = write_overview_dashboard(args.output_dir)
    gallery = write_gallery(args.output)
    print(f"wrote {dashboard}")
    print(f"wrote {gallery}")


if __name__ == "__main__":
    main()
