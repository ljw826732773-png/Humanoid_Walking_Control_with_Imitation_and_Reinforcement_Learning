import argparse
import csv
import html
import os


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def short_label(label, max_len=24):
    return label if len(label) <= max_len else label[: max_len - 3] + "..."


def svg_header(width, height, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#222}",
        ".title{font-size:22px;font-weight:700}",
        ".axis{font-size:12px;fill:#555}",
        ".label{font-size:11px;fill:#333}",
        ".grid{stroke:#ddd;stroke-width:1}",
        ".line{fill:none;stroke-width:2.5}",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text class="title" x="28" y="34">{html.escape(title)}</text>',
    ]


def svg_footer():
    return ["</svg>"]


def write_svg(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def save_bar_chart(path, title, labels, values, ylabel, highlight=None):
    if not labels:
        return None
    width, height = 980, 500
    left, right, top, bottom = 70, 24, 70, 130
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_v = max(max(values), 1.0)
    max_axis = max_v * 1.12
    bar_gap = 10
    bar_w = max(10, (plot_w - bar_gap * (len(values) + 1)) / len(values))

    lines = svg_header(width, height, title)
    for i in range(6):
        y = top + plot_h - plot_h * i / 5
        value = max_axis * i / 5
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>')
        lines.append(f'<text class="axis" x="14" y="{y + 4:.1f}">{value:.0f}</text>')
    lines.append(f'<text class="axis" x="20" y="{top - 18}">{html.escape(ylabel)}</text>')

    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + bar_gap + index * (bar_w + bar_gap)
        bar_h = plot_h * value / max_axis
        y = top + plot_h - bar_h
        color = "#1f77b4" if highlight is not None and highlight(label) else "#8a8f98"
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        lines.append(f'<text class="label" x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle">{value:.1f}</text>')
        safe_label = html.escape(short_label(label))
        label_x = x + bar_w / 2
        label_y = top + plot_h + 18
        lines.append(
            f'<text class="label" transform="translate({label_x:.1f},{label_y:.1f}) rotate(35)" '
            f'text-anchor="start">{safe_label}</text>'
        )
    lines.extend(svg_footer())
    return write_svg(path, lines)


def save_line_chart(path, title, x_values, series, xlabel, ylabel):
    if not x_values or not series:
        return None
    width, height = 900, 460
    left, right, top, bottom = 70, 28, 68, 62
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_y = [value for _, values, _ in series for value in values]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(all_y), max(all_y)
    if min_y == max_y:
        max_y += 1
    min_y = min(0, min_y)
    max_y *= 1.08

    def sx(x):
        if min_x == max_x:
            return left + plot_w / 2
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y):
        return top + plot_h - (y - min_y) / (max_y - min_y) * plot_h

    lines = svg_header(width, height, title)
    for i in range(6):
        y = top + plot_h - plot_h * i / 5
        value = min_y + (max_y - min_y) * i / 5
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>')
        lines.append(f'<text class="axis" x="14" y="{y + 4:.1f}">{value:.3g}</text>')
    lines.append(f'<text class="axis" x="{width / 2:.1f}" y="{height - 16}" text-anchor="middle">{html.escape(xlabel)}</text>')
    lines.append(f'<text class="axis" x="20" y="{top - 18}">{html.escape(ylabel)}</text>')

    legend_x = width - right - 180
    for idx, (name, y_values, color) in enumerate(series):
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(x_values, y_values))
        lines.append(f'<polyline class="line" points="{points}" stroke="{color}"/>')
        for x, y in zip(x_values, y_values):
            lines.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.2" fill="{color}"/>')
        legend_y = top + 18 + idx * 20
        lines.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="14" height="4" fill="{color}"/>')
        lines.append(f'<text class="axis" x="{legend_x + 20}" y="{legend_y - 5}">{html.escape(name)}</text>')

    lines.extend(svg_footer())
    return write_svg(path, lines)


def save_steps_comparison(rows, output_dir):
    labels = [row["experiment"] for row in rows]
    values = [as_float(row, "avg_steps") for row in rows]
    return save_bar_chart(
        os.path.join(output_dir, "report_steps_comparison.svg"),
        "Closed-loop Walking Performance",
        labels,
        values,
        "Average steps",
        highlight=lambda label: label == "rollout_selected_bc",
    )


def save_training_loss(rows, output_dir):
    epochs = [as_float(row, "epoch") for row in rows]
    return save_line_chart(
        os.path.join(output_dir, "report_training_loss.svg"),
        "Enhanced BC Training Loss",
        epochs,
        [
            ("train loss", [as_float(row, "train_loss") for row in rows], "#1f77b4"),
            ("val loss", [as_float(row, "val_loss") for row in rows], "#ff7f0e"),
        ],
        "Epoch",
        "MSE loss",
    )


def save_rollout_selection(rows, output_dir):
    epochs = [as_float(row, "epoch") for row in rows]
    return save_line_chart(
        os.path.join(output_dir, "report_rollout_selection.svg"),
        "MuJoCo Rollout-based Checkpoint Selection",
        epochs,
        [("rollout avg steps", [as_float(row, "avg_steps") for row in rows], "#2ca02c")],
        "Epoch",
        "Average steps",
    )


def save_stability(rows, output_dir):
    episodes = [as_float(row, "episode") for row in rows]
    return save_line_chart(
        os.path.join(output_dir, "report_stability_diagnostics.svg"),
        "Episode Stability Diagnostics",
        episodes,
        [
            ("steps", [as_float(row, "steps") for row in rows], "#2ca02c"),
            ("action delta x1000", [as_float(row, "avg_action_delta") * 1000 for row in rows], "#d62728"),
        ],
        "Episode",
        "Steps / scaled action delta",
    )


def save_robustness(rows, output_dir):
    labels = [row["smoothing"] for row in rows]
    values = [as_float(row, "avg_steps") for row in rows]
    return save_bar_chart(
        os.path.join(output_dir, "report_robustness_sweep.svg"),
        "Robustness and Action Smoothing Sweep",
        labels,
        values,
        "Average steps",
        highlight=lambda label: label == "0.0",
    )


def write_markdown(output_dir, data, images):
    summary = data["bc_summary"][0] if data["bc_summary"] else {}
    stability = data["stability_summary"][0] if data["stability_summary"] else {}
    robustness = data["robustness_summary"]
    best_robust = max(robustness, key=lambda row: as_float(row, "avg_steps")) if robustness else None

    lines = [
        "# Portfolio Experiment Summary",
        "",
        "This report is generated from the saved CSV artifacts in `portfolio_retrain_bc_improved/`.",
        "",
        "## Main Result",
        "",
        f"- Average steps: {as_float(summary, 'avg_steps'):.1f}",
        f"- Best steps: {as_float(summary, 'best_steps'):.0f}",
        f"- Average reward: {as_float(summary, 'avg_reward'):.2f}",
        f"- Best reward: {as_float(summary, 'best_reward'):.2f}",
        "",
        "## Stability Diagnostics",
        "",
        f"- Diagnostic average steps: {as_float(stability, 'avg_steps'):.1f}",
        f"- Average action norm: {as_float(stability, 'avg_action_norm'):.3f}",
        f"- Average action delta: {as_float(stability, 'avg_action_delta'):.3f}",
        "",
    ]

    if best_robust is not None:
        lines.extend(
            [
                "## Robustness Sweep",
                "",
                f"- Best smoothing value: {best_robust['smoothing']}",
                f"- Sweep average steps: {as_float(best_robust, 'avg_steps'):.1f}",
                f"- Sweep success rate: {as_float(best_robust, 'success_rate'):.2%}",
                "",
                "The sweep shows that naive action smoothing hurts this controller, so the final policy uses the raw normalized BC action.",
                "",
            ]
        )

    validation_rows = data.get("validation", [])
    if validation_rows:
        passed = sum(1 for row in validation_rows if row.get("ok") == "True")
        total = len(validation_rows)
        failed = total - passed
        lines.extend(
            [
                "## Artifact Validation",
                "",
                f"- Passed checks: {passed}/{total}",
                f"- Failed checks: {failed}",
                "",
            ]
        )

    lines.extend(["## Figures", ""])
    for image in images:
        if image:
            rel = os.path.basename(image)
            lines.append(f"- ![{rel}]({rel})")

    keyframe_path = os.path.join(output_dir, "keyframes", "bc_improved_keyframes.png")
    if os.path.exists(keyframe_path):
        lines.extend(
            [
                "",
                "## Video Keyframes",
                "",
                "- ![bc_improved_keyframes.png](keyframes/bc_improved_keyframes.png)",
            ]
        )

    path = os.path.join(output_dir, "portfolio_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="portfolio_retrain_bc_improved")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = {
        "comparison": read_csv(os.path.join(args.output_dir, "improvement_comparison.csv")),
        "training": read_csv(os.path.join(args.output_dir, "bc_improved_training_log.csv")),
        "rollout": read_csv(os.path.join(args.output_dir, "bc_improved_rollout_selection.csv")),
        "diagnostics": read_csv(os.path.join(args.output_dir, "stability_diagnostics.csv")),
        "stability_summary": read_csv(os.path.join(args.output_dir, "stability_summary.csv")),
        "bc_summary": read_csv(os.path.join(args.output_dir, "bc_improved_summary.csv")),
        "robustness_summary": read_csv(os.path.join(args.output_dir, "robustness_summary.csv")),
        "validation": read_csv(os.path.join(args.output_dir, "artifact_validation.csv")),
    }

    images = [
        os.path.join(args.output_dir, "report_overview_dashboard.svg")
        if os.path.exists(os.path.join(args.output_dir, "report_overview_dashboard.svg"))
        else None,
        save_steps_comparison(data["comparison"], args.output_dir),
        save_training_loss(data["training"], args.output_dir),
        save_rollout_selection(data["rollout"], args.output_dir),
        save_stability(data["diagnostics"], args.output_dir),
        save_robustness(data["robustness_summary"], args.output_dir),
    ]
    md_path = write_markdown(args.output_dir, data, images)
    print(f"wrote {md_path}")
    for image in images:
        if image:
            print(f"wrote {image}")


if __name__ == "__main__":
    main()
