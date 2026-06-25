import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


REQUIRED_FILES = [
    "README.md",
    "requirements_portfolio.txt",
    "portfolio_train_bc.py",
    "portfolio_train_bc_enhanced.py",
    "portfolio_evaluate.py",
    "portfolio_stability_diagnostics.py",
    "portfolio_robustness_sweep.py",
    "portfolio_generate_report.py",
    "portfolio_extract_keyframes.py",
    "portfolio_validate_artifacts.py",
    "portfolio_retrain_bc_improved/bc_improved_best.pth",
    "portfolio_retrain_bc_improved/bc_improved_normalizer.npz",
    "portfolio_retrain_bc_improved/bc_improved_eval.csv",
    "portfolio_retrain_bc_improved/bc_improved_summary.csv",
    "portfolio_retrain_bc_improved/stability_summary.csv",
    "portfolio_retrain_bc_improved/robustness_summary.csv",
    "portfolio_retrain_bc_improved/portfolio_summary.md",
    "portfolio_retrain_bc_improved/bc_improved_demo.mp4",
    "portfolio_retrain_bc_improved/keyframes/bc_improved_keyframes.png",
]


def read_first_row(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return rows[0]


def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row, key):
    return float(row[key])


def add_file_checks(checks):
    for path in REQUIRED_FILES:
        checks.append(Check(f"file:{path}", os.path.exists(path), "exists" if os.path.exists(path) else "missing"))


def add_metric_checks(checks, min_avg_steps, min_best_steps):
    summary_path = "portfolio_retrain_bc_improved/bc_improved_summary.csv"
    if not os.path.exists(summary_path):
        checks.append(Check("main_summary_metrics", False, "bc_improved_summary.csv missing"))
        return

    row = read_first_row(summary_path)
    avg_steps = as_float(row, "avg_steps")
    best_steps = as_float(row, "best_steps")
    video_path = row.get("video_path", "")
    checks.append(
        Check(
            "main_avg_steps",
            avg_steps >= min_avg_steps,
            f"avg_steps={avg_steps:.1f}, threshold={min_avg_steps:.1f}",
        )
    )
    checks.append(
        Check(
            "main_best_steps",
            best_steps >= min_best_steps,
            f"best_steps={best_steps:.0f}, threshold={min_best_steps:.0f}",
        )
    )
    checks.append(
        Check(
            "main_video_path",
            os.path.exists(video_path),
            f"video_path={video_path}",
        )
    )

    stability_path = "portfolio_retrain_bc_improved/stability_summary.csv"
    if os.path.exists(stability_path):
        stability = read_first_row(stability_path)
        checks.append(
            Check(
                "stability_avg_steps",
                as_float(stability, "avg_steps") >= 700,
                f"avg_steps={as_float(stability, 'avg_steps'):.1f}, threshold=700.0",
            )
        )

    robustness_path = "portfolio_retrain_bc_improved/robustness_summary.csv"
    if os.path.exists(robustness_path):
        rows = read_rows(robustness_path)
        raw = next((row for row in rows if row.get("smoothing") == "0.0"), None)
        if raw is None:
            checks.append(Check("robustness_raw_action_row", False, "smoothing=0.0 row missing"))
        else:
            checks.append(
                Check(
                    "robustness_raw_avg_steps",
                    as_float(raw, "avg_steps") >= 800,
                    f"avg_steps={as_float(raw, 'avg_steps'):.1f}, threshold=800.0",
                )
            )


def add_video_checks(checks):
    video_path = "portfolio_retrain_bc_improved/bc_improved_demo.mp4"
    if not os.path.exists(video_path):
        checks.append(Check("video_metadata", False, "video missing"))
        return

    try:
        import cv2
    except Exception as exc:
        checks.append(Check("video_metadata", False, f"opencv unavailable: {exc}"))
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        checks.append(Check("video_metadata", False, "cannot open video"))
        return

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = frames / fps if fps > 0 else 0.0
    ok = frames >= 8000 and duration >= 250 and width >= 640 and height >= 480
    checks.append(
        Check(
            "video_metadata",
            ok,
            f"frames={frames}, fps={fps:.2f}, duration={duration:.1f}s, size={width}x{height}",
        )
    )


def write_outputs(checks, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "artifact_validation.json")
    csv_path = os.path.join(output_dir, "artifact_validation.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(check) for check in checks], f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "ok", "detail"])
        writer.writeheader()
        for check in checks:
            writer.writerow(asdict(check))

    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="portfolio_retrain_bc_improved")
    parser.add_argument("--min-avg-steps", type=float, default=800.0)
    parser.add_argument("--min-best-steps", type=float, default=1000.0)
    args = parser.parse_args()

    checks = []
    add_file_checks(checks)
    add_metric_checks(checks, args.min_avg_steps, args.min_best_steps)
    add_video_checks(checks)
    json_path, csv_path = write_outputs(checks, args.output_dir)

    failed = [check for check in checks if not check.ok]
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        print(f"{status} {check.name}: {check.detail}")
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
