import csv
import os
import py_compile
import re
import sys


PYTHON_FILES = [
    "portfolio_cli.py",
    "portfolio_evaluate.py",
    "portfolio_train_bc.py",
    "portfolio_train_bc_enhanced.py",
    "portfolio_stability_diagnostics.py",
    "portfolio_robustness_sweep.py",
    "portfolio_generate_report.py",
    "portfolio_generate_results_index.py",
    "portfolio_extract_keyframes.py",
    "portfolio_validate_artifacts.py",
    "portfolio_static_checks.py",
    "portfolio_dagger_bc.py",
]

REQUIRED_FILES = [
    "README.md",
    "EXPERIMENTS.md",
    "MODEL_CARD.md",
    "RESULTS.md",
    "requirements_portfolio.txt",
    "portfolio_retrain_bc_improved/bc_improved_summary.csv",
    "portfolio_retrain_bc_improved/stability_summary.csv",
    "portfolio_retrain_bc_improved/robustness_summary.csv",
    "portfolio_retrain_bc_improved/artifact_validation.csv",
    "portfolio_retrain_bc_improved/keyframes/bc_improved_keyframes.png",
    ".github/workflows/portfolio-check.yml",
]


def fail(message):
    print(f"FAIL {message}")
    raise SystemExit(1)


def pass_msg(message):
    print(f"PASS {message}")


def read_first_row(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        fail(f"{path} has no rows")
    return rows[0]


def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row, key):
    try:
        return float(row[key])
    except KeyError:
        fail(f"missing column {key}")
    except ValueError:
        fail(f"non-numeric value for {key}: {row.get(key)}")


def check_required_files():
    for path in REQUIRED_FILES:
        if not os.path.exists(path):
            fail(f"required file missing: {path}")
    pass_msg(f"required files exist: {len(REQUIRED_FILES)}")


def check_python_syntax():
    for path in PYTHON_FILES:
        if not os.path.exists(path):
            fail(f"python file missing: {path}")
        py_compile.compile(path, doraise=True)
    pass_msg(f"python syntax valid: {len(PYTHON_FILES)} files")


def check_metrics():
    main = read_first_row("portfolio_retrain_bc_improved/bc_improved_summary.csv")
    stability = read_first_row("portfolio_retrain_bc_improved/stability_summary.csv")
    robustness_rows = read_rows("portfolio_retrain_bc_improved/robustness_summary.csv")
    validation_rows = read_rows("portfolio_retrain_bc_improved/artifact_validation.csv")

    avg_steps = as_float(main, "avg_steps")
    best_steps = as_float(main, "best_steps")
    if avg_steps < 800:
        fail(f"main avg_steps below threshold: {avg_steps}")
    if best_steps < 1000:
        fail(f"main best_steps below threshold: {best_steps}")

    stability_steps = as_float(stability, "avg_steps")
    if stability_steps < 700:
        fail(f"stability avg_steps below threshold: {stability_steps}")

    raw = next((row for row in robustness_rows if row.get("smoothing") == "0.0"), None)
    if raw is None:
        fail("robustness smoothing=0.0 row missing")
    raw_steps = as_float(raw, "avg_steps")
    if raw_steps < 800:
        fail(f"raw robustness avg_steps below threshold: {raw_steps}")

    failed_validation = [row for row in validation_rows if row.get("ok") != "True"]
    if failed_validation:
        fail(f"artifact validation contains failures: {len(failed_validation)}")

    pass_msg(
        "metrics thresholds pass: "
        f"main_avg={avg_steps:.1f}, main_best={best_steps:.0f}, "
        f"stability_avg={stability_steps:.1f}, robustness_avg={raw_steps:.1f}"
    )


def markdown_image_targets(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)


def check_markdown_image_links():
    for md_path in ["README.md", "RESULTS.md", "portfolio_retrain_bc_improved/portfolio_summary.md"]:
        base = os.path.dirname(md_path)
        for target in markdown_image_targets(md_path):
            if target.startswith(("http://", "https://")):
                continue
            resolved = os.path.normpath(os.path.join(base, target))
            if not os.path.exists(resolved):
                fail(f"broken image link in {md_path}: {target}")
    pass_msg("markdown image links resolve")


def check_result_docs_reference_limits():
    model_card = open("MODEL_CARD.md", "r", encoding="utf-8").read()
    if "Not Intended For" not in model_card or "Limitations" not in model_card:
        fail("MODEL_CARD.md must document non-intended uses and limitations")

    experiments = open("EXPERIMENTS.md", "r", encoding="utf-8").read()
    if "尚未作为主结果声称完成" not in experiments:
        fail("EXPERIMENTS.md must include result boundary wording")

    results = open("RESULTS.md", "r", encoding="utf-8").read()
    if "Artifact Inventory" not in results:
        fail("RESULTS.md must include artifact inventory")
    pass_msg("result documentation boundaries present")


def main():
    try:
        check_required_files()
        check_python_syntax()
        check_metrics()
        check_markdown_image_links()
        check_result_docs_reference_limits()
    except py_compile.PyCompileError as exc:
        print(exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
