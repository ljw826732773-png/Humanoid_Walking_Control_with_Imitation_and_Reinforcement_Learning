import argparse
import subprocess
import sys


def run(command):
    print(">", " ".join(command))
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Portfolio helper commands for the humanoid walking project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate", help="Check that the committed portfolio artifacts are complete.")
    subparsers.add_parser("report", help="Regenerate Markdown and SVG report artifacts from CSV files.")
    subparsers.add_parser("keyframes", help="Regenerate the keyframe contact sheet from the final demo video.")
    subparsers.add_parser("index", help="Regenerate RESULTS.md from committed artifacts.")
    subparsers.add_parser("showcase", help="Regenerate report, keyframes, and artifact validation.")

    args = parser.parse_args()
    py = sys.executable

    if args.command == "validate":
        run([py, "portfolio_validate_artifacts.py"])
    elif args.command == "report":
        run([py, "portfolio_generate_report.py"])
    elif args.command == "keyframes":
        run([py, "portfolio_extract_keyframes.py"])
    elif args.command == "index":
        run([py, "portfolio_generate_results_index.py"])
    elif args.command == "showcase":
        run([py, "portfolio_extract_keyframes.py"])
        run([py, "portfolio_validate_artifacts.py"])
        run([py, "portfolio_generate_report.py"])
        run([py, "portfolio_generate_results_index.py"])


if __name__ == "__main__":
    main()
