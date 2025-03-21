#! /usr/bin/env python3
import argparse
import subprocess


def _run(cmd: list[str]) -> None:
    uv = [
        "uv",
        "run",
        "--with-requirements",
        "requirements.txt",
        "--",
        *cmd,
    ]
    subprocess.run(uv, check=True)


def pytest() -> None:
    _run(["python", "-m", "pytest", "tests/src", "-vv"])


def lint() -> None:
    # Run ruff checks with auto-fix
    _run(["python", "-m", "ruff", "check", "--fix"])

    # Run ruff-format
    _run(["python", "-m", "ruff", "format"])

    # Run mypy for type checking
    _run(["python", "-m", "mypy", "src", "tests", "--ignore-missing-imports"])

    # Run isort with black profile
    _run(["python", "-m", "isort", ".", "--profile", "black", "--filter-files"])

    # Run pygrep checks
    print("Checking for blanket noqa comments and proper type annotations...")
    _run(["pre-commit", "run", "python-check-blanket-noqa", "--all-files"])
    _run(["pre-commit", "run", "python-use-type-annotations", "--all-files"])

    # Run pre-commit hooks for file formatting
    print("Running file formatting checks...")
    _run(["pre-commit", "run", "trailing-whitespace", "--all-files"])
    _run(["pre-commit", "run", "end-of-file-fixer", "--all-files"])
    _run(["pre-commit", "run", "check-yaml", "--all-files"])
    _run(["pre-commit", "run", "check-ast", "--all-files"])
    _run(["pre-commit", "run", "check-merge-conflict", "--all-files"])
    _run(["pre-commit", "run", "detect-private-key", "--all-files"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["pytest", "lint"])
    args = parser.parse_args()
    if args.cmd == "pytest":
        pytest()
    elif args.cmd == "lint":
        lint()


if __name__ == "__main__":
    main()
