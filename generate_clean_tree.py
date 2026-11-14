import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

# Directories to completely skip
EXCLUDE_DIRS = {
    "__pycache__",
    "node_modules",
    "migrations",
    "venv",
    "env",
    "myenv",
    ".git",
    ".idea",
    ".vscode",
    ".pytest_cache",
    "build",
    "dist",
    "collectstatic",
    "static",
    "media",
    "site-packages",
    "Lib",
    "Include",
    "Scripts",
}

# Files/extensions to skip
EXCLUDE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
    ".woff", ".woff2", ".ttf", ".eot",
    ".map", ".pyc", ".pyd", ".dll", ".so",
    ".log", ".sqlite3", ".bak", ".dat", ".dir",
    ".mmdb", ".txt"
}

# Exact files to skip
EXCLUDE_FILES = {
    "package-lock.json",
    "yarn.lock",
    ".DS_Store"
}


def generate_tree(path, prefix=""):
    try:
        items = sorted(os.listdir(path))
    except (FileNotFoundError, PermissionError):
        return

    visible_items = []

    for item in items:
        if item.startswith("."):
            continue

        full_path = os.path.join(path, item)

        if os.path.isdir(full_path):
            if item in EXCLUDE_DIRS:
                continue
            visible_items.append(item)

        else:
            ext = os.path.splitext(item)[1].lower()

            if item in EXCLUDE_FILES:
                continue

            if ext in EXCLUDE_EXTENSIONS:
                continue

            visible_items.append(item)

    for i, item in enumerate(visible_items):
        full_path = os.path.join(path, item)
        connector = "|-- " if i != len(visible_items) - 1 else "`-- "

        print(prefix + connector + item)

        if os.path.isdir(full_path):
            extension = "|   " if i != len(visible_items) - 1 else "    "
            generate_tree(full_path, prefix + extension)


if __name__ == "__main__":
    project_name = os.path.basename(os.getcwd())

    print("```bash")
    print(f"{project_name}/")
    generate_tree(".")
    print("```")


# Run:
# python generate_clean_tree.py > clean_tree.md