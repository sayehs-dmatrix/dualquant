#!/usr/bin/env python3
"""Produce an anonymous copy of this repository for double-blind review.

Writes a sibling directory (default ../dualquant-anon) containing a fresh git
repo with a single squashed commit authored by "Anonymous Author". The source
repository is never modified.

What it scrubs
--------------
1. Git history: author/committer names and e-mail addresses, and commit
   messages that name the originating monorepo, internal hosts or usernames.
   Handled by squashing to one commit rather than rewriting, which is both
   simpler and leaves no reflog/parent metadata to leak.
2. The vendor's quantization library. `from dmx.compressor import Format`
   becomes `from blockfmt import Format`, where `blockfmt/` is a small shim
   that resolves the backend from the BLOCKFMT_BACKEND environment variable.
   No vendor package name remains in the source.
3. Vendor and personal identifiers in file contents: company name, usernames,
   the originating monorepo name, machine-specific paths and build logs.

Run:  python tools/anonymize.py [--out DIR] [--check]
      --check  audit only; report what would leak, change nothing
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (regex, replacement, human-readable reason)
TEXT_RULES: list[tuple[str, str, str]] = [
    # vendor library import -> local shim
    (r"\bfrom dmx\.compressor import\b", "from blockfmt import", "vendor import"),
    (r"\bimport dmx\.compressor\b", "import blockfmt", "vendor import"),
    (r"\bdmx\.compressor\b", "blockfmt", "vendor module name"),
    (r"\bdmx[_-]compressor\b", "blockfmt", "vendor package name"),
    # Catch-all for any remaining dmx-prefixed identifier, e.g. the filename
    # `gptq_dmx_comprss_sfp4_vs_sbfp12.py` quoted in a legacy_vendor comment.
    # No leading \b: there the token is preceded by "_", which is a word
    # character, so a word boundary would never match. Must stay last of the
    # dmx rules so the specific spellings above win.
    (r"dmx\w*", "blockfmt", "vendor name (catch-all)"),
    # company name
    (r"\bd-Matrix\b", "the vendor", "company name"),
    (r"\bd_Matrix\b", "the vendor", "company name"),
    (r"\bdMatrix\b", "the vendor", "company name"),
    # originating monorepo + machine-specific paths. The trailing path is
    # optional: bare "/tmp/claude-0" appears inside quoted grep commands in
    # REPRODUCE.md, and a rule requiring a following slash would miss it.
    # The character class must also stop at shell and punctuation delimiters:
    # run_sweep.sh contains `: "${CUDA_HOME:=/home/coder/miniconda3}"`, and a
    # class that swallowed the closing brace broke the parameter expansion.
    (r"/tmp/claude-\d+[^\s\"'}),;]*", "<BUILD_DIR>", "build scratch path"),
    (r"/root/numrd[^\s\"':}),;]*", "<REPO>", "monorepo path"),
    (r"/home/coder/numrd[^\s\"':}),;]*", "<REPO>", "monorepo path"),
    (r"/home/coder/[^\s\"':}),;]*", "<HOME>", "username in path"),
    (r"/venv/main[^\s\"':}),;]*", "<VENV>", "machine venv path"),
    (r"\bnumrd\b", "the monorepo", "monorepo name"),
    # internal hosting
    (r"\binternal GitLab\b", "an internal package index", "internal host"),
    (r"\b[\w.+-]+@d-matrix\.ai\b", "anonymous@example.com", "corporate e-mail"),
    (r"\bd-matrix\.ai\b", "example.com", "corporate domain"),
    # venue: keep paths consistent with DIR_RENAMES above
    (r"\bRebutal_Neurips\b", "rebuttal", "venue in path"),
    (r"\bNeurIPS\b", "the venue", "venue name"),
]

# Directory renames applied to the copy. The venue is not author-identifying,
# but the artifact should not advertise which submission it belongs to.
# Format names (SFP4, SBFP12, ...) are deliberately NOT renamed: they appear in
# tracked result filenames and scale tensors, so aliasing them would break
# provenance against the recorded results.
DIR_RENAMES = {"Rebutal_Neurips": "rebuttal"}

SKIP_DIRS = {".git", "__pycache__", "hessian_cache", "scale_cache", "wrap_cache",
             "tools"}
TEXT_EXT = {".py", ".sh", ".md", ".txt", ".json", ".csv", ".cfg", ".toml",
            ".yaml", ".yml", ".log", ".tex", ".patch", ".cuh", ".cu", ".h",
            ".cpp", ".gitignore"}

SHIM = '''"""Block-format quantization backend (indirection layer).

The project depends on an external library that provides a ``Format`` class
with ``Format.from_shorthand(...)`` and ``.cast(tensor)``. That library is not
public, so it is resolved at import time from an environment variable instead
of being named in the source:

    export BLOCKFMT_BACKEND=<python.module.path>

The module named there must expose ``Format``.
"""

import importlib
import os

_BACKEND = os.environ.get("BLOCKFMT_BACKEND")

if not _BACKEND:
    raise ImportError(
        "Set BLOCKFMT_BACKEND to the import path of a module providing a "
        "`Format` class with .from_shorthand() and .cast(). This project "
        "depends on a block-format quantization library that is not publicly "
        "redistributable; see README.md."
    )

Format = importlib.import_module(_BACKEND).Format

__all__ = ["Format"]
'''


def iter_text_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            if os.path.splitext(fn)[1] in TEXT_EXT or fn == ".gitignore":
                yield p


def scrub_text(root: str, apply: bool) -> dict[str, int]:
    hits: dict[str, int] = {}
    for p in iter_text_files(root):
        try:
            s = open(p, encoding="utf-8", errors="surrogateescape").read()
        except OSError:
            continue
        orig = s
        for pat, rep, reason in TEXT_RULES:
            s, n = re.subn(pat, rep, s)
            if n:
                hits[reason] = hits.get(reason, 0) + n
        if apply and s != orig:
            open(p, "w", encoding="utf-8", errors="surrogateescape").write(s)
    return hits


def run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE),
                                                  "dualquant-anon"))
    ap.add_argument("--check", action="store_true",
                    help="audit only; do not write anything")
    a = ap.parse_args()

    if a.check:
        hits = scrub_text(HERE, apply=False)
        print("identifiers that WOULD be scrubbed (source left untouched):")
        for k in sorted(hits):
            print(f"  {hits[k]:5d}  {k}")
        authors = subprocess.run(["git", "log", "--format=%an <%ae>"], cwd=HERE,
                                 capture_output=True, text=True).stdout.split("\n")
        print("  git authors:", ", ".join(sorted({x for x in authors if x})))
        return 0

    if os.path.exists(a.out):
        print(f"refusing to overwrite existing {a.out}", file=sys.stderr)
        return 1

    # Copy tracked files only -- never the multi-hundred-GB caches.
    files = subprocess.run(["git", "ls-files"], cwd=HERE, capture_output=True,
                           text=True, check=True).stdout.split("\n")
    for rel in filter(None, files):
        if rel.startswith("tools/"):
            continue                      # this script must not ship
        parts = rel.split("/")
        parts = [DIR_RENAMES.get(p, p) for p in parts]
        dst = os.path.join(a.out, *parts)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(HERE, rel), dst)

    hits = scrub_text(a.out, apply=True)

    os.makedirs(os.path.join(a.out, "blockfmt"), exist_ok=True)
    with open(os.path.join(a.out, "blockfmt", "__init__.py"), "w") as fh:
        fh.write(SHIM)

    env = {**os.environ,
           "GIT_AUTHOR_NAME": "Anonymous Author",
           "GIT_AUTHOR_EMAIL": "anonymous@example.com",
           "GIT_COMMITTER_NAME": "Anonymous Author",
           "GIT_COMMITTER_EMAIL": "anonymous@example.com"}
    run(["git", "init", "-q", "-b", "main"], a.out)
    run(["git", "add", "-A"], a.out)
    subprocess.run(
        ["git", "-c", "user.name=Anonymous Author",
         "-c", "user.email=anonymous@example.com",
         "commit", "-q", "-m",
         "DualQuant: anonymous artifact for double-blind review"],
        cwd=a.out, env=env, check=True)

    print(f"wrote {a.out}")
    for k in sorted(hits):
        print(f"  scrubbed {hits[k]:5d}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
