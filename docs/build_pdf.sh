#!/usr/bin/env bash
# Build PROJECT_REPORT.pdf from PROJECT_REPORT.md.
#
#   bash docs/build_pdf.sh        (from the repository root, or from docs/)
#
# Requires pandoc and a PDF engine. tectonic is used by default because it
# fetches what it needs rather than assuming a full TeX install.
#
# Arrows and "implies" are not reachable in the serif font's Latin shaping, so
# they are rewritten to their LaTeX math equivalents in a scratch copy. The
# Markdown source keeps the readable Unicode characters; only the PDF build
# sees the substitution. None of them occur inside fenced code blocks, which is
# what makes a global rewrite safe -- check that again if you add new ones.
#
# Note that this trick does not work for every glyph: \approx still resolves
# through the text font and fails the same way, so prose uses the word "about"
# instead. Prefer rewording over adding substitutions here.

set -euo pipefail

cd "$(dirname "$0")"

SRC="PROJECT_REPORT.md"
OUT="PROJECT_REPORT.pdf"
TMP="$(mktemp -t agentkv_report.XXXXXX).md"
trap 'rm -f "$TMP"' EXIT

if grep -q '```' "$SRC"; then
  if awk '/^```/{f=!f} f && /→|⇒/{found=1} END{exit !found}' "$SRC"; then
    echo "error: a substituted glyph appears inside a fenced code block;" >&2
    echo "       the global rewrite below would corrupt it. Fix before building." >&2
    exit 1
  fi
fi

sed -e 's/→/$\\rightarrow$/g' \
    -e 's/⇒/$\\Rightarrow$/g' \
    "$SRC" > "$TMP"

pandoc "$TMP" -o "$OUT" \
  --pdf-engine=tectonic \
  --resource-path=".:.." \
  -V mainfont="STIX Two Text" \
  -V monofont="Menlo" \
  -V monofontoptions="Scale=0.80" \
  -V linestretch=1.06 \
  --highlight-style=tango

echo "wrote docs/$OUT"
