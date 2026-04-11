#!/bin/bash
# ─────────────────────────────────────────────────
#  B2B Reverse Funnel Planner – Starter
# ─────────────────────────────────────────────────

cd "$(dirname "$0")"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📊 B2B Revenue Reverse Funnel Planner"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Python finden ─────────────────────────────────
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "❌ Kein Python gefunden. Bitte Python 3 installieren:"
    echo "   https://www.python.org/downloads/"
    read -p "Drücke Enter zum Beenden..." dummy
    exit 1
fi

echo "→ Python: $($PYTHON --version)"

# ── Streamlit installieren falls nötig ───────────
if ! $PYTHON -m streamlit --version &>/dev/null; then
    echo "→ Installiere Pakete (einmalig, dauert ~30 Sek.)..."
    $PYTHON -m pip install --upgrade pip -q
    $PYTHON -m pip install streamlit plotly pandas numpy openpyxl fpdf2 -q
    echo "→ Installation abgeschlossen."
else
    echo "→ Alle Pakete vorhanden ✅"
fi

echo ""
echo "→ Starte App..."
echo "→ Browser öffnet sich auf http://localhost:8501"
echo ""
echo "   Zum Beenden: Strg+C drücken"
echo ""

$PYTHON -m streamlit run app.py \
    --server.headless false \
    --browser.gatherUsageStats false
