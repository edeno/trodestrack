#!/usr/bin/env bash
# scripts/check_parity.sh
#
# Run regression tests and benchmarks to verify numerical parity
# after refactoring. This script ensures that code changes don't
# introduce behavior drift.
#
# Usage:
#   ./scripts/check_parity.sh [--quick]
#
# Options:
#   --quick    Skip slow tests and benchmarks

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

QUICK_MODE=0
if [[ "${1:-}" == "--quick" ]]; then
    QUICK_MODE=1
fi

echo -e "${GREEN}=== Trodestrack Parity Check ===${NC}"
echo ""

# Change to project root
cd "$(dirname "$0")/.."

# 1. Run style checks
echo -e "${YELLOW}[1/5] Running style checks...${NC}"
uv run ruff check src/ tests/ || {
    echo -e "${RED}✗ Ruff check failed${NC}"
    exit 1
}
uv run black --check src/ tests/ || {
    echo -e "${RED}✗ Black formatting check failed${NC}"
    exit 1
}
echo -e "${GREEN}✓ Style checks passed${NC}"
echo ""

# 2. Run type checks
echo -e "${YELLOW}[2/5] Running type checks...${NC}"
uv run mypy src/trodestrack --ignore-missing-imports || {
    echo -e "${RED}✗ Mypy type check failed${NC}"
    exit 1
}
echo -e "${GREEN}✓ Type checks passed${NC}"
echo ""

# 3. Run unit tests
echo -e "${YELLOW}[3/5] Running unit tests...${NC}"
if [[ $QUICK_MODE -eq 1 ]]; then
    uv run pytest tests/ -m "not slow and not benchmark" -q || {
        echo -e "${RED}✗ Unit tests failed${NC}"
        exit 1
    }
else
    uv run pytest tests/ -m "not benchmark" -v || {
        echo -e "${RED}✗ Unit tests failed${NC}"
        exit 1
    }
fi
echo -e "${GREEN}✓ Unit tests passed${NC}"
echo ""

# 4. Run integration/acceptance tests
echo -e "${YELLOW}[4/5] Running integration tests...${NC}"
if [[ $QUICK_MODE -eq 1 ]]; then
    echo -e "${YELLOW}  (skipped in quick mode)${NC}"
else
    uv run pytest tests/integration/ -v || {
        echo -e "${RED}✗ Integration tests failed${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ Integration tests passed${NC}"
fi
echo ""

# 5. Run benchmarks (if not in quick mode)
echo -e "${YELLOW}[5/5] Running benchmarks...${NC}"
if [[ $QUICK_MODE -eq 1 ]]; then
    echo -e "${YELLOW}  (skipped in quick mode)${NC}"
else
    uv run pytest tests/ -m benchmark --benchmark-only -q || {
        echo -e "${RED}✗ Benchmarks failed${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ Benchmarks passed${NC}"
fi
echo ""

echo -e "${GREEN}=== All parity checks passed! ===${NC}"
echo ""
echo "Numerical parity verified. It is safe to proceed with refactoring."
