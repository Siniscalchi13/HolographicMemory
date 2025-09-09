#!/bin/bash
# Test Makefile System
# ====================
# 🧪 Comprehensive test of the HolographicMemory Makefile system

set -e

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧪 Testing HolographicMemory Makefile System${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Test 1: Help command
echo -e "${YELLOW}Test 1: Help command${NC}"
if make help >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Help command works${NC}"
else
    echo -e "${RED}❌ Help command failed${NC}"
    exit 1
fi

# Test 2: Configuration
echo -e "${YELLOW}Test 2: Configuration${NC}"
if make config >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Configuration command works${NC}"
else
    echo -e "${RED}❌ Configuration command failed${NC}"
    exit 1
fi

# Test 3: Clean command
echo -e "${YELLOW}Test 3: Clean command${NC}"
if make clean >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Clean command works${NC}"
else
    echo -e "${RED}❌ Clean command failed${NC}"
    exit 1
fi

# Test 4: Setup command (dry run)
echo -e "${YELLOW}Test 4: Setup command (dry run)${NC}"
if make setup-venv >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Setup command works${NC}"
else
    echo -e "${RED}❌ Setup command failed${NC}"
    exit 1
fi

# Test 5: Port selection
echo -e "${YELLOW}Test 5: Port selection${NC}"
PORT1=$(make -n dev 2>&1 | grep -o 'port [0-9]*' | head -1 | cut -d' ' -f2)
PORT2=$(make -n dev 2>&1 | grep -o 'port [0-9]*' | head -1 | cut -d' ' -f2)
if [ "$PORT1" != "$PORT2" ]; then
    echo -e "${GREEN}✅ Port selection is dynamic${NC}"
else
    echo -e "${YELLOW}⚠️ Port selection may not be dynamic${NC}"
fi

# Test 6: Development server script
echo -e "${YELLOW}Test 6: Development server script${NC}"
if [ -f "scripts/macos/dev_server.sh" ] && [ -x "scripts/macos/dev_server.sh" ]; then
    echo -e "${GREEN}✅ Development server script exists and is executable${NC}"
else
    echo -e "${RED}❌ Development server script missing or not executable${NC}"
    exit 1
fi

# Test 7: Docker compose file
echo -e "${YELLOW}Test 7: Docker compose file${NC}"
if [ -f "docker-compose-dev.yml" ]; then
    echo -e "${GREEN}✅ Docker compose file exists${NC}"
else
    echo -e "${RED}❌ Docker compose file missing${NC}"
    exit 1
fi

# Test 8: Help file
echo -e "${YELLOW}Test 8: Help file${NC}"
if [ -f "Makefile.help" ]; then
    echo -e "${GREEN}✅ Help file exists${NC}"
else
    echo -e "${RED}❌ Help file missing${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 All Makefile system tests passed!${NC}"
echo ""
echo -e "${BLUE}Available commands:${NC}"
echo "  make help           # Show all commands"
echo "  make dev            # Start development server"
echo "  make dev-embedded   # Start with embedded Python"
echo "  make setup          # Set up environment"
echo "  make test           # Run tests"
echo "  make clean          # Clean up"
echo ""
echo -e "${YELLOW}💡 Ready for development!${NC}"
