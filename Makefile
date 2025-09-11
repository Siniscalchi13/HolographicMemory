# ═══════════════════════════════════════════════════════════════════════════════
# HolographicMemory - Main Makefile
# ═══════════════════════════════════════════════════════════════════════════════

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ Include Dedicated Makefiles                                               │
# └─────────────────────────────────────────────────────────────────────────────┘

include Makefile.help
include Makefile.dev
include Makefile.prod
include Makefile.lint
include Makefile.syntax
include Makefile.testing

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ Default Target                                                             │
# └─────────────────────────────────────────────────────────────────────────────┘

.DEFAULT_GOAL := help

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ Target Declarations                                                        │
# └─────────────────────────────────────────────────────────────────────────────┘

.PHONY: help setup test build native api soa clean dev prod lint syntax dev-verify launch-dev dev-all test-all test-quick test-unit test-integration test-e2e test-performance test-coverage test-html test-open test-clean test-ci

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ Development Targets                                                        │
# └─────────────────────────────────────────────────────────────────────────────┘

setup: dev-setup
test: dev-test
build: dev-build
native: dev-native
api: dev-api
soa: dev-soa
dev: dev-setup
launch-dev: dev-launch
dev-kill: dev-kill
dev-all: dev-all
clean: dev-clean

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ Production Targets                                                         │
# └─────────────────────────────────────────────────────────────────────────────┘

deploy: prod-deploy
release: prod-release

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ Quality Targets                                                            │
# └─────────────────────────────────────────────────────────────────────────────┘

lint: lint-all
syntax: syntax-check

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ Testing Targets                                                            │
# └─────────────────────────────────────────────────────────────────────────────┘

test-all: test-all
test-quick: test-quick
test-unit: test-unit
test-integration: test-integration
test-e2e: test-e2e
test-performance: test-performance
test-coverage: test-coverage
test-html: test-html
test-open: test-open
test-clean: test-clean
test-ci: test-ci

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ Help Target                                                                │
# └─────────────────────────────────────────────────────────────────────────────┘

help: help-main