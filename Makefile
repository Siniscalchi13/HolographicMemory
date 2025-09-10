# HolographicMemory - Main Makefile
# References dedicated Makefiles for different purposes

# Include dedicated Makefiles
include Makefile.help
include Makefile.dev
include Makefile.prod
include Makefile.lint
include Makefile.syntax

# Default target
.DEFAULT_GOAL := help

# Main targets that delegate to specific Makefiles
.PHONY: help setup test build native api soa clean dev prod lint syntax

# Development targets
setup: dev-setup
test: dev-test
build: dev-build
native: dev-native
api: dev-api
soa: dev-soa

# Production targets
deploy: prod-deploy
release: prod-release

# Quality targets
lint: lint-all
syntax: syntax-check

# Help target
help: help-main