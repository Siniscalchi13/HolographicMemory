# Private Package Distribution Guide

## Overview

This guide sets up enterprise-grade private package distribution for Holographic Memory, enabling automated versioning, publishing, and consumption across your projects.

## Architecture

```
Development Workflow:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Development   │───▶│  GitHub Actions  │───▶│  Private PyPI   │
│ HolographicMemory│    │   CI/CD Pipeline │    │   (GitHub Pkg)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
   Semantic Commits         Auto-Versioning         Package Registry
   feat/fix/breaking        Changelog Generation     Authentication
```

## Quick Start

### 1. Initial Setup (One-time)

```bash
# Setup authentication
python scripts/setup_auth.py

# Configure package structure (already done)
# - pyproject.toml ✅
# - holographic_memory/__init__.py ✅
# - CI/CD workflows ✅
```

### 2. Development Workflow

```bash
# Work on holographic memory
git checkout -b feature/new-capability
# ... make changes ...

# Commit with semantic messages
git commit -m "feat: add quantum entanglement layer"
git commit -m "fix: resolve interference patterns in vault"
git commit -m "BREAKING CHANGE: update API for 7-layer architecture"

# Push to trigger CI/CD
git push origin feature/new-capability
```

### 3. Release Process

```bash
# Merge to main triggers automatic:
# 1. Version calculation (based on commits)
# 2. Package building
# 3. Publishing to private PyPI
# 4. GitHub release creation
# 5. Changelog generation

git checkout main
git merge feature/new-capability
git push origin main
# 🚀 Automatic release happens!
```

### 4. Consumer Project Setup

```bash
# In TAI or other projects
python /path/to/HolographicMemory/scripts/setup_consumer_project.py . --github

# Or manually add to pyproject.toml:
# holographic-memory = "^1.0.0"
```

## Versioning Strategy

### Semantic Versioning
- **Major (1.0.0)**: Breaking changes
- **Minor (1.1.0)**: New features, backward compatible
- **Patch (1.1.1)**: Bug fixes

### Commit Message Format
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Breaking Changes:**
```
feat: add new API endpoint

BREAKING CHANGE: removed deprecated search method
```

## Consumer Project Configuration

### Option 1: GitHub Packages (Recommended)

```toml
# pyproject.toml
[project]
dependencies = [
    "holographic-memory>=1.0.0,<2.0.0",
]

# .pip/pip.conf
[global]
extra-index-url = https://pypi.pkg.github.com/SmartHausGroup/
```

### Option 2: Private PyPI Server

```toml
# pyproject.toml
[project]
dependencies = [
    "holographic-memory>=1.0.0,<2.0.0",
]

# .pip/pip.conf
[global]
extra-index-url = https://pypi.smarthaus.ai/simple/
trusted-host = pypi.smarthaus.ai
```

## Automated Updates

Consumer projects automatically:
1. **Check weekly** for new versions
2. **Test compatibility** with existing code
3. **Create PR** if tests pass
4. **Notify** if manual intervention needed

## Security Features

### Authentication
- **GitHub Packages**: Personal Access Tokens
- **Private PyPI**: Username/password + API tokens
- **Environment Variables**: Secure secret management

### Access Control
- **Repository-based**: GitHub team permissions
- **Token-based**: Scoped access tokens
- **IP Restrictions**: Optional network-level security

### Audit Trail
- **Download tracking**: Who accessed what when
- **Version history**: Complete change log
- **Security scanning**: Automated vulnerability detection

## Monitoring & Maintenance

### Health Checks
```bash
# Test package availability
pip index versions holographic-memory --extra-index-url https://pypi.pkg.github.com/SmartHausGroup/

# Test installation
pip install holographic-memory==1.0.0 --dry-run

# Verify authentication
python -c "import holographic_memory; print(holographic_memory.__version__)"
```

### Troubleshooting

**Authentication Issues:**
```bash
# Check credentials
cat ~/.pypirc

# Test GitHub token
curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user

# Regenerate tokens if needed
python scripts/setup_auth.py
```

**Version Conflicts:**
```bash
# Check installed version
pip show holographic-memory

# Force specific version
pip install holographic-memory==1.2.3 --force-reinstall

# Clear pip cache
pip cache purge
```

## Best Practices

### Development
1. **Use semantic commits** for automatic versioning
2. **Test thoroughly** before merging to main
3. **Document breaking changes** in commit messages
4. **Keep dependencies minimal** in core package

### Consumption
1. **Pin major versions** (`^1.0.0`) for stability
2. **Test updates** in staging before production
3. **Monitor for security updates** regularly
4. **Use virtual environments** for isolation

### Security
1. **Rotate tokens** regularly (quarterly)
2. **Use least-privilege** access controls
3. **Monitor access logs** for anomalies
4. **Keep authentication** credentials secure

## Support

### Common Issues
- **Authentication failures**: Check token expiration
- **Version conflicts**: Use virtual environments
- **Network issues**: Verify firewall/proxy settings
- **Build failures**: Check C++ dependencies

### Getting Help
1. Check logs in GitHub Actions
2. Review package installation output
3. Test with minimal reproduction case
4. Contact development team with specifics

---

**Status**: ✅ Production Ready
**Last Updated**: 2024
**Maintainer**: SmartHaus Group
