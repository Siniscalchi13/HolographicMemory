#!/usr/bin/env python3
"""Setup authentication for private PyPI access."""

import os
import sys
import secrets
import hashlib
import getpass
from pathlib import Path
import configparser

def generate_api_token(username: str, length: int = 32) -> str:
    """Generate a secure API token."""
    token = secrets.token_urlsafe(length)
    return f"hm_{username}_{token}"

def create_htpasswd_entry(username: str, password: str) -> str:
    """Create htpasswd entry for basic auth."""
    # Using SHA-1 for compatibility with pypiserver
    password_hash = hashlib.sha1(password.encode()).hexdigest()
    return f"{username}:{{SHA}}{password_hash}"

def setup_github_auth():
    """Setup GitHub Packages authentication."""
    print("Setting up GitHub Packages authentication...")
    
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        github_token = getpass.getpass("Enter your GitHub Personal Access Token: ")
    
    # Create .pypirc
    pypirc_content = f"""[distutils]
index-servers = 
    pypi
    github

[pypi]
username = __token__
password = {github_token}

[github]
repository = https://upload.pypi.org/legacy/
username = __token__
password = {github_token}
"""
    
    pypirc_path = Path.home() / ".pypirc"
    pypirc_path.write_text(pypirc_content)
    pypirc_path.chmod(0o600)  # Secure permissions
    
    print(f"✅ Created .pypirc at {pypirc_path}")
    
    # Environment setup
    env_vars = f"""
# Add to your shell profile (.bashrc, .zshrc, etc.)
export GITHUB_TOKEN="{github_token}"
export PIP_EXTRA_INDEX_URL="https://pypi.pkg.github.com/SmartHausGroup/"
"""
    
    print("Environment variables to set:")
    print(env_vars)

def setup_private_pypi_auth():
    """Setup private PyPI server authentication."""
    print("Setting up private PyPI server authentication...")
    
    username = input("Enter username: ")
    password = getpass.getpass("Enter password: ")
    
    # Generate API token
    api_token = generate_api_token(username)
    
    # Create htpasswd entry
    htpasswd_entry = create_htpasswd_entry(username, password)
    
    # Write htpasswd file
    auth_dir = Path("infrastructure/pypi-server/auth")
    auth_dir.mkdir(parents=True, exist_ok=True)
    
    htpasswd_file = auth_dir / ".htpasswd"
    htpasswd_file.write_text(htpasswd_entry + "\n")
    
    print(f"✅ Created htpasswd file at {htpasswd_file}")
    
    # Create .pypirc for client
    pypirc_content = f"""[distutils]
index-servers = 
    pypi
    private

[pypi]
username = __token__
password = {api_token}

[private]
repository = https://pypi.smarthaus.ai/
username = {username}
password = {password}
"""
    
    pypirc_path = Path.home() / ".pypirc"
    pypirc_path.write_text(pypirc_content)
    pypirc_path.chmod(0o600)
    
    print(f"✅ Created .pypirc at {pypirc_path}")
    
    # Client configuration
    pip_conf = f"""[global]
extra-index-url = https://{username}:{password}@pypi.smarthaus.ai/simple/
trusted-host = pypi.smarthaus.ai
"""
    
    print("Pip configuration:")
    print(pip_conf)
    
    return api_token

def setup_environment_secrets():
    """Setup environment variables for CI/CD."""
    print("\n🔐 Environment Secrets Setup")
    print("Add these secrets to your GitHub repository:")
    print("Settings > Secrets and variables > Actions")
    
    secrets_list = [
        ("GITHUB_TOKEN", "Your GitHub Personal Access Token"),
        ("PYPI_TOKEN", "Token for publishing to PyPI"),
        ("PRIVATE_PYPI_USERNAME", "Username for private PyPI"),
        ("PRIVATE_PYPI_PASSWORD", "Password for private PyPI"),
    ]
    
    for secret_name, description in secrets_list:
        print(f"- {secret_name}: {description}")

def main():
    print("🔐 Holographic Memory Authentication Setup")
    print("=" * 50)
    
    choice = input("""
Choose authentication method:
1. GitHub Packages (recommended)
2. Private PyPI Server
3. Environment Secrets Setup
Enter choice (1-3): """).strip()
    
    if choice == "1":
        setup_github_auth()
    elif choice == "2":
        token = setup_private_pypi_auth()
        print(f"\n🔑 Generated API token: {token}")
        print("Store this securely - it won't be shown again!")
    elif choice == "3":
        setup_environment_secrets()
    else:
        print("Invalid choice")
        sys.exit(1)
    
    print("\n✅ Authentication setup complete!")
    print("\nNext steps:")
    print("1. Test authentication: pip install holographic-memory")
    print("2. Configure your consumer projects")
    print("3. Set up CI/CD secrets in GitHub")

if __name__ == "__main__":
    main()
