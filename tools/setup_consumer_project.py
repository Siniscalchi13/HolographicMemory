#!/usr/bin/env python3
"""Setup script to configure a project to consume holographic-memory."""

import os
import sys
import subprocess
from pathlib import Path
import argparse

def setup_pip_config(project_path: Path, pypi_url: str, token: str = None):
    """Configure pip to use private PyPI server."""
    pip_conf_dir = project_path / ".pip"
    pip_conf_dir.mkdir(exist_ok=True)
    
    pip_conf = pip_conf_dir / "pip.conf"
    
    config_content = f"""[global]
extra-index-url = {pypi_url}
trusted-host = {pypi_url.split('//')[1].split('/')[0]}
"""
    
    if token:
        config_content += f"""
[install]
extra-index-url = https://__token__:{token}@{pypi_url.split('//')[1]}
"""
    
    pip_conf.write_text(config_content)
    print(f"Created pip config at {pip_conf}")

def setup_github_packages(project_path: Path):
    """Setup for GitHub Packages authentication."""
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        print("Warning: GITHUB_TOKEN environment variable not set")
        print("You'll need to set this for authentication")
    
    # Create .pypirc for publishing (if needed)
    pypirc = project_path / ".pypirc"
    pypirc_content = """[distutils]
index-servers = github

[github]
repository = https://pypi.pkg.github.com/SmartHausGroup/
username = __token__
password = {github_token}
""".format(github_token=github_token or "YOUR_GITHUB_TOKEN")
    
    pypirc.write_text(pypirc_content)
    print(f"Created .pypirc at {pypirc}")

def install_holographic_memory(project_path: Path, version: str = None):
    """Install holographic-memory package."""
    os.chdir(project_path)
    
    package_spec = "holographic-memory"
    if version:
        package_spec += f"=={version}"
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            package_spec, "--upgrade"
        ], check=True)
        print(f"Successfully installed {package_spec}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_spec}: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Setup project to consume holographic-memory")
    parser.add_argument("project_path", help="Path to the consumer project")
    parser.add_argument("--pypi-url", default="https://pypi.pkg.github.com/SmartHausGroup/", 
                       help="Private PyPI server URL")
    parser.add_argument("--version", help="Specific version to install")
    parser.add_argument("--github", action="store_true", help="Setup for GitHub Packages")
    
    args = parser.parse_args()
    
    project_path = Path(args.project_path).resolve()
    if not project_path.exists():
        print(f"Error: Project path {project_path} does not exist")
        sys.exit(1)
    
    print(f"Setting up holographic-memory consumption for {project_path}")
    
    if args.github:
        setup_github_packages(project_path)
    else:
        setup_pip_config(project_path, args.pypi_url)
    
    if install_holographic_memory(project_path, args.version):
        print("✅ Setup complete!")
        print("\nNext steps:")
        print("1. Import: from holographic_memory import HolographicMemory")
        print("2. Use in your code")
        print("3. Updates will be automatic based on your version constraints")
    else:
        print("❌ Setup failed - check authentication and network access")

if __name__ == "__main__":
    main()
