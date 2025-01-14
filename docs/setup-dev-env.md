# Development Environment Setup Guide

This guide provides instructions for setting up the development environment using either manual installation or Nix shell.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Manual Installation](#manual-installation)
- [Nix Shell Setup](#nix-shell-setup)
- [Environment Verification](#environment-verification)

## Prerequisites

- Windows 10/11 with WSL2 enabled
- Ubuntu on WSL2
- Git

## Manual Installation

### 1. Update WSL Ubuntu System
```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Install Python and Development Tools
```bash
sudo apt install -y python3.11 python3.11-venv python3-pip build-essential
```

### 3. Create and Activate Virtual Environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

## Nix Shell Setup

### 1. Install Nix Package Manager
```bash
sh <(curl -L https://nixos.org/nix/install) --daemon
```

### 2. Enable Flakes (Optional)
Add to `~/.config/nix/nix.conf`:
```
experimental-features = nix-command flakes
```

### 3. Using the Development Environment
```bash
nix-shell shell.nix
```

## Environment Verification

To verify your setup is working correctly:

1. Activate the environment (either venv or nix-shell)
2. Run the verification script:
```bash
python verify_env.py
```

Expected output should show all required dependencies are available.

## Included Tools and Dependencies

- Python 3.11
- pip
- venv
- build-essential
- Development libraries

## Troubleshooting

If you encounter any issues:

1. Ensure WSL2 is properly installed and running
2. Verify all prerequisites are installed
3. Check system Python version matches requirements
4. Ensure all paths are correctly set

## Maintenance

Last updated: 2025-01-14
