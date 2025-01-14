# Nix Development Environment Setup Guide

## What is Nix?

Nix creates isolated development environments that bundle everything your project needs - from system packages (like those you'd install with apt) to language-specific dependencies (like Python packages). Unlike traditional approaches that require separate virtual environments, Nix manages all dependencies in isolation, ensuring everyone on your team has exactly the same development setup, down to the specific package versions.

**Project Repository:** [https://github.com/NixOS/nix](https://github.com/NixOS/nix)

## Setup Instructions

### 1. Installation

```bash
sudo apt install nix-bin
```

### 2. System Configuration

First, verify the Nix daemon is running:
```bash
sudo systemctl status nix-daemon
```

Check socket permissions:
```bash
ls -l /nix/var/nix/daemon-socket/socket
```

If you don't have proper access:
```bash
sudo usermod -aG nix-users $USER
sudo chmod 666 /nix/var/nix/daemon-socket/socket
```

**Important**: Log out and log back in for group changes to take effect.

Verify permissions after logging back in:
```bash
ls -l /nix/var/nix/daemon-socket/socket
```

### 3. Channel Setup

Configure the Nix package repository:
```bash
nix-channel --add https://nixos.org/channels/nixos-unstable nixpkgs
nix-channel --update
```

### 4. Using the Development Environment

Start the environment:
```bash
nix-shell
```

If you encounter `bash: $'\r': command not found`:
- Exit the shell
- Fix line endings:
  ```bash
  sed -i 's/\r$//' shell.nix
  ```
- Enter the shell again

Exit the environment:
```bash
exit
```

## Troubleshooting

### Permission Issues
- Verify group membership: `groups`
- Check socket permissions: `ls -l /nix/var/nix/daemon-socket/socket`
- Restart daemon if needed: `sudo systemctl restart nix-daemon`

### Channel Update Failures
- Check internet connection
- Verify channel: `nix-channel --list`
- Try removing and re-adding channel

## Tips for Development

- All dependencies (Python packages, system libraries, tools) are managed by Nix
- No need for virtual environments - Nix provides complete isolation
- Dependencies are pulled from Nixpkgs, ensuring version consistency across team members
- Environment changes only affect your shell session - your system remains clean

## Quick Reference

Start development:
```bash
nix-shell  # Enters isolated environment with all dependencies
```

Exit development:
```bash
exit  # Returns to normal shell
```