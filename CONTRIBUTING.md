# Contributing to Legion

Welcome! We're excited you're interested in contributing to Legion. This document outlines the process for contributing and helps you get started.

## Quick Links
- [Discord Server](https://discord.gg/9QnHYaAD)
- [GitHub Project Board](https://github.com/orgs/LLMP-io/projects/1)
- [Issues](https://github.com/LLMP-io/Legion/issues)

## Getting Started

1. **Fork the Repository**
   - Click the "Fork" button in the top right of the GitHub repository
   - Clone your fork locally: `git clone https://github.com/YOUR-USERNAME/Legion.git`
   - Add upstream remote: `git remote add upstream https://github.com/LLMP-io/Legion.git`

2. **Set Up Development Environment**
   ```bash
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

## Development Guidelines

### Code Style
We follow PEP 8 guidelines for Python code style. Key points:
- Use 4 spaces for indentation
- Maximum line length of 79 characters for code, 72 for docstrings
- Use meaningful variable names
- Add docstrings to all functions, classes, and modules

### Before Submitting
1. Run tests: 
```bash
# Run all tests except integration tests
# NOTE: Unless integration tests are needed, this is perfectly fine
pytest -v -m "not integration"

# Run only integration tests
# NOTE: Integration tests require a valid API key for the providers they are testing (usually OpenAI + gpt-4o-mini)
pytest -v -m integration

# Run all tests
pytest -v
```

2. Update documentation if needed
3. Add tests for new features
4. Ensure your branch is up to date:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

## Pull Request Process

1. **Create Pull Request**
   - Push your changes to your fork
   - Create a PR against the `main` branch
   - Fill out the PR template
   - Link any related issues

2. **PR Guidelines**
   - Keep changes focused and atomic
   - Provide clear description of changes
   - Include any necessary documentation updates
   - Add screenshots for UI changes
   - Reference any related issues

3. **Review Process**
   - Maintainers will review your PR
   - Address any requested changes
   - Once approved, maintainers will merge your PR

## Issue Guidelines

### Creating Issues
- Check existing issues to avoid duplicates
- Use issue templates when available
- Provide clear reproduction steps for bugs
- For feature requests, explain the use case

### Good First Issues
Look for issues labeled `good-first-issue` if you're new to the project. These are typically:
- Documentation improvements
- Small bug fixes
- Test additions
- Simple feature implementations

## Getting Help

- Join our Discord server for real-time discussion
- Use the #help channel for technical questions
- Tag maintainers in GitHub issues if you need clarification

## Discord Channels

- #announcements: Project updates and news
- #general: Introduce yourself to the community and ask questions
- #help: Get help with technical issues
- #development: Discuss ongoing development
- #ideas: Share and discuss feature ideas
- #pull-requests: PR discussions and reviews

## Recognition

We value all contributions, big and small! Contributors will be:
- Added to our Contributors list
- Recognized in release notes for significant contributions
- Badges applied to your profile within the Legion Discord Server
- Given credit in documentation when appropriate

Thank you for contributing to Legion! 🚀