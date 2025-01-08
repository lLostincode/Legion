# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |

## Reporting a Vulnerability

We take the security of Legion seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Reporting Process

**Please do not report security vulnerabilities through public GitHub issues.**

Instead:

1. Email us at [hayden@llmp.io](mailto:hayden@llmp.io)
2. Include as much information as possible:
   - A clear description of the vulnerability
   - Steps to reproduce the issue
   - Versions affected
   - Potential impact
   - Suggested fixes (if any)

### What to Expect

After you submit a report:

1. You'll receive an acknowledgment within 48 hours.
2. We'll investigate and keep you updated on our findings.
3. Once we've determined the impact and resolution:
   - We'll develop and test a fix
   - We'll establish a disclosure timeline
   - We'll notify affected users as appropriate

### Safe Harbor

We support safe harbor for security research that:
- Follows our reporting guidelines
- Makes a good faith effort to avoid privacy violations, data destruction, service interruption, and other harm
- Does not exploit findings beyond what's necessary to demonstrate the vulnerability

### Public Disclosure

We aim to address critical vulnerabilities within 30 days. We request that you keep vulnerabilities private until we release fixes. We'll coordinate with you on a disclosure timeline that serves both the community's need to update and your recognition as the reporter.

## Security Best Practices for Contributors

1. **Dependency Management**
   - Keep dependencies up to date
   - Review dependency changes carefully
   - Use dependabot alerts

2. **Code Review**
   - Review for security implications
   - Follow secure coding guidelines
   - Use security linters when possible

3. **Secrets and Credentials**
   - Never commit secrets or credentials
   - Use environment variables for sensitive data
   - Review code for accidental credential exposure

## Security Updates

Security updates will be released as:
1. Immediate patches for critical vulnerabilities
2. Regular updates for non-critical security improvements
3. Dependencies updates via automated tools

Updates will be announced through:
- GitHub Security Advisories
- Release notes
- Discord announcements channel

## Questions

If you have questions about this policy or Legion's security practices, please reach out on our Discord server in the #help channel.
