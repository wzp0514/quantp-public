# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not** open a public issue.

Instead, email the maintainers directly. We will respond within 48 hours and aim to publish a fix within 7 days.

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest  | Yes       |

## Best Practices for Users

- Never commit `settings.local.yaml` or any file containing API keys and tokens.
- Use environment variables for sensitive configuration.
- Review the `.gitignore` before contributing to ensure no local data files are included.
