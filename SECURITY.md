# Security policy

## Reporting

Report a vulnerability or leaked credential privately to the maintainers via
GitHub Security Advisories (Security → Report a vulnerability) or email
`security@eulogik.com`. Do not open a public issue for a live secret.

## Credential policy

- No tokens, keys or `.env` files belong in git. `.env` is ignored.
- The public history was rewritten on 2026-11-12 to remove two expired
  credentials from old notebook commits.
- GitHub secret scanning and push protection are enabled on this repository.
- If you find a token in an issue, PR, log or commit, report it privately and
  rotate it immediately.

## Scope

This is an offline assistant system, not a medical device or a hosted service.
Health protocols are fixed text and must not be used as diagnosis. The
`docs/health_safety.md` page lists the limitations.
