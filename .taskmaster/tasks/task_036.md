# Task ID: 36

**Title:** Configure Dependabot for Security Updates

**Status:** pending

**Dependencies:** None

**Priority:** medium

**Description:** Enable GitHub Dependabot for automated dependency vulnerability scanning and pull request creation for security updates.

**Details:**

**Create `.github/dependabot.yml`:**
```yaml
version: 2
updates:
  # JavaScript/TypeScript dependencies (root - mobile app)
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
      timezone: "UTC"
    open-pull-requests-limit: 10
    labels:
      - "dependencies"
      - "mobile"
    reviewers:
      - "onurtemizkan"
    commit-message:
      prefix: "deps(mobile):"
    ignore:
      # Ignore major version updates for React Native (can be breaking)
      - dependency-name: "react-native"
        update-types: ["version-update:semver-major"]
      - dependency-name: "expo"
        update-types: ["version-update:semver-major"]

  # Backend dependencies
  - package-ecosystem: "npm"
    directory: "/server"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
      timezone: "UTC"
    open-pull-requests-limit: 10
    labels:
      - "dependencies"
      - "backend"
    reviewers:
      - "onurtemizkan"
    commit-message:
      prefix: "deps(server):"
    groups:
      # Group minor and patch updates together
      production-dependencies:
        patterns:
          - "*"
        exclude-patterns:
          - "@types/*"
          - "typescript"
        update-types:
          - "minor"
          - "patch"

  # Python ML Service dependencies
  - package-ecosystem: "pip"
    directory: "/ml-service"
    schedule:
      interval: "weekly"
      day: "tuesday"
      time: "09:00"
      timezone: "UTC"
    open-pull-requests-limit: 5
    labels:
      - "dependencies"
      - "ml-service"
    reviewers:
      - "onurtemizkan"
    commit-message:
      prefix: "deps(ml):"
    ignore:
      # ML libraries can have breaking changes
      - dependency-name: "torch"
        update-types: ["version-update:semver-major"]
      - dependency-name: "scikit-learn"
        update-types: ["version-update:semver-major"]

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "wednesday"
      time: "09:00"
      timezone: "UTC"
    labels:
      - "dependencies"
      - "ci"
    commit-message:
      prefix: "ci:"

  # Docker base images
  - package-ecosystem: "docker"
    directory: "/server"
    schedule:
      interval: "weekly"
      day: "thursday"
    labels:
      - "dependencies"
      - "docker"
    commit-message:
      prefix: "docker(server):"

  - package-ecosystem: "docker"
    directory: "/ml-service"
    schedule:
      interval: "weekly"
      day: "thursday"
    labels:
      - "dependencies"
      - "docker"
    commit-message:
      prefix: "docker(ml):"
```

**Enable GitHub Security Features:**

1. Go to repository Settings > Security > Code security and analysis
2. Enable:
   - Dependency graph
   - Dependabot alerts
   - Dependabot security updates
   - Secret scanning
   - Push protection

**Create security policy `.github/SECURITY.md`:**
```markdown
# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |

## Reporting a Vulnerability

Please report security vulnerabilities by emailing security@nutri.app.

Do NOT create public GitHub issues for security vulnerabilities.

### Response Timeline

- Initial response: 24 hours
- Triage: 48 hours
- Fix timeline: Depends on severity
  - Critical: 24-48 hours
  - High: 7 days
  - Medium: 30 days
  - Low: Next release

## Security Measures

- All dependencies automatically scanned
- Security updates applied weekly
- Docker images scanned for vulnerabilities
- Secrets scanning enabled
```

**Document in `docs/deployment/SECURITY-UPDATES.md`:**
```markdown
# Dependency Security Updates

## Automated Updates

Dependabot creates PRs for:
- npm packages (mobile, server)
- pip packages (ml-service)
- GitHub Actions versions
- Docker base images

## Weekly Schedule

| Day | Package Ecosystem |
|-----|-------------------|
| Monday | npm (mobile, server) |
| Tuesday | pip (ml-service) |
| Wednesday | GitHub Actions |
| Thursday | Docker images |

## Handling PRs

1. Review changes and changelogs
2. Run CI checks
3. Merge if tests pass
4. For breaking changes, test manually first
```

**Test Strategy:**

1. Commit dependabot.yml and verify it appears in Insights > Dependency graph > Dependabot
2. Manually trigger dependency check: Settings > Security > Code security
3. Wait for first PRs to appear (within a week)
4. Verify PR labels and commit message format
5. Check grouping works for minor/patch updates
6. Verify ignore rules work (no major React Native PRs)
7. Test that CI runs on Dependabot PRs
8. Enable secret scanning and verify no alerts
