# Task ID: 22

**Title:** Create GitHub Actions Test Workflow Enhancement

**Status:** pending

**Dependencies:** None

**Priority:** high

**Description:** Enhance the existing CI workflow to improve caching, add security scanning, and prepare for the build workflow integration.

**Details:**

**Update `.github/workflows/ci.yml` to enhance existing workflow:**

Add security scanning job after existing jobs:
```yaml
  security-scan:
    name: Security Scan
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'

      - name: Run npm audit (root)
        run: npm audit --audit-level=high || true
        continue-on-error: true

      - name: Run npm audit (server)
        run: cd server && npm audit --audit-level=high || true
        continue-on-error: true

      - name: Setup Python for ML service scan
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Run safety check (ML service)
        run: |
          pip install safety
          cd ml-service && safety check -r requirements.txt || true
        continue-on-error: true

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          severity: 'CRITICAL,HIGH'
          exit-code: '0'  # Don't fail build, just report
```

Update ci-success job to include security-scan:
```yaml
  ci-success:
    name: CI Success
    runs-on: ubuntu-latest
    needs: [lint, backend-tests, mobile-tests, ml-service-lint, security-scan]
    if: always()
    steps:
      - name: Check all jobs
        run: |
          if [[ "${{ needs.lint.result }}" != "success" ]] || \
             [[ "${{ needs.backend-tests.result }}" != "success" ]] || \
             [[ "${{ needs.mobile-tests.result }}" != "success" ]] || \
             [[ "${{ needs.ml-service-lint.result }}" != "success" ]]; then
            echo "One or more jobs failed!"
            exit 1
          fi
          # Security scan is informational, don't fail on it
          echo "All required CI jobs passed!"
```

**Add caching improvements for backend tests:**
```yaml
      - name: Cache Prisma client
        uses: actions/cache@v4
        with:
          path: server/node_modules/.prisma
          key: prisma-${{ runner.os }}-${{ hashFiles('server/prisma/schema.prisma') }}
```

**Test Strategy:**

1. Create a PR and verify workflow runs
2. Check security-scan job executes without blocking
3. Verify caching works - second run should be faster
4. Check Trivy reports in workflow output
5. Verify npm audit runs for both root and server
6. Confirm ci-success job still requires core jobs to pass

## Subtasks

### 22.1. Add security-scan job with npm audit, safety check, and Trivy scanner

**Status:** pending  
**Dependencies:** None  

Create a new security-scan job in .github/workflows/ci.yml that runs after existing jobs. Include npm audit for root and server directories, safety check for ML service Python dependencies, and Trivy filesystem vulnerability scanner.

**Details:**

Add a new job to .github/workflows/ci.yml after ml-service-lint job:

1. Configure job to run on ubuntu-latest
2. Checkout code using actions/checkout@v4
3. Setup Node.js using actions/setup-node@v4 with cache: 'npm'
4. Run npm audit for root directory with --audit-level=high, use continue-on-error: true
5. Run npm audit for server directory with --audit-level=high, use continue-on-error: true
6. Setup Python 3.11 using actions/setup-python@v5
7. Install safety package and run safety check on ml-service/requirements.txt with continue-on-error: true
8. Add Trivy vulnerability scanner using aquasecurity/trivy-action@master with scan-type: 'fs', scan-ref: '.', severity: 'CRITICAL,HIGH', and exit-code: '0'

All security checks should be informational only (not fail the build) using continue-on-error or exit-code: '0'.

### 22.2. Add Prisma client caching to backend-tests job

**Status:** pending  
**Dependencies:** None  

Enhance the backend-tests job with Prisma client caching to improve CI performance by avoiding regeneration of Prisma client on subsequent runs when schema hasn't changed.

**Details:**

Update the backend-tests job in .github/workflows/ci.yml:

1. Add a new step after Node.js setup and before dependency installation
2. Use actions/cache@v4 to cache Prisma client
3. Set cache path to 'server/node_modules/.prisma'
4. Use cache key format: prisma-${{ runner.os }}-${{ hashFiles('server/prisma/schema.prisma') }}
5. This ensures cache is invalidated only when schema.prisma changes

The caching step should be positioned strategically to maximize reuse while ensuring Prisma client is available for tests.

### 22.3. Update ci-success job to include security-scan dependency

**Status:** pending  
**Dependencies:** 22.1  

Modify the ci-success job to include security-scan in its needs array while ensuring it doesn't fail the build. Security scan results should be informational only.

**Details:**

Update the ci-success job in .github/workflows/ci.yml:

1. Add 'security-scan' to the needs array: [lint, backend-tests, mobile-tests, ml-service-lint, security-scan]
2. Keep if: always() condition to ensure job runs even if security-scan has issues
3. Update the check script to only verify core jobs (lint, backend-tests, mobile-tests, ml-service-lint) succeeded
4. Do NOT check security-scan result in the failure logic
5. Add a comment in the success echo statement noting that security scan is informational
6. Ensure the job still returns exit code 1 if any core jobs fail, but ignores security-scan status

This makes security-scan visible in the workflow without blocking merges.
