# Task ID: 23

**Title:** Create GitHub Actions Build and Push Workflow

**Status:** done

**Dependencies:** 18 ✓, 19 ⧖

**Priority:** high

**Description:** Create a new workflow that builds Docker images for backend and ML service, scans them for vulnerabilities, and pushes to GitHub Container Registry on merge to main.

**Details:**

**Create `.github/workflows/build.yml`:**
```yaml
name: Build and Push

on:
  push:
    branches: [master]
  workflow_dispatch:  # Allow manual trigger

env:
  REGISTRY: ghcr.io
  BACKEND_IMAGE: ghcr.io/${{ github.repository }}/backend
  ML_SERVICE_IMAGE: ghcr.io/${{ github.repository }}/ml-service

jobs:
  build-backend:
    name: Build Backend
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.BACKEND_IMAGE }}
          tags: |
            type=sha,prefix=
            type=raw,value=latest,enable=${{ github.ref == 'refs/heads/master' }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: ./server
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.BACKEND_IMAGE }}:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-backend.sarif'
          severity: 'CRITICAL,HIGH'

      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: 'trivy-backend.sarif'

    outputs:
      image: ${{ env.BACKEND_IMAGE }}
      digest: ${{ steps.meta.outputs.digest }}

  build-ml-service:
    name: Build ML Service
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.ML_SERVICE_IMAGE }}
          tags: |
            type=sha,prefix=
            type=raw,value=latest,enable=${{ github.ref == 'refs/heads/master' }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: ./ml-service
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.ML_SERVICE_IMAGE }}:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-ml.sarif'
          severity: 'CRITICAL,HIGH'

      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: 'trivy-ml.sarif'

    outputs:
      image: ${{ env.ML_SERVICE_IMAGE }}
      digest: ${{ steps.meta.outputs.digest }}

  build-success:
    name: Build Success
    runs-on: ubuntu-latest
    needs: [build-backend, build-ml-service]
    steps:
      - name: Summary
        run: |
          echo "## Build Summary" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "✅ Backend Image: \`${{ needs.build-backend.outputs.image }}:${{ github.sha }}\`" >> $GITHUB_STEP_SUMMARY
          echo "✅ ML Service Image: \`${{ needs.build-ml-service.outputs.image }}:${{ github.sha }}\`" >> $GITHUB_STEP_SUMMARY
```

**Test Strategy:**

1. Push to master branch and verify workflow triggers
2. Check images appear in GitHub Packages
3. Verify image tags include SHA and 'latest'
4. Check Trivy scan results in Security tab
5. Pull and run images locally to verify they work
6. Test manual workflow_dispatch trigger
7. Verify build time <10 minutes with caching

## Subtasks

### 23.1. Create build.yml workflow file with triggers and environment configuration

**Status:** pending  
**Dependencies:** None  

Create the initial `.github/workflows/build.yml` file with workflow triggers (push to master branch and workflow_dispatch), and define environment variables for GitHub Container Registry (GHCR) and image names.

**Details:**

Create `.github/workflows/build.yml` with:
- Workflow name: 'Build and Push'
- Triggers: `on.push.branches: [master]` and `on.workflow_dispatch` for manual runs
- Environment variables: REGISTRY (ghcr.io), BACKEND_IMAGE, ML_SERVICE_IMAGE using `${{ github.repository }}` for dynamic repo name
- Ensure proper YAML formatting and indentation

This establishes the workflow foundation that subsequent jobs will build upon.

### 23.2. Implement build-backend job with Docker Buildx and GHCR authentication

**Status:** pending  
**Dependencies:** 23.1  

Create the build-backend job that sets up Docker Buildx, authenticates with GitHub Container Registry, extracts metadata for image tagging, and builds/pushes the backend Docker image with layer caching.

**Details:**

Add build-backend job with:
- Job permissions: `contents: read`, `packages: write`
- Steps: checkout@v4, setup-buildx-action@v3, login-action@v3 (using GITHUB_TOKEN)
- metadata-action@v5 for tags: SHA prefix and latest (conditional on master branch)
- build-push-action@v5 with context: ./server, push: true, GitHub Actions cache (type=gha)
- Job outputs: image name and digest from metadata step

References Task 18 (Backend Dockerfile) which must exist at ./server/Dockerfile.

### 23.3. Implement build-ml-service job with same Docker build pattern

**Status:** pending  
**Dependencies:** 23.1  

Create the build-ml-service job that mirrors the backend build process but targets the ML service, running in parallel with the backend build job.

**Details:**

Add build-ml-service job (parallel to build-backend) with:
- Same permissions and step structure as build-backend
- Context changed to: ./ml-service
- Image metadata using ML_SERVICE_IMAGE environment variable
- Same tagging strategy (SHA + latest on master)
- Same caching strategy (GitHub Actions cache)
- Job outputs: ML service image name and digest

References Task 19 (ML Service Dockerfile) which must exist at ./ml-service/Dockerfile.

### 23.4. Add Trivy vulnerability scanning with SARIF upload for both images

**Status:** pending  
**Dependencies:** 23.2, 23.3  

Integrate Trivy security scanning into both build jobs to scan Docker images for CRITICAL and HIGH severity vulnerabilities, output results in SARIF format, and upload to GitHub Security tab via CodeQL action.

**Details:**

Add to both build-backend and build-ml-service jobs:
- trivy-action@master step after build-push
- Scan image-ref using SHA tag: `${{ env.*_IMAGE }}:${{ github.sha }}`
- Output format: 'sarif', severity filter: 'CRITICAL,HIGH'
- Separate output files: trivy-backend.sarif and trivy-ml.sarif
- codeql-action/upload-sarif@v3 with `if: always()` to upload even on scan failures
- Ensure SARIF files are uploaded to GitHub Security tab for vulnerability tracking

### 23.5. Create build-success summary job with image output reporting

**Status:** pending  
**Dependencies:** 23.2, 23.3  

Implement the final build-success job that depends on both build jobs completing successfully and outputs a formatted summary of built images with their tags and digests to the GitHub Actions summary.

**Details:**

Add build-success job with:
- Depends on: [build-backend, build-ml-service] using `needs`
- Single step that writes to $GITHUB_STEP_SUMMARY
- Output format: Markdown with build summary header
- Display both image names with SHA tags using job outputs: `${{ needs.build-backend.outputs.image }}:${{ github.sha }}`
- Include digest information from metadata outputs
- Use checkmark emojis (✅) for visual clarity

This provides clear deployment information for the subsequent deploy workflow (Task 24).
