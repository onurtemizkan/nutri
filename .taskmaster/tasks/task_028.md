# Task ID: 28

**Title:** Create Server Setup Script for Hetzner

**Status:** done

**Dependencies:** None

**Priority:** medium

**Description:** Create an idempotent setup script that provisions a new Hetzner server with all required software, security configurations, and Coolify installation.

**Details:**

**Create `scripts/deploy/setup-server.sh`:**
```bash
#!/bin/bash
set -e

# =============================================================================
# Nutri Server Setup Script for Hetzner CX32
# Run as root on fresh Ubuntu 22.04 LTS
# =============================================================================

echo "=== Nutri Production Server Setup ==="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# =============================================================================
# System Updates and Basic Security
# =============================================================================

info "Updating system packages..."
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get upgrade -y -qq

info "Installing essential packages..."
apt-get install -y -qq \
    curl \
    wget \
    git \
    ufw \
    fail2ban \
    unattended-upgrades \
    apt-listchanges \
    htop \
    ncdu

# =============================================================================
# Security Hardening
# =============================================================================

info "Configuring firewall (UFW)..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp comment 'SSH'
ufw allow 80/tcp comment 'HTTP'
ufw allow 443/tcp comment 'HTTPS'
ufw --force enable
ufw status verbose

info "Configuring fail2ban..."
cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 86400
EOF
systemctl restart fail2ban
systemctl enable fail2ban

info "Configuring SSH security..."
sed -i 's/#PermitRootLogin yes/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart sshd

info "Enabling automatic security updates..."
dpkg-reconfigure -f noninteractive unattended-upgrades
cat > /etc/apt/apt.conf.d/50unattended-upgrades << 'EOF'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
};
Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
EOF

# =============================================================================
# Swap Configuration
# =============================================================================

if [ ! -f /swapfile ]; then
    info "Creating 4GB swap file..."
    fallocate -l 4G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo 'vm.swappiness=10' >> /etc/sysctl.conf
    sysctl -p
else
    info "Swap file already exists, skipping..."
fi

# =============================================================================
# Docker Installation (if not present)
# =============================================================================

if ! command -v docker &> /dev/null; then
    info "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    
    # Docker cleanup cron job
    cat > /etc/cron.daily/docker-cleanup << 'EOF'
#!/bin/bash
docker system prune -af --volumes --filter "until=168h" > /dev/null 2>&1
EOF
    chmod +x /etc/cron.daily/docker-cleanup
else
    info "Docker already installed, skipping..."
fi

# =============================================================================
# Coolify Installation
# =============================================================================

if [ ! -d /data/coolify ]; then
    info "Installing Coolify..."
    curl -fsSL https://cdn.coollabs.io/coolify/install.sh | bash
    info "Coolify installed! Access at https://your-server-ip:8000"
else
    info "Coolify already installed, skipping..."
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=== Setup Complete ==="
echo ""
info "Next steps:"
echo "1. Access Coolify at https://<server-ip>:8000"
echo "2. Create admin account and configure SSL"
echo "3. Set up GitHub webhook integration"
echo "4. Configure environment variables in Coolify"
echo ""
info "Security status:"
ufw status | head -10
echo ""
info "Resource usage:"
free -h
df -h /
echo ""
info "Docker version:"
docker --version
```

**Make executable:**
```bash
chmod +x scripts/deploy/setup-server.sh
```

**Test Strategy:**

1. Test on fresh Ubuntu 22.04 VM (Hetzner or local)
2. Verify idempotency - run script twice, second run should skip completed steps
3. Check UFW rules: `ufw status verbose`
4. Verify fail2ban: `fail2ban-client status sshd`
5. Test SSH still works after config changes
6. Verify swap: `free -h` shows 4GB swap
7. Verify Docker: `docker run hello-world`
8. Check Coolify accessible at :8000
9. Verify auto-updates: `apt-config dump | grep -i unattended`

## Subtasks

### 28.1. Create setup-server.sh with system updates and package installation

**Status:** pending  
**Dependencies:** None  

Create the initial setup script structure with shebang, error handling, color output functions, and implement system update and essential package installation section

**Details:**

Create scripts/deploy/setup-server.sh with executable permissions. Implement: 1) Script header with set -e for error handling, 2) Color output functions (info, warn, error), 3) System updates section using apt-get update and upgrade with quiet flags, 4) Essential packages installation (curl, wget, git, ufw, fail2ban, unattended-upgrades, apt-listchanges, htop, ncdu) using DEBIAN_FRONTEND=noninteractive for non-interactive installation. Ensure all output is properly formatted with timestamps and color-coded status messages.

### 28.2. Implement security hardening (UFW, fail2ban, SSH)

**Status:** pending  
**Dependencies:** 28.1  

Add comprehensive security configuration including firewall rules, intrusion prevention, and SSH hardening to protect the server from unauthorized access

**Details:**

Implement security hardening section: 1) UFW firewall configuration - reset existing rules, set default deny incoming/allow outgoing, allow ports 22 (SSH), 80 (HTTP), 443 (HTTPS) with comments, force enable UFW, 2) fail2ban configuration - create /etc/fail2ban/jail.local with sshd jail enabled (maxretry=3, bantime=86400, findtime=600), restart and enable fail2ban service, 3) SSH hardening - modify /etc/ssh/sshd_config to set PermitRootLogin=prohibit-password and PasswordAuthentication=no using sed, restart sshd service, 4) Automatic security updates - configure unattended-upgrades with security-only updates, auto-fix interrupted dpkg, remove unused dependencies, disable automatic reboot.

### 28.3. Add swap configuration with idempotency checks

**Status:** pending  
**Dependencies:** 28.2  

Implement swap file creation with proper size, permissions, and swappiness settings, ensuring the script can safely run multiple times without recreating swap

**Details:**

Implement swap configuration section with idempotency: 1) Check if /swapfile already exists using conditional [ ! -f /swapfile ], 2) If not exists: create 4GB swap file using fallocate -l 4G, set permissions to 600, run mkswap and swapon, add entry to /etc/fstab for persistence, 3) Configure vm.swappiness=10 in /etc/sysctl.conf for optimal performance (prefer RAM over swap), apply with sysctl -p, 4) If swap exists: skip creation and output info message. Ensure script outputs appropriate messages for both scenarios.

### 28.4. Implement Docker installation with cleanup automation

**Status:** pending  
**Dependencies:** 28.3  

Add Docker installation using official script with idempotency checks, configure automatic cleanup cron job to prevent disk space issues from old images and containers

**Details:**

Implement Docker installation section: 1) Check if Docker is already installed using command -v docker, 2) If not installed: download and run official Docker installation script (curl -fsSL https://get.docker.com | sh), enable and start Docker service using systemctl, 3) Create /etc/cron.daily/docker-cleanup script that runs 'docker system prune -af --volumes --filter until=168h' (removes unused resources older than 7 days), make cleanup script executable (chmod +x), 4) If Docker exists: skip installation and output info message. Ensure Docker daemon is running after installation.

### 28.5. Add Coolify installation and comprehensive summary output

**Status:** pending  
**Dependencies:** 28.4  

Implement Coolify installation with idempotency and create detailed summary section showing security status, resource usage, and next steps for administrator

**Details:**

Implement final sections: 1) Coolify installation - check if /data/coolify directory exists, if not: download and execute Coolify install script (curl -fsSL https://cdn.coollabs.io/coolify/install.sh | bash), output access URL, if exists: skip and inform, 2) Create comprehensive summary output section with: header banner, next steps list (access Coolify at https://<server-ip>:8000, create admin account, configure SSL, set up GitHub webhooks, configure environment variables), security status (UFW rules via 'ufw status | head -10'), resource usage (RAM via 'free -h', disk via 'df -h /'), Docker version. Use color-coded info messages throughout summary.
