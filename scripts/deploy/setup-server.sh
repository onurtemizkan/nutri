#!/bin/bash
set -e

# =============================================================================
# Nutri Production Server Setup Script for Hetzner CX32
# Run as root on fresh Ubuntu 22.04 LTS
#
# Usage: curl -fsSL https://raw.githubusercontent.com/onurtemizkan/nutri/master/scripts/deploy/setup-server.sh | bash
# Or: ./setup-server.sh
# =============================================================================

echo "=== Nutri Production Server Setup ==="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Hostname: $(hostname)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
section() { echo -e "\n${BLUE}=== $1 ===${NC}\n"; }

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    error "This script must be run as root. Use: sudo ./setup-server.sh"
fi

# =============================================================================
# System Updates and Basic Security Packages
# =============================================================================

section "System Updates and Package Installation"

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
    ncdu \
    jq \
    ca-certificates \
    gnupg \
    lsb-release \
    apt-transport-https \
    software-properties-common

info "Essential packages installed successfully"

# =============================================================================
# Security Hardening - Firewall (UFW)
# =============================================================================

section "Security Hardening - Firewall"

info "Configuring UFW firewall..."

# Reset and configure UFW
ufw --force reset > /dev/null 2>&1
ufw default deny incoming
ufw default allow outgoing

# Allow essential ports
ufw allow 22/tcp comment 'SSH'
ufw allow 80/tcp comment 'HTTP'
ufw allow 443/tcp comment 'HTTPS'
ufw allow 8000/tcp comment 'Coolify'

# Enable UFW
ufw --force enable

info "UFW firewall configured and enabled"
ufw status verbose

# =============================================================================
# Security Hardening - Fail2ban
# =============================================================================

section "Security Hardening - Fail2ban"

info "Configuring fail2ban..."

cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
# Ban hosts for one hour
bantime = 3600
# Look back window
findtime = 600
# Max retries before ban
maxretry = 5
# Backend for log monitoring
backend = systemd

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 86400
findtime = 600

[sshd-ddos]
enabled = true
port = ssh
filter = sshd-ddos
logpath = /var/log/auth.log
maxretry = 5
bantime = 48800
EOF

systemctl restart fail2ban
systemctl enable fail2ban

info "Fail2ban configured and running"
fail2ban-client status

# =============================================================================
# Security Hardening - SSH Configuration
# =============================================================================

section "Security Hardening - SSH"

info "Hardening SSH configuration..."

# Backup original config
cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup.$(date +%Y%m%d)

# Apply SSH hardening
sed -i 's/#PermitRootLogin yes/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
sed -i 's/PermitRootLogin yes/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/#MaxAuthTries 6/MaxAuthTries 3/' /etc/ssh/sshd_config
sed -i 's/MaxAuthTries 6/MaxAuthTries 3/' /etc/ssh/sshd_config
sed -i 's/#ClientAliveInterval 0/ClientAliveInterval 300/' /etc/ssh/sshd_config
sed -i 's/#ClientAliveCountMax 3/ClientAliveCountMax 2/' /etc/ssh/sshd_config

# Validate and restart SSH
sshd -t && systemctl restart sshd

info "SSH hardened successfully"
warn "Ensure you have SSH key access before disconnecting!"

# =============================================================================
# Security Hardening - Automatic Security Updates
# =============================================================================

section "Automatic Security Updates"

info "Configuring unattended upgrades..."

# Configure unattended-upgrades
cat > /etc/apt/apt.conf.d/50unattended-upgrades << 'EOF'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}";
    "${distro_id}:${distro_codename}-security";
    "${distro_id}ESMApps:${distro_codename}-apps-security";
    "${distro_id}ESM:${distro_codename}-infra-security";
};

// Remove unused packages
Unattended-Upgrade::Remove-Unused-Dependencies "true";

// Fix interrupted dpkg
Unattended-Upgrade::AutoFixInterruptedDpkg "true";

// Don't auto-reboot (we'll handle this manually)
Unattended-Upgrade::Automatic-Reboot "false";

// Log to syslog
Unattended-Upgrade::SyslogEnable "true";

// Mail notifications (configure SMTP first)
// Unattended-Upgrade::Mail "admin@example.com";
EOF

# Enable automatic updates
cat > /etc/apt/apt.conf.d/20auto-upgrades << 'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
EOF

dpkg-reconfigure -f noninteractive unattended-upgrades

info "Automatic security updates enabled"

# =============================================================================
# Swap Configuration
# =============================================================================

section "Swap Configuration"

SWAP_SIZE="4G"
SWAPPINESS=10

if [ ! -f /swapfile ]; then
    info "Creating ${SWAP_SIZE} swap file..."

    # Create swap file
    fallocate -l ${SWAP_SIZE} /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile

    # Make permanent
    echo '/swapfile none swap sw 0 0' >> /etc/fstab

    # Configure swappiness
    echo "vm.swappiness=${SWAPPINESS}" >> /etc/sysctl.conf
    echo "vm.vfs_cache_pressure=50" >> /etc/sysctl.conf
    sysctl -p > /dev/null 2>&1

    info "Swap file created and configured"
else
    info "Swap file already exists, skipping creation"
fi

info "Current swap status:"
free -h | grep -E "(Mem|Swap)"

# =============================================================================
# Docker Installation
# =============================================================================

section "Docker Installation"

if ! command -v docker &> /dev/null; then
    info "Installing Docker..."

    # Remove old versions
    apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

    # Install using official script
    curl -fsSL https://get.docker.com | sh

    # Enable and start Docker
    systemctl enable docker
    systemctl start docker

    # Add docker group if it doesn't exist
    groupadd docker 2>/dev/null || true

    info "Docker installed successfully"
else
    info "Docker already installed, skipping..."
fi

# Docker version
docker --version

# =============================================================================
# Docker Cleanup Automation
# =============================================================================

section "Docker Cleanup Automation"

info "Configuring Docker cleanup cron job..."

# Create cleanup script
cat > /etc/cron.daily/docker-cleanup << 'EOF'
#!/bin/bash
# Nutri Docker Cleanup Script
# Runs daily to remove unused resources older than 7 days

LOG_FILE="/var/log/docker-cleanup.log"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) - Starting Docker cleanup" >> $LOG_FILE

# Remove stopped containers older than 7 days
docker container prune -f --filter "until=168h" >> $LOG_FILE 2>&1

# Remove unused images older than 7 days
docker image prune -af --filter "until=168h" >> $LOG_FILE 2>&1

# Remove unused networks
docker network prune -f >> $LOG_FILE 2>&1

# Remove unused volumes (be careful - only removes anonymous volumes)
# docker volume prune -f >> $LOG_FILE 2>&1

# Show disk usage after cleanup
echo "Disk usage after cleanup:" >> $LOG_FILE
docker system df >> $LOG_FILE 2>&1

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) - Docker cleanup complete" >> $LOG_FILE
EOF

chmod +x /etc/cron.daily/docker-cleanup

info "Docker cleanup automation configured"

# =============================================================================
# Docker Compose Installation
# =============================================================================

section "Docker Compose"

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
    info "Docker Compose is included with Docker Engine"
fi

# Verify docker compose works
docker compose version

# =============================================================================
# Coolify Installation
# =============================================================================

section "Coolify Installation"

if [ ! -d /data/coolify ]; then
    info "Installing Coolify..."

    # Run Coolify installer
    curl -fsSL https://cdn.coollabs.io/coolify/install.sh | bash

    info "Coolify installed!"
else
    info "Coolify already installed, skipping..."
fi

# =============================================================================
# System Optimizations
# =============================================================================

section "System Optimizations"

info "Applying system optimizations..."

# Network optimizations
cat >> /etc/sysctl.conf << 'EOF'

# Network performance optimizations
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_keepalive_time = 300
net.ipv4.tcp_keepalive_probes = 5
net.ipv4.tcp_keepalive_intvl = 15

# File descriptor limits
fs.file-max = 2097152
EOF

# Apply sysctl changes
sysctl -p > /dev/null 2>&1

# Increase file descriptor limits
cat >> /etc/security/limits.conf << 'EOF'

# Nutri production limits
* soft nofile 65535
* hard nofile 65535
root soft nofile 65535
root hard nofile 65535
EOF

info "System optimizations applied"

# =============================================================================
# Setup Log Rotation
# =============================================================================

section "Log Rotation"

info "Configuring log rotation..."

cat > /etc/logrotate.d/nutri << 'EOF'
/var/log/docker-cleanup.log {
    weekly
    rotate 4
    compress
    delaycompress
    missingok
    notifempty
    create 0640 root root
}

/data/coolify/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF

info "Log rotation configured"

# =============================================================================
# Summary and Next Steps
# =============================================================================

section "Setup Complete"

SERVER_IP=$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')

echo ""
echo -e "${GREEN}=== Nutri Production Server Ready ===${NC}"
echo ""
info "Server IP: $SERVER_IP"
echo ""

info "Security Status:"
echo "  - UFW Firewall: $(ufw status | head -1)"
echo "  - Fail2ban: $(systemctl is-active fail2ban)"
echo "  - SSH: Key-only auth, root login via key only"
echo "  - Auto-updates: Enabled"
echo ""

info "Resources:"
echo "  Memory:"
free -h | grep -E "(Mem|Swap)" | sed 's/^/    /'
echo ""
echo "  Disk:"
df -h / | tail -1 | awk '{print "    Used: " $3 " / " $2 " (" $5 " used)"}'
echo ""

info "Docker:"
docker --version | sed 's/^/  /'
docker compose version | sed 's/^/  /'
echo ""

info "Next Steps:"
echo "  1. Access Coolify at https://${SERVER_IP}:8000"
echo "  2. Create admin account and configure SSL"
echo "  3. Add GitHub source for deployments"
echo "  4. Set up services: Backend, ML Service"
echo "  5. Configure environment variables:"
echo "     - DATABASE_URL (Supabase PostgreSQL)"
echo "     - REDIS_URL (Upstash Redis)"
echo "     - JWT_SECRET"
echo "     - Other secrets from .env.prod.example"
echo "  6. Set up GitHub webhook integration"
echo "  7. Test deployments"
echo ""

info "Documentation: https://github.com/onurtemizkan/nutri/tree/master/docs/deployment"
echo ""

warn "Remember to:"
echo "  - Save your SSH key backup"
echo "  - Document the server IP in your secrets"
echo "  - Set up monitoring (recommended: UptimeRobot free tier)"
echo ""

echo "=== Setup completed at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
