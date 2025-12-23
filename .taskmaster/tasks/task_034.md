# Task ID: 34

**Title:** Configure Server Security Hardening

**Status:** pending

**Dependencies:** 28 ✓

**Priority:** medium

**Description:** Implement additional security hardening measures including SSH hardening, intrusion detection setup, and kernel security parameters.

**Details:**

**Create `scripts/deploy/harden-server.sh`:**
```bash
#!/bin/bash
set -e

# =============================================================================
# Nutri Server Security Hardening
# Run after initial setup script
# =============================================================================

echo "=== Security Hardening ==="

# =============================================================================
# SSH Hardening
# =============================================================================

echo "Hardening SSH configuration..."
cat >> /etc/ssh/sshd_config << 'EOF'

# Nutri Security Hardening
Protocol 2
MaxAuthTries 3
MaxSessions 5
LoginGraceTime 30
ClientAliveInterval 300
ClientAliveCountMax 2
PermitEmptyPasswords no
X11Forwarding no
AllowTcpForwarding no
AllowAgentForwarding no
PermitUserEnvironment no
EOF

# Restart SSH
systemctl restart sshd
echo "SSH hardened"

# =============================================================================
# Kernel Security Parameters
# =============================================================================

echo "Configuring kernel security parameters..."
cat >> /etc/sysctl.conf << 'EOF'

# Nutri Security Hardening

# Network security
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.tcp_synack_retries = 2
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1

# Disable IPv6 if not needed
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1

# Memory protection
kernel.randomize_va_space = 2
kernel.dmesg_restrict = 1
kernel.kptr_restrict = 2
EOF

sysctl -p
echo "Kernel parameters configured"

# =============================================================================
# Advanced Fail2Ban Configuration
# =============================================================================

echo "Configuring advanced fail2ban rules..."
cat > /etc/fail2ban/jail.d/nutri.conf << 'EOF'
[nutri-api]
enabled = true
port = http,https
filter = nutri-api
logpath = /var/log/docker-nutri-backend.log
maxretry = 10
findtime = 60
bantime = 3600
EOF

cat > /etc/fail2ban/filter.d/nutri-api.conf << 'EOF'
[Definition]
failregex = ^.*"statusCode":401.*"ip":"<HOST>".*$
            ^.*"statusCode":403.*"ip":"<HOST>".*$
ignoreregex =
EOF

systemctl restart fail2ban
echo "Fail2ban configured for API protection"

# =============================================================================
# Docker Security
# =============================================================================

echo "Configuring Docker security..."
cat > /etc/docker/daemon.json << 'EOF'
{
  "icc": false,
  "userns-remap": "default",
  "no-new-privileges": true,
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "live-restore": true
}
EOF

systemctl restart docker
echo "Docker security configured"

# =============================================================================
# File Permissions
# =============================================================================

echo "Securing file permissions..."
chmod 700 /root
chmod 600 /etc/ssh/sshd_config
chmod 644 /etc/passwd
chmod 000 /etc/shadow  # Only root can read
chmod 644 /etc/group

# =============================================================================
# Login Banner
# =============================================================================

cat > /etc/issue.net << 'EOF'
***************************************************************************
                      AUTHORIZED ACCESS ONLY
***************************************************************************
This system is for authorized use only. All activities may be monitored
and recorded. Unauthorized access will be prosecuted.
***************************************************************************
EOF

echo "Banner /etc/issue.net" >> /etc/ssh/sshd_config
systemctl restart sshd

echo ""
echo "=== Security Hardening Complete ==="
echo ""
echo "Verify changes:"
echo "- SSH: ssh to server, should still work"
echo "- Fail2ban: fail2ban-client status"
echo "- Sysctl: sysctl -a | grep net.ipv4.tcp_syncookies"
```

**Security Checklist `docs/deployment/SECURITY-CHECKLIST.md`:**
```markdown
# Security Checklist

## Pre-Deployment

- [ ] SSH keys configured, password auth disabled
- [ ] UFW firewall enabled (22, 80, 443 only)
- [ ] Fail2ban active and configured
- [ ] Automatic security updates enabled
- [ ] Kernel security parameters applied
- [ ] Docker security settings configured

## Application Security

- [ ] All secrets stored in environment variables
- [ ] No secrets in source code or logs
- [ ] HTTPS enforced (redirect HTTP)
- [ ] Security headers configured (HSTS, CSP)
- [ ] Rate limiting enabled
- [ ] CORS properly configured

## Monitoring

- [ ] Failed SSH attempts monitored
- [ ] API abuse patterns monitored
- [ ] Error tracking enabled (Sentry)
- [ ] Uptime monitoring configured

## Backup & Recovery

- [ ] Database backups automated
- [ ] Server snapshots enabled
- [ ] Recovery procedure tested
```

**Test Strategy:**

1. Run hardening script on test server
2. Verify SSH still works with key auth
3. Verify password auth fails
4. Test fail2ban bans IP after failed attempts
5. Verify kernel parameters with sysctl -a
6. Test Docker containers still run
7. Run security scanner (lynis) and compare scores
8. Document any issues for production
