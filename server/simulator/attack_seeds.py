"""
Deterministic attack scenarios with full ground truth.
Each scenario is seeded so results are 100% reproducible.
These define what actually happened — graders compare agent actions against this.
"""
from __future__ import annotations
from typing import Dict, Any, List


# ---------------------------------------------------------------------------
# SCENARIO BANK — used across all 3 tasks
# ---------------------------------------------------------------------------

ATTACK_SCENARIOS: Dict[str, Dict[str, Any]] = {

    # -----------------------------------------------------------------------
    # SCENARIO 1: Brute-force SSH followed by successful login
    # Used in: Task 1 (alert triage), Task 2 (incident investigation)
    # -----------------------------------------------------------------------
    "brute_force_ssh_001": {
        "scenario_id": "brute_force_ssh_001",
        "attack_type": "brute_force",
        "attacker_ip": "185.220.101.47",
        "target_hostname": "WEB-SERVER-01",
        "target_ip": "10.0.1.20",
        "compromised_user": "admin",
        "kill_chain": ["reconnaissance", "brute_force", "initial_access"],
        "ground_truth": {
            "classification": "critical",
            "priority": "P1",
            "verdict": "true_positive",
            "attack_type": "brute_force",
            "key_evidence": [
                "auth_log_failed_logins",
                "auth_log_success_after_failures",
                "firewall_external_ssh",
            ],
            "affected_assets": ["WEB-SERVER-01"],
            "attacker_ips": ["185.220.101.47"],
            "mitre_techniques": [
                {"technique_id": "T1110.001", "name": "Password Guessing", "tactic": "Credential Access"},
                {"technique_id": "T1078", "name": "Valid Accounts", "tactic": "Initial Access"},
            ],
        },
        "alerts": [
            {
                "alert_id": "ALT-001",
                "alert_type": "Multiple Failed SSH Logins",
                "severity": "high",
                "timestamp": "2024-01-15T02:14:33Z",
                "source_ip": "185.220.101.47",
                "dest_ip": "10.0.1.20",
                "dest_port": 22,
                "description": "47 failed SSH login attempts in 90 seconds from external IP",
                "raw_data": {"attempts": 47, "window_seconds": 90, "protocol": "SSH"},
            },
            {
                "alert_id": "ALT-002",
                "alert_type": "Successful Login After Brute Force",
                "severity": "critical",
                "timestamp": "2024-01-15T02:16:01Z",
                "source_ip": "185.220.101.47",
                "dest_ip": "10.0.1.20",
                "dest_port": 22,
                "description": "Successful SSH login from same IP responsible for failed attempts",
                "raw_data": {"user": "admin", "auth_method": "password"},
            },
        ],
        "logs": {
            "auth": [
                {
                    "log_id": "AUTH-001",
                    "log_source": "auth",
                    "timestamp": "2024-01-15T02:14:33Z",
                    "source_ip": "185.220.101.47",
                    "user": "admin",
                    "event_type": "authentication_failure",
                    "details": {"reason": "invalid_password", "attempt_count": 47},
                    "is_malicious": True,
                },
                {
                    "log_id": "AUTH-002",
                    "log_source": "auth",
                    "timestamp": "2024-01-15T02:16:01Z",
                    "source_ip": "185.220.101.47",
                    "user": "admin",
                    "event_type": "authentication_success",
                    "details": {"session_id": "ssh-9f3a2b", "shell": "/bin/bash"},
                    "is_malicious": True,
                },
            ],
            "firewall": [
                {
                    "log_id": "FW-001",
                    "log_source": "firewall",
                    "timestamp": "2024-01-15T02:14:30Z",
                    "source_ip": "185.220.101.47",
                    "dest_ip": "10.0.1.20",
                    "event_type": "connection_allowed",
                    "details": {"port": 22, "protocol": "TCP", "rule": "ALLOW_SSH_INBOUND"},
                    "is_malicious": True,
                },
            ],
            "process": [],
            "dns": [],
            "network": [],
        },
    },

    # -----------------------------------------------------------------------
    # SCENARIO 2: Phishing → credential theft → lateral movement
    # Used in: Task 2 (incident investigation), Task 3 (threat response)
    # -----------------------------------------------------------------------
    "phishing_lateral_001": {
        "scenario_id": "phishing_lateral_001",
        "attack_type": "lateral_movement",
        "attacker_ip": "94.102.49.190",
        "initial_victim": "WS-HR-03",
        "lateral_target": "DC-01",
        "compromised_users": ["jsmith", "domain_admin"],
        "kill_chain": [
            "phishing",
            "credential_theft",
            "lateral_movement",
            "privilege_escalation",
        ],
        "ground_truth": {
            "classification": "critical",
            "priority": "P1",
            "verdict": "true_positive",
            "attack_type": "lateral_movement",
            "key_evidence": [
                "email_phishing_link",
                "process_powershell_encoded",
                "auth_lateral_movement",
                "network_c2_beacon",
            ],
            "affected_assets": ["WS-HR-03", "DC-01"],
            "attacker_ips": ["94.102.49.190", "10.0.2.15"],
            "mitre_techniques": [
                {"technique_id": "T1566.001", "name": "Spearphishing Attachment", "tactic": "Initial Access"},
                {"technique_id": "T1059.001", "name": "PowerShell", "tactic": "Execution"},
                {"technique_id": "T1071.001", "name": "Web Protocols", "tactic": "Command and Control"},
                {"technique_id": "T1550.002", "name": "Pass the Ticket", "tactic": "Lateral Movement"},
                {"technique_id": "T1078.002", "name": "Domain Accounts", "tactic": "Privilege Escalation"},
            ],
        },
        "alerts": [
            {
                "alert_id": "ALT-010",
                "alert_type": "Suspicious PowerShell Execution",
                "severity": "high",
                "timestamp": "2024-01-16T09:45:12Z",
                "source_ip": "10.0.2.15",
                "dest_ip": None,
                "dest_port": None,
                "description": "Encoded PowerShell command executed by non-admin user on HR workstation",
                "raw_data": {
                    "hostname": "WS-HR-03",
                    "user": "jsmith",
                    "command_length": 2048,
                    "encoded": True,
                },
            },
            {
                "alert_id": "ALT-011",
                "alert_type": "Unusual Outbound Connection",
                "severity": "medium",
                "timestamp": "2024-01-16T09:47:30Z",
                "source_ip": "10.0.2.15",
                "dest_ip": "94.102.49.190",
                "dest_port": 443,
                "description": "Workstation connecting to known C2 IP over HTTPS",
                "raw_data": {"bytes_sent": 48200, "duration_seconds": 312},
            },
            {
                "alert_id": "ALT-012",
                "alert_type": "Admin Login from Workstation",
                "severity": "critical",
                "timestamp": "2024-01-16T10:02:44Z",
                "source_ip": "10.0.2.15",
                "dest_ip": "10.0.0.5",
                "dest_port": 445,
                "description": "Domain admin credentials used from HR workstation to access Domain Controller",
                "raw_data": {"user": "domain_admin", "target": "DC-01", "protocol": "SMB"},
            },
        ],
        "logs": {
            "process": [
                {
                    "log_id": "PROC-001",
                    "log_source": "process",
                    "timestamp": "2024-01-16T09:45:12Z",
                    "source_ip": "10.0.2.15",
                    "user": "jsmith",
                    "event_type": "process_created",
                    "details": {
                        "process": "powershell.exe",
                        "parent": "outlook.exe",
                        "args": "-EncodedCommand JABzAD0ATgBlAHcA...",
                        "hostname": "WS-HR-03",
                    },
                    "is_malicious": True,
                },
            ],
            "network": [
                {
                    "log_id": "NET-001",
                    "log_source": "network",
                    "timestamp": "2024-01-16T09:47:30Z",
                    "source_ip": "10.0.2.15",
                    "dest_ip": "94.102.49.190",
                    "event_type": "outbound_connection",
                    "details": {"port": 443, "bytes": 48200, "threat_intel": "known_c2"},
                    "is_malicious": True,
                },
            ],
            "auth": [
                {
                    "log_id": "AUTH-010",
                    "log_source": "auth",
                    "timestamp": "2024-01-16T10:02:44Z",
                    "source_ip": "10.0.2.15",
                    "user": "domain_admin",
                    "event_type": "authentication_success",
                    "details": {"target_host": "DC-01", "protocol": "Kerberos", "anomaly": "unusual_source"},
                    "is_malicious": True,
                },
            ],
            "firewall": [
                {
                    "log_id": "FW-010",
                    "log_source": "firewall",
                    "timestamp": "2024-01-16T09:47:25Z",
                    "source_ip": "10.0.2.15",
                    "dest_ip": "94.102.49.190",
                    "event_type": "connection_allowed",
                    "details": {"port": 443, "direction": "outbound"},
                    "is_malicious": True,
                },
            ],
            "dns": [
                {
                    "log_id": "DNS-001",
                    "log_source": "dns",
                    "timestamp": "2024-01-16T09:47:20Z",
                    "source_ip": "10.0.2.15",
                    "event_type": "dns_query",
                    "details": {"query": "update.microsoft-cdn.net", "resolved_ip": "94.102.49.190", "threat_intel": "domain_spoofing"},
                    "is_malicious": True,
                },
            ],
        },
    },

    # -----------------------------------------------------------------------
    # SCENARIO 3: False positive — port scan from internal IT scanner
    # Used in: Task 1 (alert triage)
    # -----------------------------------------------------------------------
    "false_positive_scan_001": {
        "scenario_id": "false_positive_scan_001",
        "attack_type": "false_positive",
        "ground_truth": {
            "classification": "benign",
            "priority": "P4",
            "verdict": "false_positive",
            "attack_type": "false_positive",
            "key_evidence": ["known_scanner_ip", "scheduled_scan_window"],
            "affected_assets": [],
            "attacker_ips": [],
            "mitre_techniques": [],
        },
        "alerts": [
            {
                "alert_id": "ALT-020",
                "alert_type": "Port Scan Detected",
                "severity": "medium",
                "timestamp": "2024-01-17T03:00:10Z",
                "source_ip": "10.0.0.100",
                "dest_ip": "10.0.0.0/24",
                "dest_port": None,
                "description": "Internal IP scanning full subnet on common service ports",
                "raw_data": {"ports_scanned": [22, 80, 443, 3389, 8080], "scanner": "nmap"},
            },
        ],
        "logs": {
            "firewall": [
                {
                    "log_id": "FW-020",
                    "log_source": "firewall",
                    "timestamp": "2024-01-17T03:00:10Z",
                    "source_ip": "10.0.0.100",
                    "dest_ip": "10.0.0.0/24",
                    "event_type": "port_scan",
                    "details": {
                        "scanner_hostname": "IT-SCANNER-01",
                        "authorized": True,
                        "schedule": "weekly_sunday_0300",
                    },
                    "is_malicious": False,
                },
            ],
            "auth": [], "process": [], "dns": [], "network": [],
        },
    },

    # -----------------------------------------------------------------------
    # SCENARIO 4: Ransomware — file encryption + SMB lateral spread
    # Used in: Task 2 variant, Task 3 variant
    # -----------------------------------------------------------------------
    "ransomware_001": {
        "scenario_id": "ransomware_001",
        "attack_type": "malware",
        "attacker_ip": "192.168.50.99",
        "target_hostname": "WS-FINANCE-01",
        "target_ip": "10.0.2.50",
        "compromised_user": "bwalker",
        "kill_chain": ["initial_access", "execution", "impact"],
        "ground_truth": {
            "classification": "critical",
            "priority": "P1",
            "verdict": "true_positive",
            "attack_type": "malware",
            "key_evidence": ["PROC-R01", "PROC-R02", "NET-R01"],
            "key_evidence_sources": ["process", "network", "firewall"],
            "affected_assets": ["WS-FINANCE-01"],
            "attacker_ips": ["192.168.50.99"],
            "mitre_techniques": [
                {"technique_id": "T1486", "name": "Data Encrypted for Impact", "tactic": "Impact"},
                {"technique_id": "T1490", "name": "Inhibit System Recovery", "tactic": "Impact"},
                {"technique_id": "T1021.002", "name": "SMB/Windows Admin Shares", "tactic": "Lateral Movement"},
            ],
        },
        "alerts": [
            {
                "alert_id": "ALT-040",
                "alert_type": "Shadow Copy Deletion",
                "severity": "critical",
                "timestamp": "2024-01-18T04:12:08Z",
                "source_ip": "10.0.2.50",
                "dest_ip": None,
                "dest_port": None,
                "description": "vssadmin.exe used to delete all volume shadow copies on finance workstation",
                "raw_data": {"hostname": "WS-FINANCE-01", "command": "vssadmin delete shadows /all /quiet", "user": "bwalker"},
            },
            {
                "alert_id": "ALT-041",
                "alert_type": "Ransomware File Encryption",
                "severity": "critical",
                "timestamp": "2024-01-18T04:13:45Z",
                "source_ip": "10.0.2.50",
                "dest_ip": None,
                "dest_port": None,
                "description": "Unknown executable writing .encrypted extensions to C:\\Finance\\ directory",
                "raw_data": {"hostname": "WS-FINANCE-01", "files_affected": 347, "extension": ".encrypted"},
            },
            {
                "alert_id": "ALT-042",
                "alert_type": "SMB Lateral Spread Attempt",
                "severity": "high",
                "timestamp": "2024-01-18T04:15:22Z",
                "source_ip": "10.0.2.50",
                "dest_ip": "10.0.2.0/24",
                "dest_port": 445,
                "description": "SMB broadcast packets from finance workstation attempting to spread to subnet",
                "raw_data": {"protocol": "SMB", "direction": "broadcast", "targets_attempted": 12},
            },
        ],
        "logs": {
            "process": [
                {
                    "log_id": "PROC-R01",
                    "log_source": "process",
                    "timestamp": "2024-01-18T04:12:08Z",
                    "source_ip": "10.0.2.50",
                    "user": "bwalker",
                    "event_type": "process_created",
                    "details": {
                        "process": "vssadmin.exe",
                        "parent": "cmd.exe",
                        "args": "delete shadows /all /quiet",
                        "hostname": "WS-FINANCE-01",
                    },
                    "is_malicious": True,
                },
                {
                    "log_id": "PROC-R02",
                    "log_source": "process",
                    "timestamp": "2024-01-18T04:13:30Z",
                    "source_ip": "10.0.2.50",
                    "user": "SYSTEM",
                    "event_type": "process_created",
                    "details": {
                        "process": "svchost_update.exe",
                        "parent": "cmd.exe",
                        "args": "--encrypt C:\\Finance\\ --ext .encrypted",
                        "hostname": "WS-FINANCE-01",
                        "signature": "unsigned",
                    },
                    "is_malicious": True,
                },
            ],
            "network": [
                {
                    "log_id": "NET-R01",
                    "log_source": "network",
                    "timestamp": "2024-01-18T04:15:22Z",
                    "source_ip": "10.0.2.50",
                    "dest_ip": "10.0.2.255",
                    "event_type": "smb_broadcast",
                    "details": {"port": 445, "protocol": "SMB", "payload_hash": "a3f2c1..."},
                    "is_malicious": True,
                },
            ],
            "firewall": [
                {
                    "log_id": "FW-R01",
                    "log_source": "firewall",
                    "timestamp": "2024-01-18T04:10:00Z",
                    "source_ip": "192.168.50.99",
                    "dest_ip": "10.0.2.50",
                    "event_type": "connection_allowed",
                    "details": {"port": 443, "protocol": "HTTPS", "direction": "inbound"},
                    "is_malicious": True,
                },
            ],
            "auth": [
                {
                    "log_id": "AUTH-R01",
                    "log_source": "auth",
                    "timestamp": "2024-01-18T04:11:55Z",
                    "source_ip": "10.0.2.50",
                    "user": "bwalker",
                    "event_type": "authentication_success",
                    "details": {"method": "local_login", "hostname": "WS-FINANCE-01"},
                    "is_malicious": False,
                },
            ],
            "dns": [],
        },
    },

    # -----------------------------------------------------------------------
    # SCENARIO 5: Insider threat — authorised user exfiltrating data
    # Used in: Task 2 variant
    # -----------------------------------------------------------------------
    "insider_threat_001": {
        "scenario_id": "insider_threat_001",
        "attack_type": "data_exfiltration",
        "attacker_ip": None,
        "compromised_user": "bwalker",
        "target_hostname": "WS-FINANCE-01",
        "target_ip": "10.0.2.50",
        "kill_chain": ["collection", "exfiltration"],
        "ground_truth": {
            "classification": "suspicious",
            "priority": "P2",
            "verdict": "true_positive",
            "attack_type": "data_exfiltration",
            "key_evidence": ["NET-I01", "AUTH-I01", "DNS-I01"],
            "key_evidence_sources": ["network", "auth", "dns"],
            "affected_assets": ["WS-FINANCE-01"],
            "attacker_ips": [],
            "mitre_techniques": [
                {"technique_id": "T1074.001", "name": "Local Data Staging", "tactic": "Collection"},
                {"technique_id": "T1567.002", "name": "Exfiltration to Cloud Storage", "tactic": "Exfiltration"},
            ],
        },
        "alerts": [
            {
                "alert_id": "ALT-050",
                "alert_type": "After-Hours Bulk File Access",
                "severity": "medium",
                "timestamp": "2024-01-19T02:14:00Z",
                "source_ip": "10.0.2.50",
                "dest_ip": "10.0.0.30",
                "dest_port": 1433,
                "description": "Finance user accessing 240+ database records at 2:14 AM outside business hours",
                "raw_data": {"user": "bwalker", "records_accessed": 243, "database": "DB-FINANCE-01"},
            },
            {
                "alert_id": "ALT-051",
                "alert_type": "Large Outbound Transfer to Cloud Storage",
                "severity": "high",
                "timestamp": "2024-01-19T02:32:10Z",
                "source_ip": "10.0.2.50",
                "dest_ip": "104.16.85.20",
                "dest_port": 443,
                "description": "1.8GB upload to personal cloud storage domain from finance workstation",
                "raw_data": {"user": "bwalker", "bytes_sent": 1932735283, "domain": "my-personal-drive.cloud"},
            },
        ],
        "logs": {
            "auth": [
                {
                    "log_id": "AUTH-I01",
                    "log_source": "auth",
                    "timestamp": "2024-01-19T02:10:00Z",
                    "source_ip": "10.0.2.50",
                    "user": "bwalker",
                    "event_type": "authentication_success",
                    "details": {"method": "badge_swipe", "location": "Building A", "time_anomaly": "outside_business_hours"},
                    "is_malicious": True,
                },
            ],
            "network": [
                {
                    "log_id": "NET-I01",
                    "log_source": "network",
                    "timestamp": "2024-01-19T02:32:10Z",
                    "source_ip": "10.0.2.50",
                    "dest_ip": "104.16.85.20",
                    "event_type": "outbound_connection",
                    "details": {"port": 443, "bytes_sent": 1932735283, "domain": "my-personal-drive.cloud", "anomaly": "bulk_upload"},
                    "is_malicious": True,
                },
            ],
            "dns": [
                {
                    "log_id": "DNS-I01",
                    "log_source": "dns",
                    "timestamp": "2024-01-19T02:31:55Z",
                    "source_ip": "10.0.2.50",
                    "event_type": "dns_query",
                    "details": {"query": "my-personal-drive.cloud", "resolved_ip": "104.16.85.20", "category": "personal_cloud_storage"},
                    "is_malicious": True,
                },
            ],
            "firewall": [
                {
                    "log_id": "FW-I01",
                    "log_source": "firewall",
                    "timestamp": "2024-01-19T02:32:08Z",
                    "source_ip": "10.0.2.50",
                    "dest_ip": "104.16.85.20",
                    "event_type": "connection_allowed",
                    "details": {"port": 443, "direction": "outbound", "bytes": 1932735283},
                    "is_malicious": True,
                },
            ],
            "process": [],
        },
    },

    # -----------------------------------------------------------------------
    # SCENARIO 6: Supply chain — trusted update installs backdoor
    # Used in: Adaptive difficulty Tier 2
    # -----------------------------------------------------------------------
    "supply_chain_001": {
        "scenario_id": "supply_chain_001",
        "attack_type": "malware",
        "attacker_ip": "203.0.113.42",
        "target_hostname": "WEB-SERVER-01",
        "target_ip": "10.0.1.20",
        "compromised_user": None,
        "kill_chain": ["initial_access", "persistence", "command_and_control"],
        "ground_truth": {
            "classification": "critical",
            "priority": "P1",
            "verdict": "true_positive",
            "attack_type": "malware",
            "key_evidence": ["PROC-S01", "NET-S01", "DNS-S01"],
            "key_evidence_sources": ["process", "network", "dns"],
            "affected_assets": ["WEB-SERVER-01"],
            "attacker_ips": ["203.0.113.42"],
            "mitre_techniques": [
                {"technique_id": "T1195.002", "name": "Compromise Software Supply Chain", "tactic": "Initial Access"},
                {"technique_id": "T1543.002", "name": "Systemd Service", "tactic": "Persistence"},
                {"technique_id": "T1071.001", "name": "Web Protocols", "tactic": "Command and Control"},
            ],
        },
        "alerts": [
            {
                "alert_id": "ALT-060",
                "alert_type": "Software Update from Unusual Source",
                "severity": "low",
                "timestamp": "2024-01-20T10:05:00Z",
                "source_ip": "10.0.1.20",
                "dest_ip": "203.0.113.42",
                "dest_port": 443,
                "description": "Scheduled software update downloaded from non-standard CDN mirror",
                "raw_data": {"package": "monitoring-agent-3.2.1.deb", "expected_hash": "a1b2c3", "actual_hash": "x9y8z7"},
            },
            {
                "alert_id": "ALT-061",
                "alert_type": "New Persistent Service Registered",
                "severity": "low",
                "timestamp": "2024-01-20T10:06:30Z",
                "source_ip": "10.0.1.20",
                "dest_ip": None,
                "dest_port": None,
                "description": "New systemd service 'monitoring-helper' registered with auto-start",
                "raw_data": {"service": "monitoring-helper", "binary": "/opt/monitoring/helper.bin", "auto_start": True},
            },
            {
                "alert_id": "ALT-062",
                "alert_type": "Low-Volume Periodic Outbound Connection",
                "severity": "low",
                "timestamp": "2024-01-20T10:15:00Z",
                "source_ip": "10.0.1.20",
                "dest_ip": "203.0.113.42",
                "dest_port": 8443,
                "description": "Small periodic HTTPS beacons every 300 seconds to external IP",
                "raw_data": {"interval_seconds": 300, "bytes_per_beacon": 128, "duration_observed": "2h"},
            },
        ],
        "logs": {
            "process": [
                {
                    "log_id": "PROC-S01",
                    "log_source": "process",
                    "timestamp": "2024-01-20T10:06:30Z",
                    "source_ip": "10.0.1.20",
                    "user": "root",
                    "event_type": "process_created",
                    "details": {
                        "process": "/opt/monitoring/helper.bin",
                        "parent": "apt-get",
                        "args": "--daemon --beacon-interval 300",
                        "hostname": "WEB-SERVER-01",
                        "signature": "unsigned",
                    },
                    "is_malicious": True,
                },
            ],
            "network": [
                {
                    "log_id": "NET-S01",
                    "log_source": "network",
                    "timestamp": "2024-01-20T10:15:00Z",
                    "source_ip": "10.0.1.20",
                    "dest_ip": "203.0.113.42",
                    "event_type": "outbound_connection",
                    "details": {"port": 8443, "bytes": 128, "interval": "periodic_300s", "threat_intel": "unknown"},
                    "is_malicious": True,
                },
            ],
            "dns": [
                {
                    "log_id": "DNS-S01",
                    "log_source": "dns",
                    "timestamp": "2024-01-20T10:14:55Z",
                    "source_ip": "10.0.1.20",
                    "event_type": "dns_query",
                    "details": {"query": "updates.monitoring-cdn.net", "resolved_ip": "203.0.113.42", "category": "uncategorized"},
                    "is_malicious": True,
                },
            ],
            "auth": [],
            "firewall": [
                {
                    "log_id": "FW-S01",
                    "log_source": "firewall",
                    "timestamp": "2024-01-20T10:05:00Z",
                    "source_ip": "10.0.1.20",
                    "dest_ip": "203.0.113.42",
                    "event_type": "connection_allowed",
                    "details": {"port": 443, "direction": "outbound"},
                    "is_malicious": True,
                },
            ],
        },
    },

    # -----------------------------------------------------------------------
    # SCENARIO 7: Multi-stage APT — 6-stage kill chain with decoy IPs
    # Used in: Adaptive difficulty Tier 3
    # -----------------------------------------------------------------------
    "multi_stage_apt_001": {
        "scenario_id": "multi_stage_apt_001",
        "attack_type": "lateral_movement",
        "attacker_ip": "45.33.32.156",
        "decoy_ips": ["198.51.100.10", "198.51.100.20"],
        "target_hostname": "DC-01",
        "target_ip": "10.0.0.5",
        "compromised_user": "svc_backup",
        "kill_chain": [
            "reconnaissance", "initial_access", "execution",
            "persistence", "lateral_movement", "privilege_escalation",
        ],
        "ground_truth": {
            "classification": "critical",
            "priority": "P1",
            "verdict": "true_positive",
            "attack_type": "lateral_movement",
            "key_evidence": ["AUTH-A01", "AUTH-A02", "PROC-A01", "NET-A01", "DNS-A01"],
            "key_evidence_sources": ["auth", "process", "network", "dns", "firewall"],
            "affected_assets": ["BACKUP-SRV-01", "DC-01"],
            "attacker_ips": ["45.33.32.156"],
            "mitre_techniques": [
                {"technique_id": "T1078.001", "name": "Default Accounts", "tactic": "Initial Access"},
                {"technique_id": "T1134.001", "name": "Token Impersonation", "tactic": "Privilege Escalation"},
                {"technique_id": "T1053.005", "name": "Scheduled Task", "tactic": "Persistence"},
                {"technique_id": "T1572", "name": "Protocol Tunneling", "tactic": "Command and Control"},
                {"technique_id": "T1021.002", "name": "SMB/Windows Admin Shares", "tactic": "Lateral Movement"},
            ],
        },
        "alerts": [
            {
                "alert_id": "ALT-070",
                "alert_type": "Unusual Service Account Activity",
                "severity": "medium",
                "timestamp": "2024-01-21T01:20:00Z",
                "source_ip": "10.0.0.20",
                "dest_ip": "10.0.0.5",
                "dest_port": 445,
                "description": "Service account svc_backup authenticating to Domain Controller outside maintenance window",
                "raw_data": {"user": "svc_backup", "source_host": "BACKUP-SRV-01", "target": "DC-01"},
            },
            {
                "alert_id": "ALT-071",
                "alert_type": "Suspicious Scheduled Task Creation",
                "severity": "high",
                "timestamp": "2024-01-21T01:25:00Z",
                "source_ip": "10.0.0.5",
                "dest_ip": None,
                "dest_port": None,
                "description": "New scheduled task created on DC-01 to run encoded PowerShell at boot",
                "raw_data": {"hostname": "DC-01", "task_name": "WindowsHealthCheck", "trigger": "at_startup"},
            },
            {
                "alert_id": "ALT-072",
                "alert_type": "DNS Tunneling Suspected",
                "severity": "medium",
                "timestamp": "2024-01-21T01:30:00Z",
                "source_ip": "10.0.0.5",
                "dest_ip": None,
                "dest_port": 53,
                "description": "High-entropy TXT record queries to unusual subdomain pattern from DC-01",
                "raw_data": {"query_count": 84, "avg_subdomain_length": 48, "domain": "data.ns1.apt-infra.net"},
            },
        ],
        "logs": {
            "auth": [
                {
                    "log_id": "AUTH-A01",
                    "log_source": "auth",
                    "timestamp": "2024-01-21T01:15:00Z",
                    "source_ip": "10.0.0.20",
                    "user": "svc_backup",
                    "event_type": "authentication_success",
                    "details": {"target_host": "DC-01", "protocol": "NTLM", "anomaly": "outside_maintenance_window"},
                    "is_malicious": True,
                },
                {
                    "log_id": "AUTH-A02",
                    "log_source": "auth",
                    "timestamp": "2024-01-21T01:22:00Z",
                    "source_ip": "10.0.0.5",
                    "user": "svc_backup",
                    "event_type": "privilege_escalation",
                    "details": {"method": "token_impersonation", "target_user": "DOMAIN\\Administrator"},
                    "is_malicious": True,
                },
            ],
            "process": [
                {
                    "log_id": "PROC-A01",
                    "log_source": "process",
                    "timestamp": "2024-01-21T01:25:00Z",
                    "source_ip": "10.0.0.5",
                    "user": "SYSTEM",
                    "event_type": "process_created",
                    "details": {
                        "process": "schtasks.exe",
                        "parent": "powershell.exe",
                        "args": "/create /tn WindowsHealthCheck /sc ONSTART /tr powershell.exe -enc ...",
                        "hostname": "DC-01",
                    },
                    "is_malicious": True,
                },
            ],
            "network": [
                {
                    "log_id": "NET-A01",
                    "log_source": "network",
                    "timestamp": "2024-01-21T01:30:00Z",
                    "source_ip": "10.0.0.5",
                    "dest_ip": "45.33.32.156",
                    "event_type": "outbound_connection",
                    "details": {"port": 53, "protocol": "DNS", "bytes": 42000, "anomaly": "dns_tunneling"},
                    "is_malicious": True,
                },
            ],
            "dns": [
                {
                    "log_id": "DNS-A01",
                    "log_source": "dns",
                    "timestamp": "2024-01-21T01:30:00Z",
                    "source_ip": "10.0.0.5",
                    "event_type": "dns_query",
                    "details": {"query": "aGVsbG8gd29ybGQ.data.ns1.apt-infra.net", "record_type": "TXT", "anomaly": "high_entropy_subdomain"},
                    "is_malicious": True,
                },
            ],
            "firewall": [
                {
                    "log_id": "FW-A01",
                    "log_source": "firewall",
                    "timestamp": "2024-01-21T01:14:50Z",
                    "source_ip": "45.33.32.156",
                    "dest_ip": "10.0.0.20",
                    "event_type": "connection_allowed",
                    "details": {"port": 22, "protocol": "SSH", "direction": "inbound"},
                    "is_malicious": True,
                },
            ],
        },
    },
}


# ---------------------------------------------------------------------------
# TASK 1: Alert queue (10 alerts, mix of scenarios)
# ---------------------------------------------------------------------------

TASK1_ALERT_QUEUE = [
    # From brute_force_ssh_001
    {**ATTACK_SCENARIOS["brute_force_ssh_001"]["alerts"][0], "ground_truth_classification": "critical", "ground_truth_priority": "P1"},
    {**ATTACK_SCENARIOS["brute_force_ssh_001"]["alerts"][1], "ground_truth_classification": "critical", "ground_truth_priority": "P1"},
    # From phishing_lateral_001
    {**ATTACK_SCENARIOS["phishing_lateral_001"]["alerts"][0], "ground_truth_classification": "suspicious", "ground_truth_priority": "P2"},
    {**ATTACK_SCENARIOS["phishing_lateral_001"]["alerts"][1], "ground_truth_classification": "suspicious", "ground_truth_priority": "P2"},
    {**ATTACK_SCENARIOS["phishing_lateral_001"]["alerts"][2], "ground_truth_classification": "critical", "ground_truth_priority": "P1"},
    # From false positive
    {**ATTACK_SCENARIOS["false_positive_scan_001"]["alerts"][0], "ground_truth_classification": "benign", "ground_truth_priority": "P4"},
    # Additional synthetic benign alerts
    {
        "alert_id": "ALT-030",
        "alert_type": "User Password Changed",
        "severity": "low",
        "timestamp": "2024-01-17T09:00:00Z",
        "source_ip": "10.0.1.50",
        "dest_ip": None, "dest_port": None,
        "description": "Standard password reset via IT helpdesk portal",
        "raw_data": {"user": "bwalker", "method": "helpdesk_ticket"},
        "ground_truth_classification": "benign",
        "ground_truth_priority": "P4",
    },
    {
        "alert_id": "ALT-031",
        "alert_type": "After-hours Login",
        "severity": "medium",
        "timestamp": "2024-01-17T23:15:00Z",
        "source_ip": "10.0.1.75",
        "dest_ip": "10.0.0.10", "dest_port": 443,
        "description": "Employee login outside business hours from internal IP",
        "raw_data": {"user": "mchen", "vpn": True, "hr_approved_overtime": True},
        "ground_truth_classification": "benign",
        "ground_truth_priority": "P3",
    },
    {
        "alert_id": "ALT-032",
        "alert_type": "Large File Transfer",
        "severity": "medium",
        "timestamp": "2024-01-17T14:30:00Z",
        "source_ip": "10.0.2.88",
        "dest_ip": "10.0.0.20", "dest_port": 443,
        "description": "2.1GB file transfer to internal backup server",
        "raw_data": {"bytes": 2254857830, "destination": "BACKUP-SRV-01", "scheduled": True},
        "ground_truth_classification": "benign",
        "ground_truth_priority": "P4",
    },
    {
        "alert_id": "ALT-033",
        "alert_type": "Tor Exit Node Connection Attempt",
        "severity": "high",
        "timestamp": "2024-01-17T11:45:00Z",
        "source_ip": "198.96.155.3",
        "dest_ip": "10.0.1.20", "dest_port": 80,
        "description": "Inbound connection attempt from known Tor exit node, blocked by firewall",
        "raw_data": {"blocked": True, "threat_intel": "tor_exit_node"},
        "ground_truth_classification": "suspicious",
        "ground_truth_priority": "P2",
    },
]


# ---------------------------------------------------------------------------
# ASSET INVENTORY
# ---------------------------------------------------------------------------

ASSET_INVENTORY = [
    {"hostname": "WEB-SERVER-01", "ip_address": "10.0.1.20", "asset_type": "server", "criticality": 4, "owner": "IT Ops", "department": "Engineering", "is_compromised": False, "is_isolated": False},
    {"hostname": "DC-01", "ip_address": "10.0.0.5", "asset_type": "domain_controller", "criticality": 5, "owner": "IT Security", "department": "IT", "is_compromised": False, "is_isolated": False},
    {"hostname": "WS-HR-03", "ip_address": "10.0.2.15", "asset_type": "workstation", "criticality": 2, "owner": "Jane Smith", "department": "HR", "is_compromised": False, "is_isolated": False},
    {"hostname": "DB-FINANCE-01", "ip_address": "10.0.0.30", "asset_type": "database", "criticality": 5, "owner": "Finance", "department": "Finance", "is_compromised": False, "is_isolated": False},
    {"hostname": "IT-SCANNER-01", "ip_address": "10.0.0.100", "asset_type": "workstation", "criticality": 1, "owner": "IT Ops", "department": "IT", "is_compromised": False, "is_isolated": False},
    {"hostname": "BACKUP-SRV-01", "ip_address": "10.0.0.20", "asset_type": "server", "criticality": 3, "owner": "IT Ops", "department": "IT", "is_compromised": False, "is_isolated": False},
    {"hostname": "WS-FINANCE-01", "ip_address": "10.0.2.50", "asset_type": "workstation", "criticality": 3, "owner": "Bob Walker", "department": "Finance", "is_compromised": False, "is_isolated": False},
]
