"""Manual endpoint verification script — Task 7.4"""
import httpx
import json

BASE = "http://localhost:8000"
client = httpx.Client(timeout=10)
results = []

def test(method, path, json_body=None, expect_status=200, label=None):
    name = label or f"{method} {path}"
    try:
        if method == "GET":
            resp = client.get(f"{BASE}{path}")
        else:
            resp = client.post(f"{BASE}{path}", json=json_body)
        ok = resp.status_code == expect_status
        status = "PASS" if ok else "FAIL"
        body_preview = json.dumps(resp.json(), indent=None)[:120]
        results.append((status, name, resp.status_code))
        print(f"  [{status}] {name} -> {resp.status_code} {body_preview}")
    except Exception as e:
        results.append(("FAIL", name, str(e)))
        print(f"  [FAIL] {name} -> {e}")

print("=" * 70)
print("ENDPOINT VERIFICATION")
print("=" * 70)

# 1. Health
test("GET", "/health", label="1. GET /health")

# 2. Root
test("GET", "/", label="2. GET /")

# 3. Tasks
test("GET", "/tasks", label="3. GET /tasks")

# 4. Metrics
test("GET", "/metrics", label="4. GET /metrics")

# 5. Scenarios
test("GET", "/scenarios", label="5. GET /scenarios")

# 6. Reset (alert_triage)
test("POST", "/reset", json_body={"task_id": "alert_triage"}, label="6. POST /reset (alert_triage)")

# 7. Step (classify_alert)
test("POST", "/step", json_body={"action_type": "classify_alert", "parameters": {"alert_id": "ALT-001", "classification": "critical", "priority": "P1"}}, label="7. POST /step (classify_alert)")

# 8. State
test("GET", "/state", label="8. GET /state")

# 9. Difficulty set
test("POST", "/difficulty", json_body={"tier": 2}, label="9. POST /difficulty (tier=2)")

# 10. Reset (incident_investigation)
test("POST", "/reset", json_body={"task_id": "incident_investigation"}, label="10. POST /reset (incident_investigation)")

# 11. Step (query_logs)
test("POST", "/step", json_body={"action_type": "query_logs", "parameters": {"log_source": "auth"}}, label="11. POST /step (query_logs)")

# 12. Reset (threat_response)
test("POST", "/reset", json_body={"task_id": "threat_response"}, label="12. POST /reset (threat_response)")

# 13. Step (isolate_asset)
test("POST", "/step", json_body={"action_type": "isolate_asset", "parameters": {"hostname": "WS-HR-03"}}, label="13. POST /step (isolate_asset)")

# 14. Step (block_ip)
test("POST", "/step", json_body={"action_type": "block_ip", "parameters": {"ip_address": "94.102.49.190"}}, label="14. POST /step (block_ip)")

# 15. Invalid reset
test("POST", "/reset", json_body={"task_id": "nonexistent"}, expect_status=400, label="15. POST /reset (invalid)")

# 16. Invalid step
test("POST", "/step", json_body={"action_type": "bad_action", "parameters": {}}, expect_status=400, label="16. POST /step (invalid)")

# 17. Invalid difficulty
test("POST", "/difficulty", json_body={"tier": 99}, expect_status=400, label="17. POST /difficulty (invalid)")

client.close()

print("\n" + "=" * 70)
passed = sum(1 for r in results if r[0] == "PASS")
failed = sum(1 for r in results if r[0] == "FAIL")
print(f"RESULT: {passed}/{len(results)} passed, {failed} failed")
print("=" * 70)
