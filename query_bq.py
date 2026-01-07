#!/usr/bin/env python3
"""Query BigQuery to verify synced data."""
from google.cloud import bigquery

client = bigquery.Client()
query = """
SELECT TimeGenerated, Role, CameraName, RiskTier, Summary
FROM security_guard.security_logs
ORDER BY TimeGenerated DESC
LIMIT 10
"""
print("=== BigQuery Data ===")
print()
results = client.query(query)
for row in results:
    ts = str(row.TimeGenerated)[:19] if row.TimeGenerated else "N/A"
    role = row.Role or "N/A"
    camera = row.CameraName or "N/A"
    risk = row.RiskTier or "N/A"
    summary = (row.Summary or "")[:60]
    print(f"{ts} | {role:9} | {camera:10} | {risk:6} | {summary}")
print()
print("=== End ===")
