"""Compare v1 and v2 mappings and identify differences."""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

BASE = r"C:\Users\hjm0830\OneDrive - MIDAS\바탕 화면\MIDAS\01_Study\PyTorch\API_Data"

with open(f"{BASE}/api_to_feature_mapping.json", "r", encoding="utf-8") as f:
    v1 = json.load(f)
with open(f"{BASE}/api_to_feature_mapping_v2.json", "r", encoding="utf-8") as f:
    v2 = json.load(f)
with open(f"{BASE}/api_to_feature_mapping_v2_detail.json", "r", encoding="utf-8") as f:
    detail = json.load(f)

all_eps = sorted(set(list(v1.keys()) + list(v2.keys())))

agree = 0
disagree = 0
v1_blank_v2_filled = 0
v1_filled_v2_blank = 0
both_blank = 0
v2_wrong = []

for ep in all_eps:
    m1 = v1.get(ep, "")
    m2 = v2.get(ep, "")
    score = detail.get(ep, {}).get("match_score", 0)

    if m1 == m2:
        if m1:
            agree += 1
        else:
            both_blank += 1
    elif not m1 and m2:
        v1_blank_v2_filled += 1
    elif m1 and not m2:
        v1_filled_v2_blank += 1
    else:
        disagree += 1

print(f"Total endpoints: {len(all_eps)}")
print(f"Agree (same non-empty): {agree}")
print(f"Both blank: {both_blank}")
print(f"v1 blank, v2 filled: {v1_blank_v2_filled}")
print(f"v1 filled, v2 blank: {v1_filled_v2_blank}")
print(f"Disagree (different non-empty): {disagree}")

print("\n=== v1 BLANK but v2 FILLED ===")
for ep in all_eps:
    m1 = v1.get(ep, "")
    m2 = v2.get(ep, "")
    score = detail.get(ep, {}).get("match_score", 0)
    if not m1 and m2:
        print(f"  {ep}: v2=\"{m2}\" (score={score})")

print("\n=== v1 FILLED but v2 BLANK ===")
for ep in all_eps:
    m1 = v1.get(ep, "")
    m2 = v2.get(ep, "")
    if m1 and not m2:
        print(f"  {ep}: v1=\"{m1}\"")

print("\n=== DISAGREE (both filled but different) ===")
for ep in all_eps:
    m1 = v1.get(ep, "")
    m2 = v2.get(ep, "")
    score = detail.get(ep, {}).get("match_score", 0)
    if m1 and m2 and m1 != m2:
        print(f"  {ep}:")
        print(f"    v1: \"{m1}\"")
        print(f"    v2: \"{m2}\" (score={score})")
