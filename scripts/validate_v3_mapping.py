"""
Validate api_to_feature_mapping_v3.json against:
1. API Schema (_index.json + individual JSONs for menu_name)
2. Feature index (_index.json)
"""
import html
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

BASE = r"C:\Users\hjm0830\OneDrive - MIDAS\바탕 화면\MIDAS\01_Study\PyTorch\API_Data"
API_SCHEMA_DIR = os.path.join(BASE, "GENNX_API_Schema")
FEATURE_DIR = os.path.join(BASE, "GENNX_Feature")

# ============================================================
# Step 1: Load data sources
# ============================================================

# v3 mapping
with open(os.path.join(BASE, "api_to_feature_mapping_v3.json"), "r", encoding="utf-8") as f:
    v3 = json.load(f)

# API Schema index (flatten nested entries)
with open(os.path.join(API_SCHEMA_DIR, "_index.json"), "r", encoding="utf-8") as f:
    api_index_raw = json.load(f)

api_index = {}
for endpoint, value in api_index_raw.items():
    if isinstance(value, str):
        api_index[endpoint] = value
    elif isinstance(value, dict):
        first_key = next(iter(value))
        api_index[endpoint] = value[first_key]

# Feature index
with open(os.path.join(FEATURE_DIR, "_index.json"), "r", encoding="utf-8") as f:
    feat_index_raw = json.load(f)

# Clean feature names: remove "↗" suffix (with optional spaces/nbsp)
# Keep HTML entities as-is since v3 now uses them too
feat_names = set()
for name in feat_index_raw.keys():
    clean = re.sub(r'[\s\u00a0]*↗$', '', name).strip()
    feat_names.add(clean)

# Build endpoint → menu_name lookup from individual API JSONs
endpoint_to_menu = {}
for endpoint, article_id in api_index.items():
    path = os.path.join(API_SCHEMA_DIR, f"{article_id}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            endpoint_to_menu[endpoint] = data.get("menu_name", "")
        except Exception:
            endpoint_to_menu[endpoint] = ""
    else:
        endpoint_to_menu[endpoint] = ""

# ============================================================
# Step 2: Validation checks
# ============================================================

# A. v3 keys not in API Schema
missing_in_api = []
for key in sorted(v3.keys()):
    if key not in api_index:
        missing_in_api.append(key)

# B. v3 values not in Feature index
missing_in_feat = []
for key in sorted(v3.keys()):
    val = v3[key]
    if val and val not in feat_names:
        missing_in_feat.append((key, val))

# C. menu_name comparison (informational)
def normalize(s):
    """Normalize for fuzzy comparison: lowercase, strip punctuation/spaces"""
    return re.sub(r'[^a-z0-9]', '', s.lower())

exact_match = []
fuzzy_match = []
mismatch = []
no_menu = []

for key in sorted(v3.keys()):
    val = v3[key]
    if not val:
        continue
    menu = endpoint_to_menu.get(key, "")
    if not menu:
        no_menu.append((key, val))
        continue
    if val == menu:
        exact_match.append((key, val, menu))
    elif normalize(val) == normalize(menu):
        fuzzy_match.append((key, val, menu))
    else:
        mismatch.append((key, val, menu))

# D. Empty values
empty_vals = [key for key in sorted(v3.keys()) if not v3[key]]

# ============================================================
# Step 3: Report
# ============================================================
print("=" * 70)
print("VALIDATION REPORT: api_to_feature_mapping_v3.json")
print("=" * 70)

total = len(v3)
mapped = sum(1 for v in v3.values() if v)
print(f"\nv3 summary: {total} endpoints, {mapped} mapped, {total - mapped} empty")
print(f"API Schema: {len(api_index)} endpoints")
print(f"Feature index: {len(feat_names)} features")

# --- A ---
print(f"\n{'─' * 70}")
print(f"A. v3 KEYS NOT IN API SCHEMA: {len(missing_in_api)}")
print(f"{'─' * 70}")
if missing_in_api:
    for key in missing_in_api:
        print(f"  {key}: \"{v3[key]}\"")
else:
    print("  (none)")

# --- B ---
print(f"\n{'─' * 70}")
print(f"B. v3 VALUES NOT IN FEATURE INDEX: {len(missing_in_feat)}")
print(f"{'─' * 70}")
if missing_in_feat:
    for key, val in missing_in_feat:
        print(f"  {key}: \"{val}\"")
else:
    print("  (none)")

# --- C ---
print(f"\n{'─' * 70}")
print(f"C. MENU_NAME COMPARISON (mapped endpoints only)")
print(f"{'─' * 70}")
print(f"  Exact match (v3 value == menu_name):  {len(exact_match)}")
print(f"  Fuzzy match (case/punct diff only):   {len(fuzzy_match)}")
print(f"  Mismatch (different strings):         {len(mismatch)}")
print(f"  No menu_name available:               {len(no_menu)}")

if fuzzy_match:
    print(f"\n  --- Fuzzy matches ---")
    for key, val, menu in fuzzy_match:
        print(f"    {key}: v3=\"{val}\" | menu=\"{menu}\"")

if mismatch:
    print(f"\n  --- Mismatches (expected: v3 maps to Feature name, not menu_name) ---")
    for key, val, menu in mismatch:
        print(f"    {key}: v3=\"{val}\" | menu=\"{menu}\"")

# --- D ---
print(f"\n{'─' * 70}")
print(f"D. EMPTY VALUES: {len(empty_vals)}")
print(f"{'─' * 70}")
if empty_vals:
    for key in empty_vals:
        menu = endpoint_to_menu.get(key, "")
        suffix = f"  (menu: \"{menu}\")" if menu else ""
        print(f"  {key}{suffix}")
else:
    print("  (none)")

print(f"\n{'=' * 70}")
print("END OF REPORT")
print(f"{'=' * 70}")
