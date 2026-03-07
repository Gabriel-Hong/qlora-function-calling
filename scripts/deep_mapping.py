"""
Deep content-based mapping of API endpoints to Feature pages.
Reads: API JSON (schema/tables/examples) + Source code handlers + Feature JSONs
"""
import json
import os
import re
import sys
import glob

sys.stdout.reconfigure(encoding='utf-8')

API_SCHEMA_DIR = r"C:\Users\hjm0830\OneDrive - MIDAS\바탕 화면\MIDAS\01_Study\PyTorch\API_Data\GENNX_API_Schema"
FEATURE_DIR = r"C:\Users\hjm0830\OneDrive - MIDAS\바탕 화면\MIDAS\01_Study\PyTorch\API_Data\GENNX_Feature"
SOURCE_API_DIR = r"C:\MIDAS_Source\genw_new\src\GC_api"
SOURCE_DB_DIR = r"C:\MIDAS_Source\genw_new\src\wg_db"
OUTPUT_DIR = r"C:\Users\hjm0830\OneDrive - MIDAS\바탕 화면\MIDAS\01_Study\PyTorch\API_Data"

# ============================================================
# Step 1: Load all API endpoints
# ============================================================
def load_api_index():
    with open(os.path.join(API_SCHEMA_DIR, "_index.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_api_index(api_index):
    """Flatten nested entries (like db/SECT with sub-types) to get all article IDs"""
    flat = {}
    for endpoint, value in api_index.items():
        if isinstance(value, str):
            flat[endpoint] = value
        elif isinstance(value, dict):
            # Nested - take first entry as representative
            first_key = next(iter(value))
            flat[endpoint] = value[first_key]
    return flat

# ============================================================
# Step 2: Load API JSON content
# ============================================================
def load_api_json(article_id):
    path = os.path.join(API_SCHEMA_DIR, f"{article_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_api_info(api_data):
    """Extract key information from API JSON for matching"""
    if not api_data:
        return {}

    info = {
        "endpoint": api_data.get("endpoint", ""),
        "title": api_data.get("title", ""),
        "menu_name": api_data.get("menu_name", ""),
        "methods": api_data.get("active_methods", []),
    }

    # Extract parameter names from json_schema
    schema = api_data.get("json_schema", {})
    if isinstance(schema, dict):
        props = schema.get("properties", {})
        # Go deeper into Assign > additionalProperties > properties
        if "Assign" in props:
            assign = props["Assign"]
            if isinstance(assign, dict):
                addl = assign.get("additionalProperties", {})
                if isinstance(addl, dict):
                    inner_props = addl.get("properties", {})
                    info["param_names"] = list(inner_props.keys())
                else:
                    info["param_names"] = list(assign.get("properties", {}).keys())
            else:
                info["param_names"] = []
        else:
            info["param_names"] = list(props.keys())
    else:
        info["param_names"] = []

    # Extract table descriptions
    tables = api_data.get("tables", [])
    if isinstance(tables, list):
        descs = []
        for table in tables:
            if isinstance(table, list):
                for row in table:
                    if isinstance(row, dict) and "description" in row:
                        descs.append(row["description"])
            elif isinstance(table, dict) and "description" in table:
                descs.append(table["description"])
        info["table_descriptions"] = descs[:10]  # First 10 descriptions
    else:
        info["table_descriptions"] = []

    return info

# ============================================================
# Step 3: Read source code handler
# ============================================================
def read_source_handler(endpoint):
    """Read the source code handler for an endpoint"""
    parts = endpoint.split("/")
    if len(parts) != 2:
        return None

    category, name = parts

    # Determine handler file
    if category.lower() in ("doc", "ope", "view", "post"):
        # Business logic handler
        # Try various naming patterns
        candidates = [
            f"APIBusinessLogicHandler{name}.cpp",
            f"APIBusinessLogicHandler{name.upper()}.cpp",
            f"APIBusinessLogicHandler{name.replace('_', '')}.cpp",
        ]
    else:
        # Database handler
        candidates = [
            f"APIDatabaseHandler{name}.cpp",
            f"APIDatabaseHandler{name.upper()}.cpp",
        ]

    for candidate in candidates:
        path = os.path.join(SOURCE_API_DIR, candidate)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(5000)  # First 5KB
                return {
                    "file": candidate,
                    "content": content,
                    "register_macro": extract_register_macro(content),
                    "cdb_refs": extract_cdb_refs(content),
                }
            except:
                pass

    return None

def extract_register_macro(content):
    """Extract REGISTER_API_DATABASE_HANDLER or similar macro"""
    match = re.search(r'REGISTER_API_\w+_HANDLER\([^)]+\)', content)
    if match:
        return match.group(0)
    return ""

def extract_cdb_refs(content):
    """Extract CDB_ class references from source code"""
    refs = set(re.findall(r'CDB_(\w+)', content))
    refs.update(re.findall(r'm_pDoc->m_p(\w+)', content))
    refs.update(re.findall(r'T_(\w+)_[DK]', content))
    return list(refs)

# ============================================================
# Step 4: Read DB header
# ============================================================
def read_db_header(name):
    """Read DB_{NAME}.h to understand data structure"""
    path = os.path.join(SOURCE_DB_DIR, f"DB_{name.upper()}.h")
    if not os.path.exists(path):
        # Try without uppercase
        path = os.path.join(SOURCE_DB_DIR, f"DB_{name}.h")
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(3000)  # First 3KB

        # Extract struct fields
        fields = re.findall(r'(?:int|double|float|bool|BOOL|char|CString|TCHAR)\s+(\w+)', content)
        # Extract comments
        comments = re.findall(r'//\s*(.+)', content)

        return {
            "file": f"DB_{name.upper()}.h",
            "fields": fields[:20],
            "comments": comments[:10],
        }
    except:
        return None

# ============================================================
# Step 5: Load Feature index and content
# ============================================================
def load_feature_index():
    with open(os.path.join(FEATURE_DIR, "_index.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def load_feature_json(article_id):
    path = os.path.join(FEATURE_DIR, f"{article_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

def extract_feature_keywords(feature_data):
    """Extract keywords from feature for matching"""
    if not feature_data:
        return set()

    keywords = set()

    # From title
    title = feature_data.get("title", "").lower()
    keywords.update(title.split())

    # From sections
    sections = feature_data.get("sections", {})
    for key in ["function", "overview"]:
        text = sections.get(key, "").lower()
        keywords.update(text.split()[:50])

    # From full_text (first 200 words)
    full = feature_data.get("full_text", "").lower()
    keywords.update(full.split()[:200])

    return keywords

# ============================================================
# Step 6: Content-based matching
# ============================================================
def compute_match_score(api_info, feature_data, source_info, db_info):
    """Compute a content-based match score between API and Feature"""
    score = 0
    reasons = []

    if not feature_data:
        return 0, []

    api_title = api_info.get("title", "").lower()
    feature_title = feature_data.get("title", "").lower()
    feature_name = feature_data.get("feature_name", "").lower()

    # Title similarity
    api_words = set(api_title.split())
    feat_words = set(feature_title.split())
    common = api_words & feat_words
    if common:
        score += len(common) * 10
        reasons.append(f"title_common_words:{common}")

    # Exact title match
    if api_title == feature_title or api_title == feature_name:
        score += 50
        reasons.append("exact_title_match")

    # Parameter names in feature text
    full_text = feature_data.get("full_text", "").upper()
    sections = feature_data.get("sections", {})
    input_text = sections.get("input", "").upper()

    param_names = api_info.get("param_names", [])
    if param_names:
        matched_params = [p for p in param_names if p.upper() in full_text]
        if matched_params:
            score += len(matched_params) * 5
            reasons.append(f"params_in_text:{matched_params[:5]}")

    # Source code CDB references match
    if source_info:
        register = source_info.get("register_macro", "")
        if register:
            # Extract category from register macro
            cat_match = re.search(r'"([^"]+)"[^"]*$', register)
            if cat_match:
                category = cat_match.group(1).lower()
                if category in feature_title.lower() or category in full_text.lower():
                    score += 15
                    reasons.append(f"source_category_match:{category}")

    # DB field names in feature text
    if db_info:
        db_fields = db_info.get("fields", [])
        matched_fields = [f for f in db_fields if f.upper() in full_text and len(f) > 2]
        if matched_fields:
            score += len(matched_fields) * 3
            reasons.append(f"db_fields_in_text:{matched_fields[:5]}")

    # Section function description match
    func_desc = sections.get("function", "").lower()
    if api_title in func_desc:
        score += 20
        reasons.append("title_in_function_desc")

    # Menu name match
    menu = api_info.get("menu_name", "").lower()
    if menu and menu in feature_title.lower():
        score += 15
        reasons.append(f"menu_name_match:{menu}")

    return score, reasons

def find_best_feature_match(api_info, source_info, db_info, feature_index, feature_cache):
    """Find the best matching feature for an API endpoint"""

    api_title = api_info.get("title", "").lower()
    endpoint = api_info.get("endpoint", "")

    best_score = 0
    best_feature = ""
    best_feature_id = ""
    best_reasons = []

    for feat_name, feat_id in feature_index.items():
        # Clean feature name
        clean_name = feat_name.replace(" ↗", "").strip()

        # Quick pre-filter: skip obviously unrelated features
        # (features in totally different categories)

        # Load feature data (with caching)
        if feat_id not in feature_cache:
            feature_cache[feat_id] = load_feature_json(feat_id)

        feat_data = feature_cache[feat_id]
        if not feat_data:
            # Feature JSON not available (might be behind auth)
            # Still check title similarity
            clean_lower = clean_name.lower()
            if api_title == clean_lower:
                score = 100
                reasons = ["exact_title_match_no_json"]
            elif api_title in clean_lower or clean_lower in api_title:
                score = 30
                reasons = ["partial_title_match_no_json"]
            else:
                continue

            if score > best_score:
                best_score = score
                best_feature = clean_name
                best_feature_id = feat_id
                best_reasons = reasons
            continue

        score, reasons = compute_match_score(api_info, feat_data, source_info, db_info)

        if score > best_score:
            best_score = score
            best_feature = clean_name
            best_feature_id = feat_id
            best_reasons = reasons

    return best_feature, best_feature_id, best_score, best_reasons

# ============================================================
# Main
# ============================================================
def main():
    print("Loading data...")
    api_index = load_api_index()
    flat_apis = flatten_api_index(api_index)
    feature_index = load_feature_index()

    print(f"APIs: {len(flat_apis)}, Features: {len(feature_index)}")

    feature_cache = {}
    results = {}

    total = len(flat_apis)
    for i, (endpoint, article_id) in enumerate(flat_apis.items()):
        # Step 1: Load API JSON
        api_data = load_api_json(article_id)
        api_info = extract_api_info(api_data)
        api_info["endpoint"] = endpoint

        # Step 2: Read source code
        source_info = read_source_handler(endpoint)

        # Step 3: Read DB header
        name = endpoint.split("/")[-1] if "/" in endpoint else endpoint
        db_info = read_db_header(name)

        # Step 4: Find best feature match
        best_feature, feat_id, score, reasons = find_best_feature_match(
            api_info, source_info, db_info, feature_index, feature_cache
        )

        # Build result
        result = {
            "api_title": api_info.get("title", ""),
            "matched_feature": best_feature if score >= 15 else "",
            "feature_article_id": feat_id if score >= 15 else "",
            "match_score": score,
            "match_reasons": reasons,
            "source_handler": source_info.get("file", "") if source_info else "",
            "source_register": source_info.get("register_macro", "") if source_info else "",
            "source_cdb_refs": source_info.get("cdb_refs", []) if source_info else [],
            "db_header": db_info.get("file", "") if db_info else "",
            "api_params": api_info.get("param_names", []),
        }

        results[endpoint] = result

        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] {endpoint}: {result['matched_feature'] or '(no match)'} (score={score})")

    # Save detailed results
    detail_path = os.path.join(OUTPUT_DIR, "api_to_feature_mapping_v2_detail.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save simple mapping (same format as v1)
    simple_mapping = {ep: r["matched_feature"] for ep, r in results.items()}
    simple_path = os.path.join(OUTPUT_DIR, "api_to_feature_mapping_v2.json")
    with open(simple_path, "w", encoding="utf-8") as f:
        json.dump(simple_mapping, f, ensure_ascii=False, indent=2)

    # Summary
    mapped = sum(1 for r in results.values() if r["matched_feature"])
    unmapped = sum(1 for r in results.values() if not r["matched_feature"])
    with_source = sum(1 for r in results.values() if r["source_handler"])
    with_db = sum(1 for r in results.values() if r["db_header"])

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Total APIs: {len(results)}")
    print(f"  Mapped: {mapped}")
    print(f"  Unmapped: {unmapped}")
    print(f"  With source handler: {with_source}")
    print(f"  With DB header: {with_db}")
    print(f"\nFiles saved:")
    print(f"  Detail: {detail_path}")
    print(f"  Simple: {simple_path}")

    # Print low-confidence matches for review
    print(f"\n{'='*60}")
    print("LOW CONFIDENCE MATCHES (score 15-30):")
    for ep, r in sorted(results.items()):
        if 15 <= r["match_score"] <= 30:
            print(f"  {ep}: '{r['api_title']}' -> '{r['matched_feature']}' (score={r['match_score']}, reasons={r['match_reasons']})")

    print(f"\n{'='*60}")
    print("UNMAPPED APIs:")
    for ep, r in sorted(results.items()):
        if not r["matched_feature"]:
            src = f" [handler: {r['source_handler']}]" if r['source_handler'] else ""
            print(f"  {ep}: '{r['api_title']}'{src}")

if __name__ == "__main__":
    main()
