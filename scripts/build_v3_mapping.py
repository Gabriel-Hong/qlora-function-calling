"""
Build v3 API-to-Feature mapping.
Start with v1 as base, apply verified corrections.
Agent verification results will be merged in as they complete.
"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

BASE = r"C:\Users\hjm0830\OneDrive - MIDAS\바탕 화면\MIDAS\01_Study\PyTorch\API_Data"

with open(f"{BASE}/api_to_feature_mapping.json", "r", encoding="utf-8") as f:
    v1 = json.load(f)

# Start with v1 as base (more reliable for what it mapped)
v3 = dict(v1)

# ============================================================
# 1. HTML entity fixes: Feature _index.json uses &gt; &amp; etc.
# v3 values must match Feature index keys (after ↗ removal) exactly.
# ============================================================
html_entity_fixes = {
    "db/DCON": "(Design&gt; RC) Design Code Option",
    "db/DSTL": "(Design&gt; Steel) Design Code Option",
    "db/GRDP": "Group Damping: Element Mass &amp; Stiffness Proportional",
    "db/MATD": "(Design&gt; RC) Modify Concrete Material",
    "db/WMAK": "(Design&gt; RC) Modify Wall Mark Data",
    "post/STEELCODECHECK": "(Design&gt; Steel) Steel Code Check",
}
v3.update(html_entity_fixes)

# ============================================================
# 2. Sub-variant mappings (v1 left blank, logical parent match)
# These are sub-endpoints that share the parent's Feature page
# ============================================================
sub_variants = {
    # Traffic Line Lanes sub-variants (parent: db/LLAN → "Traffic Line Lanes")
    "db/LLANch": "Traffic Line Lanes",
    "db/LLANid": "Traffic Line Lanes",
    "db/LLANtr": "Traffic Line Lanes",
    "db/LLANop": "Traffic Line Lanes",

    # Traffic Surface Lanes sub-variants (parent: db/SLAN → "Traffic Surface Lanes")
    "db/SLANch": "Traffic Surface Lanes",
    "db/SLANop": "Traffic Surface Lanes",

    # Moving Load Cases sub-variants (parent: db/MVLD → "Moving Load Cases")
    "db/MVLDbs": "Moving Load Cases",
    "db/MVLDch": "Moving Load Cases",
    "db/MVLDeu": "Moving Load Cases",
    "db/MVLDid": "Moving Load Cases",
    "db/MVLDpl": "Moving Load Cases",
    "db/MVLDtr": "Moving Load Cases",

    # Vehicles sub-variant (parent: db/MVHL → "Vehicles")
    "db/MVHLtr": "Vehicles",

    # Moving Load Analysis Control sub-variants (parent: db/MVCT → "Moving Load Analysis Control")
    "db/MVCTbs": "Moving Load Analysis Control",
    "db/MVCTch": "Moving Load Analysis Control",
    "db/MVCTid": "Moving Load Analysis Control",
    "db/MVCTTR": "Moving Load Analysis Control",

    # Lane Supports sub-variant (parent: db/MLSP → "Lane Supports...")
    "db/MLSR": "Lane Supports (Negative Moments at Interior Piers)",
}
v3.update(sub_variants)

# ============================================================
# 3. Clear v1 correct cases where v2 was wrong
# (already in v1 base, just documenting)
# ============================================================
# db/NODE: v1="Create Nodes" ✓ (v2 matched "Renumbering Node" - wrong)
# db/ELEM: v1="Create Elements" ✓ (v2 matched "Renumbering Element" - wrong)
# db/CONS: v1="Define Supports" ✓ (v2 matched "Define Constraint Label Direction" - wrong)
# db/LDGR: v1="Define Load Group" ✓ (v2 matched "Change Load Group" - wrong)
# db/BNGR: v1="Define Boundary Group" ✓ (v2 matched "Change Boundary Group" - wrong)
# doc/IMPORT: v1="Import" ✓ (v2 matched "Print" - wrong)
# doc/EXPORT: v1="Export" ✓ (v2 matched "Print" - wrong)
# doc/IMPORTMXT: v1="Merge Data File" ✓ (v2 matched "Print" - wrong)

# ============================================================
# 4. Source-code verified corrections
# Based on: REGISTER macro category + API JSON title + Feature index search
# ============================================================
verified_corrections = {
    # --- Disputed cases: v1 confirmed correct ---
    # db/CONS: API="Constraint Support", category="Boundary" → Feature "Define Supports" ✓
    # db/BCCT: API="Boundary Change Assignment", category="Analysis" → Feature exact match ✓
    # db/EWSF: API="Effective Width Scale Factor", category="Properties" → Feature "Wall Stiffness Scale Factor" ✓
    # db/PSSF: API="Section Manager - Plate Stiffness Scale Factor" → Feature "Plate Stiffness Scale Factor" ✓
    # db/PSLT: API="Define Pressure Load Type", category="Static Loads" → Feature exact match ✓
    # db/POSL: API="Parameter of Seismic Loads", category="Properties" → Feature "Seismic Loads" ✓
    # db/POGD: API="Pushover Analysis Control Data", category="Pushover" → Feature "Pushover Global Control" ✓
    # db/POLC: API="Pushover Load Cases", category="Pushover" → Feature exact match ✓
    # db/IEPI: API="Ignore Elements for Pushover Initial Load", category="Boundary" → Feature close match ✓
    # db/NLCT: API="Nonlinear Analysis Control Data", category="Analysis" → Feature "Nonlinear Analysis Control" ✓
    # db/STCT: API="Construction Stage Analysis Control Data", category="Analysis" → Feature exact match ✓
    # db/THGC: API="Time History Global Control", category="Dynamic Loads" → Feature "Global Control" ✓
    # db/INMF: API="Small Displacement - Initial Element Force", category="Miscellaneous Loads" → Feature "Initial Element Forces" ✓
    # db/TDMT: API="Time Dependent Material - Creep/Shrinkage", category="Properties" → Feature "Creep/Shrinkage" ✓
    # db/TDME: API="Time Dependent Material - Compressive Strength", category="Properties" → Feature "Comp. Strength" ✓
    # db/TMAT: API="Time Dependent Material Link", category="Properties" → Feature "Material Link" ✓
    # db/STOR: API="Story Data", category="Properties" → Feature "Story" ✓
    # db/NSPR: API="Point Spring", category="Boundary" → Feature "Point Spring Supports" ✓
    # db/SDHY: API="Seismic Device - Hysteretic Isolator(MSS)", category="Boundary" → Feature exact match ✓
    # db/SDIS: API="Seismic Device - Isolator(MSS)", category="Boundary" → Feature exact match ✓
    # db/DCON: API="RC Design Code", category="Design" → Feature "(Design> RC) Design Code Option" ✓

    # --- Disputed cases: v1 WRONG, corrected ---
    # db/IMPF: API="Additional Impact Factor", category="Moving Loads" (NOT "Imperfection Data")
    # Source code comment: "Moving Load Impact Factor". No "Impact Factor" feature page → blank
    "db/IMPF": "",
    # db/EWSF: API="Effective Width Scale Factor", CIVIL-only (#ifdef _CIVIL)
    # v1 had "Wall Stiffness Scale Factor" but API is about effective width, not wall stiffness → blank
    "db/EWSF": "",

    # --- v1 blank cases with verified Feature match ---
    # db/CLWP: API="Plate Cutting Line Diagram" → Feature exact match
    "db/CLWP": "Plate Cutting Line Diagram",
    # db/CUTL: API="Cutting Line" → data definition for cutting lines used in plate cutting
    "db/CUTL": "Plate Cutting Line Diagram",
    # db/DOEL: API="Domain-Element", category="Node/Element" → domain-element assignment
    "db/DOEL": "Define Domain",
    # db/HHND: API="Heat of Hydration Result Graph" → Feature "(Heat of Hydration Analysis) Graph"
    "db/HHND": "(Heat of Hydration Analysis) Graph",
    # db/THRE-S: Time History result graph sub-types → parent feature
    "db/THRE": "Time History Graph - Time History Graph",
    "db/THRG": "Time History Graph - Time History Graph",
    "db/THRI": "Time History Graph - Time History Graph",
    "db/THRS": "Time History Graph - Time History Graph",
    # doc/EXPORTMXT: API="Export to mct/mgt" → Export feature covers MCT/MGT export
    "doc/EXPORTMXT": "Export",
    # ope/STOR: API="Story Calc OPT" → auto-generates story data from model
    "ope/STOR": "Story",
    # ope/STORPROP: API="Story Properties" → calculates story-level properties
    "ope/STORPROP": "Story",
    # ope/STORY_PARAM: API="Story Check Parameter" → story checking config
    "ope/STORY_PARAM": "Story",
    # ope/STORY_IRR_PARAM: API="Story Irregularity Check Parameter" → story irregularity config
    "ope/STORY_IRR_PARAM": "Story",

    # --- Remaining blank cases: verified no Feature match ---
    # Display color settings (no dedicated feature pages):
    # db/CO_M="Material Color", db/CO_S="Section Color", db/CO_T="Thickness Color", db/CO_F="Floor Load Color"
    # CIVIL-only features (no GEN NX feature page):
    # db/SPAN="Span Information", db/CAMB="FCM Camber Control", db/GALD="Grid Analysis Load"
    # db/GCMB="General Camber Control", db/GSBG="Bridge Girder Diagrams"
    # db/DYFG="Railway Dynamic Factor", db/DYLA="Dynamic Load Allowance", db/DYNF="Railway Dynamic Factor by Element"
    # db/CJFG="Concurrent Joint Force Group", db/CRGR="Concurrent Reaction Group"
    # db/WVLD="Wave Loads", db/FMLD="Finishing Material Loads"
    # db/CMCS="Camber for Construction Stage", db/STBK="Set-Back Loads for NL C.S."
    # Section manager internals (no standalone feature pages):
    # db/VBEM="Virtual Beam", db/VSEC="Virtual Section", db/STRPSSM="Stress Points", db/RPSC="Reinforcements"
    # Other no-match cases:
    # db/CGLP="Change General Link Property", db/MLFC="Force-Deformation Function"
    # db/SMCT="Settlement Analysis Control Data", db/RCHK="Rebar Input for Checking"
    # view/PRECAPTURE="Dialog Capture", view/RESULTGRAPHIC=multi-type endpoint
    # post/TABLE=multi-type endpoint, POST/TABLE=multi-type endpoint, post/PM="P-M Interaction Diagram"
}
agent_corrections = verified_corrections

# Apply agent corrections
v3.update(agent_corrections)

# ============================================================
# Save v3
# ============================================================
output_path = f"{BASE}/api_to_feature_mapping_v3.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(v3, f, ensure_ascii=False, indent=2)

mapped = sum(1 for v in v3.values() if v)
unmapped = sum(1 for v in v3.values() if not v)
print(f"v3 mapping saved: {output_path}")
print(f"  Total: {len(v3)}, Mapped: {mapped}, Unmapped: {unmapped}")

# Show what changed from v1
changes = []
for ep in sorted(v3.keys()):
    old = v1.get(ep, "")
    new = v3.get(ep, "")
    if old != new:
        changes.append(f"  {ep}: '{old}' -> '{new}'")

if changes:
    print(f"\nChanges from v1 ({len(changes)}):")
    for c in changes:
        print(c)
