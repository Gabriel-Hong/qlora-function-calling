"""Create API-to-Feature mapping file based on analysis of both datasets."""
import json
import os
import sys

OUTPUT_DIR = r"C:\Users\hjm0830\OneDrive - MIDAS\바탕 화면\MIDAS\01_Study\PyTorch\API_Data"

mapping = {
    # === doc/ category: Document operations ===
    "doc/NEW": "New Project",
    "doc/OPEN": "Open Project",
    "doc/CLOSE": "Close Project",
    "doc/SAVE": "Save",
    "doc/SAVEAS": "Save As",
    "doc/STAGAS": "Save Current Stage As",
    "doc/IMPORT": "Import",
    "doc/EXPORT": "Export",
    "doc/IMPORTMXT": "Merge Data File",
    "doc/EXPORTMXT": "",
    "doc/ANAL": "Perform Analysis",

    # === db/ category: Database CRUD ===
    # -- Project & Structure --
    "db/UNIT": "Unit System",
    "db/PJCF": "Project Information",
    "db/STYP": "Structure Type",
    "db/GRUP": "Structure Group",
    "db/BNGR": "Define Boundary Group",
    "db/LDGR": "Define Load Group",
    "db/TDGR": "Define Tendon Group",
    "db/NPLN": "Named Plane",
    "db/CO_M": "",
    "db/CO_S": "",
    "db/CO_T": "",
    "db/CO_F": "",
    "db/SPAN": "",
    "db/STOR": "Story",

    # -- Nodes & Elements --
    "db/NODE": "Create Nodes",
    "db/SKEW": "Node Local Axis",
    "db/MADO": "Define Domain",
    "db/DOEL": "",
    "db/ELEM": "Create Elements",
    "db/SBDO": "Define Sub-Domain",

    # -- Materials --
    "db/MATL": "Material Properties",
    "db/EDMP": "Change Property",
    "db/EPMT": "Plastic Material",
    "db/IMFM": "Inelastic Material Properties",
    "db/FIMP": "Inelastic Material Properties",
    "db/TDMT": "Creep/Shrinkage",
    "db/TDME": "Comp. Strength",
    "db/TDMF": "User Define",
    "db/TMAT": "Material Link",

    # -- Sections --
    "db/SECT": "Section Properties",
    "db/THIK": "Thickness",
    "db/TSGR": "Tapered Section Group",
    "db/SECF": "Section Stiffness Scale Factor",
    "db/ESSF": "Element Stiffness Scale Factor",
    "db/EWSF": "Wall Stiffness Scale Factor",
    "db/PSSF": "Plate Stiffness Scale Factor",
    "db/VBEM": "",
    "db/VSEC": "",
    "db/STRPSSM": "",
    "db/RPSC": "",

    # -- Inelastic Hinge --
    "db/IEHC": "Inel. Control Data",
    "db/IEHG": "Assign Inelastic Hinge Properties",
    "db/FIBR": "Fiber Division of Section (Beam-Column)",

    # -- Damping --
    "db/GRDP": "Group Damping: Element Mass & Stiffness Proportional",

    # -- Springs & Links --
    "db/GSTP": "Define General Spring Type",
    "db/NSPR": "Point Spring Supports",
    "db/GSPR": "General Spring Supports",
    "db/SSPS": "Surface Spring Supports",
    "db/ELNK": "Elastic Link",
    "db/RIGD": "Rigid Link",
    "db/NLLP": "General Link Properties",
    "db/NLNK": "General Link",
    "db/CGLP": "",
    "db/MLFC": "",

    # -- Beam/Plate End Conditions --
    "db/FRLS": "Beam End Release",
    "db/OFFS": "Beam End Offsets",
    "db/PRLS": "Plate End Release",

    # -- Seismic Devices --
    "db/SDVI": "Viscous Damper/Oil Damper",
    "db/SDVE": "Viscoelastic Damper",
    "db/SDST": "Steel Damper",
    "db/SDHY": "Hysteretic Isolator(MSS)",
    "db/SDIS": "Isolator(MSS)",

    # -- Constraints & Supports --
    "db/CONS": "Define Supports",
    "db/MCON": "Linear Constraints",
    "db/PZEF": "Panel Zone Effects",
    "db/CLDR": "Define Constraint Label Direction",
    "db/DRLS": "Diaphragm Disconnect",

    # -- Static Loads --
    "db/STLD": "Static Load Cases",
    "db/BODF": "Self Weight",
    "db/CNLD": "Nodal Loads",
    "db/BMLD": "Element Beam Loads",
    "db/NMAS": "Nodal Masses",
    "db/LTOM": "Loads to Masses",
    "db/NBOF": "Nodal Body Force",
    "db/SDSP": "Specified Displacements of Supports",
    "db/PSLT": "Define Pressure Load Type",
    "db/PRES": "Assign Pressure Loads",
    "db/PNLD": "Define Plane Load Type",
    "db/PNLA": "Assign Plane Loads",
    "db/FBLD": "Define Floor Load Type",
    "db/FBLA": "Assign Floor Loads",
    "db/FMLD": "",
    "db/POSP": "Parameters of Soil Properties",
    "db/EPST": "Static Earth Pressure",
    "db/POSL": "Seismic Loads",

    # -- Temperature Loads --
    "db/ETMP": "Element Temperature",
    "db/GTMP": "Temperature Gradient",
    "db/BTMP": "Beam Section Temperatures",
    "db/STMP": "System Temperature",
    "db/NTMP": "Nodal Temperature",

    # -- Tendon --
    "db/TDNT": "Tendon Property",
    "db/TDNA": "Tendon Profile",
    "db/TDCS": "Tendon Location for Composite Section",
    "db/TDPL": "Tendon Prestress Loads",
    "db/PRST": "Prestress Beam Loads",
    "db/PTNS": "Pretension Loads",
    "db/EXLD": "External Type Loadcase for Pretension",

    # -- Moving Load --
    "db/MVCD": "Moving Load Code",
    "db/LLAN": "Traffic Line Lanes",
    "db/LLANch": "",
    "db/LLANid": "",
    "db/LLANtr": "",
    "db/LLANop": "",
    "db/SLAN": "Traffic Surface Lanes",
    "db/SLANch": "",
    "db/SLANop": "",
    "db/MVHL": "Vehicles",
    "db/MVHLtr": "",
    "db/MVLD": "Moving Load Cases",
    "db/MVLDbs": "",
    "db/MVLDch": "",
    "db/MVLDeu": "",
    "db/MVLDid": "",
    "db/MVLDpl": "",
    "db/MVLDtr": "",
    "db/MVHC": "Vehicle Classes",
    "db/SINF": "Plate Elements for Influence Surface",
    "db/MLSP": "Lane Supports (Negative Moments at Interior Piers)",
    "db/MLSR": "",
    "db/CRGR": "",
    "db/CJFG": "",
    "db/DYFG": "",
    "db/DYLA": "",
    "db/DYNF": "",
    "db/IMPF": "Imperfection Data",

    # -- Response Spectrum --
    "db/SPFC": "Response Spectrum Functions",
    "db/SPLC": "Response Spectrum Load Cases",

    # -- Time History --
    "db/THGC": "Global Control",
    "db/THIS": "Time History Load Cases",
    "db/THFC": "Time History Functions",
    "db/THGA": "Ground Acceleration",
    "db/THNL": "Dynamic Nodal Loads",
    "db/THSL": "Time Varying Static Loads",
    "db/THMS": "Multiple Support Excitation",
    "db/THRE": "",
    "db/THRG": "",
    "db/THRI": "",
    "db/THRS": "",

    # -- Construction Stage --
    "db/STAG": "Define Construction Stage",
    "db/CSCS": "Composite Section for Construction Stage",
    "db/TMLD": "Time Loads for Construction Stage",
    "db/CRPC": "Creep Coefficient for Construction Stage",
    "db/CMCS": "",
    "db/STBK": "",

    # -- Heat of Hydration --
    "db/ETFC": "Ambient Temperature Functions",
    "db/CCFC": "Convection Coefficient Functions",
    "db/HECB": "Element Convection Boundary",
    "db/HSPT": "Prescribed Temperature",
    "db/HSFC": "Heat Source Functions",
    "db/HAHS": "Assign Heat Source",
    "db/HPCE": "Pipe Cooling",
    "db/HSTG": "Define Construction Stage for Hydration",

    # -- Settlement --
    "db/SMPT": "Settlement Group",
    "db/SMLC": "Settlement Load Cases",

    # -- Other Load Data --
    "db/LDSQ": "Load Sequence for Nonlinear",
    "db/PLCB": "Pre-Composite Section",
    "db/WVLD": "",
    "db/IELC": "Ignore Elements for Load Cases",
    "db/GALD": "",

    # -- Initial Forces --
    "db/EFCT": "Initial Forces Control Data",
    "db/IFGS": "Initial Forces for Geometric Stiffness",
    "db/INMF": "Initial Element Forces",

    # -- Analysis Control --
    "db/ACTL": "Main Control Data",
    "db/PDEL": "P-Delta Analysis Control",
    "db/BUCK": "Buckling Analysis Control",
    "db/EIGV": "Eigenvalue Analysis Control",
    "db/HHCT": "Heat of Hydration Analysis Control",
    "db/MVCT": "Moving Load Analysis Control",
    "db/MVCTbs": "",
    "db/MVCTch": "",
    "db/MVCTid": "",
    "db/MVCTTR": "",
    "db/NLCT": "Nonlinear Analysis Control",
    "db/STCT": "Construction Stage Analysis Control",
    "db/SMCT": "",
    "db/BCCT": "Boundary Change Assignment to Load Case / Analysis",

    # -- Pushover --
    "db/POGD": "Pushover Global Control",
    "db/IEPI": "Ignore Elements for NL. Analysis Initial Load",
    "db/POLC": "Pushover Load Cases",
    "db/PHGE": "Assign Pushover Hinge Properties",

    # -- Load Combinations --
    "db/LCOM": "Load Combinations",

    # -- Results / Post-processing data --
    "db/CUTL": "",
    "db/CLWP": "",
    "db/CAMB": "",
    "db/GSBG": "",
    "db/ULFC": "Unknown Load Factor",
    "db/GCMB": "",
    "db/HHND": "",

    # -- Design --
    "db/DCON": "(Design> RC) Design Code Option",
    "db/DSTL": "(Design> Steel) Design Code Option",
    "db/DCTL": "(Design) Definition of Frame",
    "db/LENG": "(Design) Unbraced Length (L,Lb)",
    "db/LTSR": "(Design) Limiting Slenderness Ratio",
    "db/MEMB": "(Design) Member Assignment",
    "db/WMAK": "(Design> RC) Modify Wall Mark Data",
    "db/MBTP": "(Design) Modify Member Type",
    "db/MATD": "(Design> RC) Modify Concrete Material",
    "db/RCHK": "",

    # === ope/ category: Operations ===
    "ope/DIVIDEELEM": "Divide Elements",
    "ope/SECTPROP": "Sectional Property Calculator",
    "ope/USLC": "Create Load Cases Using Load Combinations",
    "ope/PROJECTSTATUS": "Project Status",
    "ope/SSPS": "Surface Spring Supports",
    "ope/LINEBMLD": "Line Beam Loads",
    "ope/EDMP": "Change Property",
    "ope/AUTOMESH": "Auto-Mesh Planar Area",
    "ope/STORY_PARAM": "",
    "ope/STORPROP": "",
    "ope/STOR": "",
    "ope/MEMB": "(Design) Member Assignment",
    "ope/STORY_IRR_PARAM": "",

    # === view/ category: View/Display ===
    "view/SELECT": "Select",
    "view/CAPTURE": "Graphic Files",
    "view/PRECAPTURE": "",
    "view/ANGLE": "View Point",
    "view/ACTIVE": "Activities",
    "view/DISPLAY": "Display",
    "view/RESULTGRAPHIC": "",

    # === post/ category: Post-processing ===
    "post/TABLE": "",
    "POST/TABLE": "",
    "post/TEXT": "Text Output",
    "post/PM": "",
    "post/STEELCODECHECK": "(Design> Steel) Steel Code Check",
}

output_path = os.path.join(OUTPUT_DIR, "api_to_feature_mapping.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)

mapped = sum(1 for v in mapping.values() if v)
unmapped = sum(1 for v in mapping.values() if not v)
print(f"Saved {len(mapping)} API-to-Feature mappings to:")
print(f"  {output_path}")
print(f"  Mapped: {mapped}, Unmapped (blank): {unmapped}")
