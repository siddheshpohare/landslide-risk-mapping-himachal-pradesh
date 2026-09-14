import re, ast

src = open("phase3_validation.py", encoding="utf-8").read()

def extract_func(src, name):
    pattern = r"(^def " + name + r"\b.*?)(?=^def |\Z)"
    m = re.search(pattern, src, re.MULTILINE | re.DOTALL)
    return m.group(1).rstrip() if m else ""

# Header: imports + config (before first # --- section)
header = src.split("# -----------")[0].rstrip()

# Extra config
extra = """
TARGET_SITES = {"Lahaul_Spiti", "Sirmaur_Giri", "Rampur_Sutlej"}
LS_FIXED_THRESHOLDS = [0.05, 0.04, 0.03]
N_BASELINE_SAMPLES  = 1000
RNG_SEED = 42
GLOBAL_AUC_V1 = 0.6239120774838096
"""

# Reused helper functions from original
reused = []
for fn in ["_site_key", "discover_sites", "verify_alignment",
           "extract_pixels", "compute_metrics", "per_site_breakdown"]:
    f = extract_func(src, fn)
    reused.append(f)

# Grab visualization functions, patch them for v2
plot_roc = extract_func(src, "plot_roc_curve")
plot_roc = plot_roc.replace("def plot_roc_curve(", "def plot_roc_curve_v2(")
plot_roc = plot_roc.replace("Phase 3 -- AHP", "Phase 3 (v2 Masks) --")

plot_bar = extract_func(src, "plot_per_site_auc_bar")
plot_bar = plot_bar.replace("def plot_per_site_auc_bar(", "def plot_per_site_auc_bar_v2(")

# New functions for the fixes
new_code = open("phase3_fix_funcs.py", encoding="utf-8").read()

# Assemble
script = "\n".join([header, extra, "\n".join(reused), plot_roc, "", plot_bar, "", new_code])

# Validate
ast.parse(script)
with open("phase3_mask_fixes.py", "w", encoding="utf-8") as fh:
    fh.write(script)
print("Written OK:", len(script), "chars")
