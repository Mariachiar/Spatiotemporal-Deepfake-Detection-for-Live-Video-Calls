# gen_tables.py
import json, math
from pathlib import Path

# ====== CONFIG (modifica qui se servono nomi/cartelle diversi) ======
ROOT = Path("dualrun/resultsffiw")
MODELS = [
    ("Base",     ROOT/"clip_noaug_alignuni"),
    ("Pretrain", ROOT/"pretrain"),
    ("Test7",    ROOT/"test7"),
]
FILTER_MODES = None  # es: ["median","mean"] per limitare i metodi di pooling
# ================================================================

def safe_load_json(p: Path):
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception: return None

def load_modes(model_dir: Path):
    j = safe_load_json(model_dir/"reports_index.json")
    return list(j.get("modes", [])) if j else []

def read_report(model_dir: Path, mode: str):
    return safe_load_json(model_dir / f"report_{mode}.json") or {}

def fmt_num(x):
    if x is None: return "-.-"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)): return "-.-"
    return f"{x:.3f}"

def tex_m(s):  # escape semplice per underscore nel nome metodo
    return str(s).replace("_", r"\_")

def extract_auc_boot(rep: dict, which: str):
    vm = rep.get("video_metrics_at_t", {}) or rep.get("video_metrics_global", {})
    if which == "roc":
        point = vm.get("roc_auc")
        boot  = vm.get("roc_auc_boot") or rep.get("video_roc_boot") or rep.get("roc_boot_video")
    else:
        point = vm.get("pr_auc")
        boot  = vm.get("pr_auc_boot") or rep.get("video_pr_auc_boot") or rep.get("pr_boot_video")

    lo = hi = mean = median = None
    if isinstance(boot, dict):
        ci = boot.get("ci95")
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            lo, hi = float(ci[0]), float(ci[1])
        mean = boot.get("mean")
        median = boot.get("median")
    return point, lo, hi, mean, median

def extract_tau_boot(rep: dict, key: str):
    vm = rep.get("video_metrics_at_t", {}) or {}
    point_map = {"precision_at_t":"precision", "recall_at_t":"recall", "f1_at_t":"f1"}
    point = vm.get(point_map[key])
    tb = vm.get("@tau_boot") or {}
    node = tb.get(key, {}) if isinstance(tb, dict) else {}
    lo = hi = mean = median = None
    ci = node.get("ci95")
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        lo, hi = float(ci[0]), float(ci[1])
    mean = node.get("mean")
    median = node.get("median")
    return point, lo, hi, mean, median

# ---- calcolo metodi comuni ----
modes_sets = []
for _, mdir in MODELS:
    ms = set(load_modes(mdir))
    if ms: modes_sets.append(ms)
if not modes_sets:
    raise SystemExit("Nessun reports_index.json trovato in: " + ", ".join(str(p) for _,p in MODELS))

if FILTER_MODES:
    modes = FILTER_MODES
else:
    inter = set.intersection(*modes_sets) if len(modes_sets) > 1 else next(iter(modes_sets))
    modes = sorted(inter) if inter else sorted(next(iter(modes_sets)))

def latex_header(caption, label, n_models):
    cols = "l" + "c"*n_models
    out = [r"\begin{table}[t]", r"\centering",
           rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
           "Metodo & " + " & ".join([rf"\textit{{{name}}}" for name,_ in MODELS]) + r" \\",
           r"\midrule"]
    return out

def latex_footer():
    return [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

lines = []

# --- Tabella AUC-ROC ---
lines += latex_header(
    "Confronto AUC-ROC (video-level) con IC95. Valore in forma $\\text{AUC}\\,[\\text{IC95}]$.",
    "tab:aucroc_video",
    len(MODELS)
)
for m in modes:
    row = [tex_m(m)]
    for _, mdir in MODELS:
        rep = read_report(mdir, m)
        auc, lo, hi, _, _ = extract_auc_boot(rep, "roc")
        row.append(f"{fmt_num(auc)} [{fmt_num(lo)}, {fmt_num(hi)}]")
    lines.append(" & ".join(row) + r" \\")
lines += latex_footer()
lines.append("")

# --- Tabella AUC-PR ---
lines += latex_header(
    "Confronto AUC-PR / AP (video-level) con IC95. Valore in forma $\\text{AP}\\,[\\text{IC95}]$.",
    "tab:aucpr_video",
    len(MODELS)
)
for m in modes:
    row = [tex_m(m)]
    for _, mdir in MODELS:
        rep = read_report(mdir, m)
        ap, lo, hi, _, _ = extract_auc_boot(rep, "pr")
        row.append(f"{fmt_num(ap)} [{fmt_num(lo)}, {fmt_num(hi)}]")
    lines.append(" & ".join(row) + r" \\")
lines += latex_footer()
lines.append("")

# --- Tabelle @τ (F1, Precision, Recall) ---
def table_tau(metric_key, nice_name, label_suffix):
    out = latex_header(
        f"{nice_name} @ $\\tau$ (video-level) con IC95. Valore in forma $m@\\tau\\,[\\text{{IC95}}]$.",
        f"tab:{label_suffix}_video",
        len(MODELS)
    )
    for m in modes:
        row = [tex_m(m)]
        for _, mdir in MODELS:
            rep = read_report(mdir, m)
            point, lo, hi, _, _ = extract_tau_boot(rep, metric_key)
            row.append(f"{fmt_num(point)} [{fmt_num(lo)}, {fmt_num(hi)}]")
        out.append(" & ".join(row) + r" \\")
    out += latex_footer()
    return out

lines += table_tau("f1_at_t",        "F1",        "f1tau")
lines.append("")
lines += table_tau("precision_at_t", "Precision", "pretau")
lines.append("")
lines += table_tau("recall_at_t",    "Recall",    "rectau")

# ---- stampa finale ----
print("\n".join(lines))
