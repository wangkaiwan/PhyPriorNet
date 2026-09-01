#!/bin/bash
# Eval all 4 proton-CT fold-experiment arms on fold_1's 15 held-out (eval_proton_plan.py: full-plan
# gamma 1%/1mm + 3%/3mm vs GT plan). Prints a comparison table.
# Usage: bash scripts/eval_foldexp_protonct.sh [GPU]
set -u
cd /home/kaiwang/project/DoseRad2026
CR="conda run -n doserad --no-capture-output python -u"
CFG=configs/experiments/foldexp
RUNS=/home/kaiwang/doserad2026_workdir/runs
G=${1:-1}
ARMS=(
  "A_base48        : pct_f1_A_base48_teacher        : foldexp_pct_f1_A_base48"
  "B_base32_gtkd   : pct_f1_B_base32_gtkd           : foldexp_pct_f1_B_base32gtkd"
  "C_base32_scratch: pct_f1_C_base32_scratch        : foldexp_pct_f1_C_base32scratch"
  "D_base32_ftgt   : pct_f1_D_base32_distillinit_ft : foldexp_pct_f1_D_base32ftgt"
)
for row in "${ARMS[@]}"; do
  name=$(echo "$row" | cut -d: -f1 | xargs)
  cfg=$(echo "$row"  | cut -d: -f2 | xargs)
  rd=$(echo "$row"   | cut -d: -f3 | xargs)
  ck="$RUNS/$rd/best.pt"
  if [ ! -f "$ck" ]; then echo "[eval] $name: MISSING ckpt $ck — skip"; continue; fi
  echo "[eval] $name  ckpt=$ck"
  CUDA_VISIBLE_DEVICES=$G $CR scripts/eval_proton_plan.py \
    --config "$CFG/$cfg.yaml" --ckpt "$ck" --label "foldexp_pct_${name}" \
    2>&1 | grep -viE "win_data|Warning|warn|Future|cudart" | grep -iE "PLAN γ|Error|Traceback" | tail -3
done
echo ""
echo "===== PROTON-CT FOLD-1 HELD-OUT (15) γ SUMMARY ====="
python3 - <<'PY'
import csv, os, statistics
RUNS="/home/kaiwang/doserad2026_workdir/runs"
order=["A_base48","B_base32_gtkd","C_base32_scratch","D_base32_ftgt"]
print(f"{'arm':22s} {'held-out γ1/1':>14s} {'γ3/3':>8s}  (abd / lung)")
for a in order:
    f=f"{RUNS}/proton_plan_foldexp_pct_{a}.csv"
    if not os.path.exists(f): print(f"{a:22s} {'(no csv)':>14s}"); continue
    rows=list(csv.DictReader(open(f)))
    def m(name, site=None):
        v=[float(r[name])*100 for r in rows if (site is None or r.get('site')==site) and r.get(name) not in (None,'','nan')]
        return statistics.mean(v) if v else float('nan')
    g11=m('plan_g1'); g33=m('plan_g3'); abd=m('plan_g1','abdomen'); lung=m('plan_g1','lung')
    print(f"{a:22s} {g11:>13.2f}  {g33:>7.2f}  ({abd:.1f} / {lung:.1f})")
print("\nRead: A=base48 upper bound. C>=B → scratch beats distill (user's hypothesis). D>B,C → init+ft wins.")
print("A−base32 = base32 capacity cost. Winning base32 dose net → pair with 8/1 sCT for a proton-MRI candidate.")
PY
