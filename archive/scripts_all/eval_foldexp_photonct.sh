#!/bin/bash
# Eval all 4 fold-experiment arms on fold_1's 15 held-out (cv_eval_photonct_full: deploy-path pred,
# margin-24, cutoff quantise, gamma vs full-grid GT plans). Prints a comparison table.
# Usage: bash scripts/eval_foldexp_photonct.sh [GPU]
set -u
cd /home/kaiwang/project/DoseRad2026
CR="conda run -n doserad --no-capture-output python -u"
CFG=configs/experiments/foldexp
RUNS=/home/kaiwang/doserad2026_workdir/runs
LOGS=/home/kaiwang/doserad2026_workdir/logs
G=${1:-0}
# arm : config : run_dir
ARMS=(
  "A_base48       : ct_f1_A_base48_teacher        : foldexp_ct_f1_A_base48"
  "B_base32_gtkd  : ct_f1_B_base32_gtkd           : foldexp_ct_f1_B_base32gtkd"
  "C_base32_scratch: ct_f1_C_base32_scratch       : foldexp_ct_f1_C_base32scratch"
  "D_base32_ftgt  : ct_f1_D_base32_distillinit_ft : foldexp_ct_f1_D_base32ftgt"
)
for row in "${ARMS[@]}"; do
  name=$(echo "$row" | cut -d: -f1 | xargs)
  cfg=$(echo "$row"  | cut -d: -f2 | xargs)
  rd=$(echo "$row"   | cut -d: -f3 | xargs)
  ck="$RUNS/$rd/state.pt"
  out="$LOGS/foldexp_${name}.csv"
  if [ ! -f "$ck" ]; then echo "[eval] $name: MISSING ckpt $ck — skip"; continue; fi
  echo "[eval] $name  ckpt=$ck"
  CUDA_VISIBLE_DEVICES=$G $CR scripts/cv_eval_photonct_full.py \
    --config "$CFG/$cfg.yaml" --ckpt "$ck" --out "$out" \
    2>&1 | grep -viE "win_data|Warning|warn|Future|cudart" | grep -iE "TRUE mean g1|Error|Traceback|MISSING" | tail -4
done
echo ""
echo "===== FOLD-1 HELD-OUT (15) γ SUMMARY ====="
python3 - <<'PY'
import csv, glob, os
LOGS="/home/kaiwang/doserad2026_workdir/logs"
order=["A_base48","B_base32_gtkd","C_base32_scratch","D_base32_ftgt"]
print(f"{'arm':22s} {'held-out γ1/1':>14s} {'γ3/3':>8s}  (mean over 15)")
for a in order:
    f=f"{LOGS}/foldexp_{a}.csv"
    if not os.path.exists(f): print(f"{a:22s} {'(no csv)':>14s}"); continue
    import statistics
    rows=list(csv.DictReader(open(f)))
    def col(name):   # cv_eval_photonct_full writes plan_g1/plan_g3 as FRACTIONS 0-1
        vals=[float(r[name]) for r in rows if r.get(name) not in (None,'','nan')]
        return statistics.mean(vals)*100 if vals else float('nan')
    g11=col('plan_g1'); g33=col('plan_g3')
    lung=[float(r['plan_g1'])*100 for r in rows if r.get('site')=='lung' and r.get('plan_g1') not in (None,'','nan')]
    abd =[float(r['plan_g1'])*100 for r in rows if r.get('site')=='abdomen' and r.get('plan_g1') not in (None,'','nan')]
    extra=f"  (abd {statistics.mean(abd):.1f} / lung {statistics.mean(lung):.1f})" if lung and abd else ""
    print(f"{a:22s} {g11:>13.2f}  {g33:>7.2f}{extra}")
print("\nRead: A=base48 upper bound. Compare B(distill) vs C(scratch) vs D(init+ft) at base32.")
print("If C>=B: user's overfitting-hypothesis holds (scratch beats distill on small data).")
print("If D>B,C: distill-init + GT-finetune is the winner. Gap to A = the base32 capacity cost.")
PY
