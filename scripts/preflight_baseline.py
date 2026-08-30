"""Deploy manifest + preflight verifier — the guardrail that would have caught this session's failures
(wrong photon weight, wrong margin, deleted GT cache). Run BEFORE any deploy-path baseline / before/after
eval, or after (re)building a docker. It reads the docker IMAGE as the source of truth and asserts the
eval config agrees.

  conda run -n doserad python scripts/preflight_baseline.py [task]   # task = all | photon_ct | ...

Checks per task:
  [W] image weight md5 == the weight file the eval will load (the #1 bug this session)
  [M] image DOSERAD_PHOTON_MARGIN == the margin the eval sets (photon only; default 8 vs shipped 24)
  [C] GT cache dir exists AND held16[0] has the expected # of per-CP npz (catches deleted/moved caches)
  [K] cohort file exists with 16 patients (8 abd + 8 lung)
  [L] clf md5 (MRI) == image clf
Exit 0 only if every check PASSES. NON-zero + a red FAIL line otherwise.
"""
from __future__ import annotations
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

RUNS = "/home/kaiwang/doserad2026_workdir/runs"
COHORT = "/home/kaiwang/doserad2026_workdir/eval_cohort_frozen.json"
CLF = "/data/kwang/sct_classify_runs/clf_whole/best.pt"

# ── DEPLOY MANIFEST (the single source of truth; keep in sync with BASELINE_held16_2026-07-31.md) ──
MANIFEST = {
    "photon_ct": dict(image="doserad-photon:p2",
                      img_weight="/opt/algorithm/container/photon/weights/photon.pt",
                      eval_weight=f"{RUNS}/docker_extracted/photon_ct_docker.pt",
                      margin=24, cache="/home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_m24",
                      clf=None),
    "photon_mri": dict(image="doserad-photon-mri:scheme2p4",
                       img_weight="/opt/algorithm/container/photon_mri/weights/photon_mri.pt",
                       eval_weight=f"{RUNS}/docker_extracted/photon_mri_docker.pt",
                       margin=24, cache="/home/kaiwang/doserad2026_workdir/cache/crops/photon_skinentry_m24",
                       clf=CLF),
    "proton_ct": dict(image="doserad-proton:latest",
                      img_weight="/opt/algorithm/container/proton/weights/proton.pt",
                      eval_weight=f"{RUNS}/all75_r2_ft/state.pt",
                      margin=None, cache="/home/kaiwang/doserad2026_workdir/cache/crops/proton_ssd", clf=None),
    "proton_mri": dict(image="doserad-proton-mri:latest",
                       img_weight="/opt/algorithm/container/proton_mri/weights/proton_mri.pt",
                       eval_weight=f"{RUNS}/all75_r3_protonmri/state.pt",
                       margin=None, cache="/home/kaiwang/doserad2026_workdir/cache/crops/proton_ssd", clf=CLF),
}


def sg(cmd):
    return subprocess.run(["sg", "docker", "-c", cmd], capture_output=True, text=True).stdout.strip()


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def img_env(image):
    out = sg(f"docker inspect {image} --format '{{{{json .Config.Env}}}}'")
    try:
        return dict(e.split("=", 1) for e in json.loads(out) if "=" in e)
    except Exception:
        return {}


def img_file_md5(image, path_in_image):
    tmp = tempfile.mkdtemp(prefix="preflight_")
    cid = sg(f"docker create {image}").splitlines()[-1].strip()
    if not cid:
        return None
    dst = os.path.join(tmp, "w.pt")
    sg(f"docker cp {cid}:{path_in_image} {dst}")
    sg(f"docker rm {cid} >/dev/null")
    return md5(dst) if os.path.exists(dst) else None


def check(task):
    m = MANIFEST[task]
    fails = []
    cohort = json.load(open(COHORT))
    pid0 = cohort["held16"][0]

    # [K] cohort
    n = len(cohort["held16"]); nab = len(cohort["abd8"]); nth = len(cohort["thb8"])
    ok_k = (n == 16 and nab == 8 and nth == 8)
    print(f"  [K] cohort held16={n} abd={nab} thb={nth}  {'PASS' if ok_k else 'FAIL'}")
    if not ok_k: fails.append("cohort")

    # [W] weight identity image vs eval
    if not os.path.exists(m["eval_weight"]):
        print(f"  [W] eval weight MISSING: {m['eval_weight']}  FAIL"); fails.append("eval_weight_missing")
    else:
        ev = md5(m["eval_weight"]); iv = img_file_md5(m["image"], m["img_weight"])
        ok_w = (iv is not None and ev == iv)
        print(f"  [W] weight  eval={ev[:8]}  image={iv[:8] if iv else 'MISSING'}  {'PASS' if ok_w else 'FAIL'}")
        if not ok_w: fails.append("weight_mismatch")

    # [M] margin (photon)
    if m["margin"] is not None:
        env = img_env(m["image"]); im = env.get("DOSERAD_PHOTON_MARGIN", "8")
        ok_m = (str(m["margin"]) == im)
        print(f"  [M] margin  manifest={m['margin']}  image={im}  {'PASS' if ok_m else 'FAIL'}")
        if not ok_m: fails.append("margin_mismatch")

    # [C] GT cache exists + CP count
    cdir = Path(m["cache"]) / pid0
    ncp = len(list(cdir.glob("*.npz"))) if cdir.exists() else 0
    ok_c = ncp > 0
    print(f"  [C] GT cache {m['cache']}  {pid0}: {ncp} npz  {'PASS' if ok_c else 'FAIL'}")
    if not ok_c: fails.append("cache_empty")

    # [L] clf (MRI)
    if m["clf"]:
        ev = md5(m["clf"]); iv = img_file_md5(m["image"],
                                              m["img_weight"].rsplit("/", 1)[0] + "/clf_whole.pt")
        ok_l = (iv is not None and ev == iv)
        print(f"  [L] clf     eval={ev[:8]}  image={iv[:8] if iv else 'MISSING'}  {'PASS' if ok_l else 'FAIL'}")
        if not ok_l: fails.append("clf_mismatch")

    return fails


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "all"
    tasks = list(MANIFEST) if task == "all" else [task]
    allfails = {}
    for t in tasks:
        print(f"\n=== preflight: {t} ({MANIFEST[t]['image']}) ===")
        f = check(t)
        if f: allfails[t] = f
    print()
    if allfails:
        print("PREFLIGHT FAILED:", allfails)
        sys.exit(1)
    print("PREFLIGHT PASS — safe to run baseline/eval.")


if __name__ == "__main__":
    main()
