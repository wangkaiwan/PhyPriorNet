import os,sys,json,numpy as np,torch
sys.path.insert(0,"scripts")
import train_sct_classifier as T
val=json.load(open("/home/kaiwang/doserad2026_workdir/sct_data_1x1x3.json"))["val"]
DEV="cuda"
def ev(ckpt,patch):
    net=T.model().to(DEV); net.load_state_dict(torch.load(ckpt,map_location="cpu")["net"])
    d=T.validate(net,val,patch,DEV,whole=False)   # same path as training best-selection
    return d
r128=ev("/data/kwang/sct_classify_runs/clf_1x1x3/best.pt",(128,128,128))
rani=ev("/data/kwang/sct_classify_runs/clf_1x1x3_aniso/best.pt",(64,192,192))
def show(t,d): print(f"  {t:8s} air {d[0]:.3f} lung {d[1]:.3f} soft {d[2]:.3f} bone {d[3]:.3f} | mean-fb {d[1:].mean():.3f}")
print("val Dice (trainer's exact validate):")
show("128cube",r128); show("aniso",rani)
print(">>> WINNER:", "128cube" if r128[1:].mean()>rani[1:].mean() else "aniso")
