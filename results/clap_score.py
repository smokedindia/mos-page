#!/usr/bin/env python3
"""CLAP score baseline for BATON clips.

CLAP score = cosine similarity between the LAION-CLAP text embedding of the
caption and the audio embedding of the generated clip (the metric reported as
"CLAP score" in AudioLDM / Tango).  Same input contract as eval_aqa.py: reads
rewards.json from each audio dir, writes {label}_clap.json to --output-dir.
"""
import argparse, json, os, numpy as np, soundfile as sf, librosa, torch
from transformers import ClapModel, ClapProcessor

SR = 48000


def load_wav(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--model", default="laion/clap-htsat-unfused")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = ClapModel.from_pretrained(args.model).to(args.device).eval()
    proc = ClapProcessor.from_pretrained(args.model)

    for d in args.dirs:
        label = os.path.basename(os.path.normpath(d))
        entries = json.load(open(os.path.join(d, "rewards.json")))
        results = []
        for i in range(0, len(entries), args.batch_size):
            chunk = entries[i:i + args.batch_size]
            audios = [load_wav(os.path.join(d, e["output_file"])) for e in chunk]
            texts = [e["prompt"] for e in chunk]
            ai = proc(audios=audios, sampling_rate=SR, return_tensors="pt")
            ti = proc(text=texts, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                ae = model.get_audio_features(**{k: v.to(args.device) for k, v in ai.items()})
                te = model.get_text_features(**{k: v.to(args.device) for k, v in ti.items()})
            ae = torch.nn.functional.normalize(ae, dim=-1)
            te = torch.nn.functional.normalize(te, dim=-1)
            sim = (ae * te).sum(-1).float().cpu().numpy()
            for e, s in zip(chunk, sim):
                results.append({"index": e["index"], "prompt": e["prompt"],
                                "output_file": e["output_file"], "clap": float(s)})
            print(f"  {label}: {len(results)}/{len(entries)}", flush=True)

        out = os.path.join(args.output_dir, f"{label}_clap.json")
        vals = np.array([r["clap"] for r in results])
        json.dump({"label": label, "model": args.model, "n": len(results),
                   "mean_clap": float(vals.mean()), "results": results},
                  open(out, "w"), indent=2)
        print(f"{label}: n={len(results)} mean_clap={vals.mean():.4f} -> {out}", flush=True)
    print("CLAP_DONE")


if __name__ == "__main__":
    main()
