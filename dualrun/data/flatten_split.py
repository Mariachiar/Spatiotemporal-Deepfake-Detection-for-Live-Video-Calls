# flatten_split.py
import json, sys
inp, outp = sys.argv[1], sys.argv[2]
d = json.load(open(inp))
assert "splits" in d and all(k in d["splits"] for k in ("train","val","test"))
def flat(split):
    clips=[]
    for v in d["splits"][split]:
        for t in v["tracks"]:
            clips += t.get("clips",[])
    return clips
flat_idx = {"train": flat("train"), "val": flat("val"), "test": flat("test")}
json.dump(flat_idx, open(outp,"w"), indent=2)
print({k: len(v) for k,v in flat_idx.items()})
