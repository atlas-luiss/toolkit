"""Mean ± std over seeds of every <name>_s<seed>/results.json under a folder.

    python experiments/summarize.py runs/repro
    python experiments/summarize.py runs/reference
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

root = sys.argv[1]
groups = defaultdict(list)
for f in sorted(glob.glob(os.path.join(root, "*_s[0-9]*", "results.json"))):
    name = re.sub(r"_s\d+$", "", os.path.basename(os.path.dirname(f)))
    groups[name].append(json.load(open(f)))


def ms(rs, k):
    v = [r[k] for r in rs if r.get(k) is not None]
    return f"{np.mean(v):6.2f} ± {np.std(v):4.2f}" if v else "     -       "


print(f"{'run':26s} {'n':>2s}  {'final acc':>13s}  {'avg incr acc':>13s}  {'forgetting':>13s}  {'routing':>13s}  model / preprocess")
for name, rs in groups.items():
    c = rs[0]["config"]
    print(f"{name:26s} {len(rs):2d}  {ms(rs, 'avg_accuracy')}  {ms(rs, 'avg_incremental_accuracy')}  "
          f"{ms(rs, 'avg_forgetting')}  {ms(rs, 'final_routing_accuracy')}  {c['model']} / {c.get('preprocess')}")
