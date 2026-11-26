"""Replay selected methods and write one row per model, task and method."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from search import CACHE, OUT, TIMES

here = Path(__file__).resolve().parent
for script in ["search", "confidence_information", "projected_leader", "candidate_gain"]:
    subprocess.run([sys.executable, str(here / (script + ".py"))], check=True)

tables = []
for filename, ids in [("confidence_information_cells.csv", [261,262,263,264]),
                      ("projected_leader_cells.csv", [292,293,294]),
                      ("candidate_gain_cells.csv", [327,328,329,330])]:
    table = pd.read_csv(OUT / filename)
    tables.append(table[table.id.isin(ids) & table.split.eq("test")])
references = []
for path in sorted(CACHE.glob("*.npz")):
    with np.load(path, allow_pickle=True) as data:
        test = data["split"] == "test"
        stop = (data["margin"][test] >= 3) & data["valid"][test]
        stop[:, -1] = True
        for number, end in [(12, stop.argmax(axis=1)), (0, np.full(test.sum(), len(TIMES)-1))]:
            hit = data["hit"][test][np.arange(test.sum()), end]
            references.append(dict(id=number, cell=path.stem, total=len(hit), correct=hit.sum(),
                                   accuracy=hit.mean(), solutions=TIMES[end].mean()))
columns = ["id", "cell", "total", "correct", "accuracy", "solutions"]
results = pd.concat([*tables, pd.DataFrame(references)])[columns].sort_values(["id", "cell"])
results.to_csv(OUT / "by_model_task.csv", index=False)
print(results.groupby("id")[["accuracy", "solutions"]].mean())
