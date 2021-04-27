# ---------------------------------------------------------------------------

import json
import pickle

import pandas as pd

# ---------------------------------------------------------------------------

fname = "./noaa_scenarios/NOAA_SLR_scenarios.pickle"
with open(fname, "rb") as f:
    scn = pd.read_pickle(f)

fname = "./noaa_scenarios/stations.csv"
with open(fname, "rb") as f:
    sta = pd.read_csv(f)

sta["psmsl_id"] = None

meta = {}

remove = [8747437, 8656483, 8652587, 9440910, 8631044, 8636580]  # no kopp
remove = remove + [8771013]
# remove = remove + [1619910]  # midway

rmv_idx = [n for n in sta.index if sta.loc[n, "noaa id"] in remove]

sta.drop(rmv_idx, inplace=True)
sta.index = range(sta.shape[0])

for s in sta.index:

    d = (sta.loc[s].lat - scn["meta"].lat) ** 2 + (
        sta.loc[s].lon - scn["meta"].lon
    ) ** 2

    sta.loc[s, "psmsl_id"] = d.idxmin()
    # print([sta.loc[s, 'name'], scn['meta'].loc[d.idxmin(), 'name']])

    uid_str = str(sta.loc[s, "noaa id"])
    meta[uid_str] = {}

    meta[uid_str]["name"] = sta.loc[s, "name"]
    meta[uid_str]["lat"] = sta.loc[s].lat
    meta[uid_str]["lon"] = sta.loc[s].lon

# ---------------------------------------------------------------------------

with open("./stations.pickle", "wb") as f:
    pickle.dump(sta, f)

fname = "../projections/output/json/stations.json"
with open(fname, "w") as f:
    json.dump(meta, f, indent=4)

# ---------------------------------------------------------------------------
