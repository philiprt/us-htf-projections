import pickle
import analysis_module as anlyz

stats_path = "./ensemble_stats/"
os.makedirs(stats_path, exist_ok=True)

# get meta for list of NOS stations to analyze (excluding a few)
exclude = [
    1619910,  # midway
    8557380,  # lewes
    8467150,  # bridgeport
]
stations = anlyz.station_list(exclude=exclude)

# choose of time of day
tod = [0, 23]
# tod = [6, 18]

# loop over each flooding threshold
for thrsh in ["minor", "moderate"]:

    # loop over each SLR scenario
    for scn in ["int", "int_low", "int_high", "kopp"]:

        # perform statistical analysis and save results
        exp_str = anlyz.experiment_string(scn, thrsh, tod)

        analysis = []
        pbar = anlyz.ProgressBar(stations.index.size, description=exp_str)
        for idx, sta in stations.iterrows():
            analysis.append(
                anlyz.station_analysis(
                    sta, threshold=thrsh, scenario=scn, time_of_day=tod
                )
            )
            pbar.update()

        fname = stats_path + exp_str + ".pickle"
        with open(fname, "wb") as f:
            pickle.dump(analysis, f)
