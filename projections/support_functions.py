# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

import numpy as np
import time

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# create a string describing station for directories/filenames


def station_string(sta):
    loc_str = "".join(
        [
            c if c != " " else "_"
            for c in [c for c in sta["name"].lower() if c not in [",", "."]]
        ]
    )

    sta_str = loc_str + "_" + str(sta["noaa id"])

    return sta_str


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class LoopInfo:
    def __init__(self, N):
        self.N = N
        self.count = 0
        self.time_elapsed = 0
        self.t0 = 0

    def begin_iteration(self, sta):
        self.count += 1
        print(
            "\n"
            + str(self.count)
            + " of "
            + str(self.N)
            + ". "
            + sta["name"]
            + " (NOAA ID: "
            + str(sta["noaa id"])
            + ")"
        )
        self.t0 = time.time()

    def end_iteration(self, failed=False):
        dt = time.time() - self.t0  # seconds
        self.time_elapsed += dt
        tm_sta_str = "{:0.1f}".format(dt / 60)
        tm_avg = self.time_elapsed / (60 * self.count)
        tm_avg_str = "{:0.1f}".format(tm_avg)
        tm_rem_mns = tm_avg * (self.N - self.count)
        tm_rem_hrs = int(np.floor(tm_rem_mns / 60))
        tm_rem_mns -= tm_rem_hrs * 60
        tm_rem_hrs_str = str(tm_rem_hrs)
        tm_rem_mns_str = "{:0.1f}".format(tm_rem_mns)
        print("Time this station: " + tm_sta_str + " mins")
        print("Average time/station so far: " + tm_avg_str + " mins")
        print(
            "Estimated time remaining: "
            + tm_rem_hrs_str
            + " hrs, "
            + tm_rem_mns_str
            + " mins"
        )
        if failed:
            print("**************")
            print("*** FAILED ***")
            print("**************")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Make a progress bar for loops


class ProgressBar:
    def __init__(self, total, description=None):

        self.total = total
        self.decimals = 1
        self.length = 50
        self.fill = ">"
        self.iteration = 0
        self.intervals = np.ceil(np.linspace(1, self.total, self.length))
        self.percent = np.linspace(0, 100, self.length + 1)[1:].astype(int)

        unq, cnt = np.unique(self.intervals, return_counts=True)
        idx = np.array([np.where(self.intervals == u)[0][-1] for u in unq])
        self.intervals = self.intervals[idx]
        self.percent = self.percent[idx]
        self.cumcnt = np.cumsum(cnt)

        if description:
            print(description)
        print("|%s| 0%% complete" % (" " * self.length), end="\r")

    def update(self):

        self.iteration += 1

        if self.iteration in self.intervals:
            prog = np.where(self.intervals == self.iteration)[0][-1]
            pctstr = str(self.percent[prog]) + "%"
            bar = self.fill * self.cumcnt[prog] + " " * (
                self.length - self.cumcnt[prog]
            )
            print("\r|%s| %s complete" % (bar, pctstr), end="\r")
            if self.iteration == self.total:
                print()  # blank line on completion


# ---------------------------------------------------------------------------
