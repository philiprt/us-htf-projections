# ---------------------------------------------------------------------------

import requests
import json
import datetime as dt
import numpy as np
import pandas as pd
import netCDF4 as nc
import os

# ---------------------------------------------------------------------------
# target directory

nos_dir = "./nos_hourly/"
os.makedirs(nos_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# get list of stations

url = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?"
stations = json.loads(requests.get(url).text)

stations = [s for s in stations["stations"] if s["tidal"] & ~s["greatlakes"]]

# ---------------------------------------------------------------------------
# define static url segments

url_base = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
url_end = "datum=MHHW&time_zone=gmt&units=metric&format=json"
prdct_wl = "product=hourly_height&"
prdct_tp = "product=predictions&"
res = "interval=h&"

# ---------------------------------------------------------------------------
# meta fields to include

meta_fields = [
    "id",
    "name",
    "state",
    "latitude",
    "longitude",
    "timezone",
    "timezonecorr",
    "tideType",
    "affiliations",
    "nearby",
    "details",
    "sensors",
    "floodlevels",
    "datums",
    "supersededdatums",
    "harmonicConstituents",
    "benchmarks",
    "tidePredOffsets",
    "products",
    "disclaimers",
    "notices",
]

# ---------------------------------------------------------------------------
# time units for .nc file

t_units = "days since 1900-01-01 00:00:00"

# ---------------------------------------------------------------------------
# get current year

this_year = dt.datetime.today().year

# ---------------------------------------------------------------------------
# first year to download if starting from scratch

first_year = 1900

# ---------------------------------------------------------------------------
# loop over stations

restart = 114

N = len(stations)
for k, sta in enumerate(stations[restart:]):

    # if sta["id"] not in ["8726520", "8443970", "9410230", "1612340"]:
    #     continue

    k0 = k + restart

    print("")
    print(
        "["
        + "{:03}".format(k0 + 1)
        + " of "
        + str(N)
        + "] "
        + sta["id"]
        + ": "
        + sta["name"]
        + ", "
        + sta["state"]
    )

    # ----------------------------------------------------------------------
    # prep meta data

    sta["latitude"] = sta.pop("lat")
    sta["longitude"] = sta.pop("lng")
    meta = {
        m: sta[m] if type(sta[m]) is not dict else sta[m]["self"]
        for m in meta_fields
    }
    meta = {
        m: "None" if (meta[m] is False) | (meta[m] is None) else meta[m]
        for m in meta
    }

    # ----------------------------------------------------------------------
    # keep trying until station successfully downloads

    success = False
    attempts = 1

    while success is False:

        try:

            # --------------------------------------------------------------
            # netcdf filename

            nc_fname = nos_dir + sta["id"] + ".nc"

            # --------------------------------------------------------------
            # if file exists, prep for appending

            if os.path.exists(nc_fname):

                append = True

                ncds = nc.Dataset(nc_fname, mode="a")

                nc_t = ncds["time"]
                nc_obs = ncds["observed"]
                nc_prd = ncds["predicted"]

                jdt_last = nc_t[-1]
                year_of_last = nc.num2date(
                    jdt_last, units=ncds["time"].units
                ).year
                years = range(year_of_last, this_year + 1)

            # --------------------------------------------------------------
            # if file does not exist, create ntecdf file structure

            else:

                append = False

                ncds = nc.Dataset(
                    nc_fname, mode="w", clobber=True, format="NETCDF4_CLASSIC"
                )

                dim_time = ncds.createDimension("time", None)

                nc_t = ncds.createVariable("time", "f8", dimensions=("time",))
                nc_t.setncatts({"units": t_units})

                nc_obs = ncds.createVariable(
                    "observed", "f4", dimensions=("time",)
                )
                nc_obs.setncatts({"units": "meters", "datum": "MHHW"})

                nc_prd = ncds.createVariable(
                    "predicted", "f4", dimensions=("time",)
                )
                nc_prd.setncatts({"units": "meters", "datum": "MHHW"})

                ncds.setncatts(meta)

                years = range(first_year, this_year + 1)

            # --------------------------------------------------------------
            # download data by year and aggregate into dataframe

            df = pd.DataFrame()

            for yr in years:

                d1 = "begin_date=" + str(yr) + "0101&"
                d2 = "end_date=" + str(yr) + "1231&"
                sid = "station=" + sta["id"] + "&"

                url = url_base + d1 + d2 + sid + prdct_wl + url_end
                data = json.loads(requests.get(url).text)
                if "error" in data:
                    continue
                else:
                    print("  " + str(yr))
                    times = [d["t"] for d in data["data"]]
                    heights = [
                        float(d["v"]) if len(d["v"]) > 0 else None
                        for d in data["data"]
                    ]
                    obs = pd.Series(heights, index=pd.to_datetime(times))

                url = url_base + d1 + d2 + sid + prdct_tp + res + url_end
                data = json.loads(requests.get(url).text)
                if "error" in data:
                    prd = pd.Series(None, index=pd.to_datetime(times))
                else:
                    times = [d["t"] for d in data["predictions"]]
                    predict = [float(d["v"]) for d in data["predictions"]]
                    prd = pd.Series(predict, index=pd.to_datetime(times))
                    prd = prd.loc[obs.index]

                df = df.append(
                    pd.DataFrame({"observed": obs, "predicted": prd})
                )

            # --------------------------------------------------------------
            # julian dates of new data

            jdts = nc.date2num(df.index.to_pydatetime(), t_units)

            # --------------------------------------------------------------
            # insert data into .nc

            if append:

                idx_first_new = np.where(jdts == jdt_last)[0][0] + 1
                df_new = df.iloc[idx_first_new:, :]

                idx_last_old = ncds["time"].shape[0]
                nc_t[idx_last_old + 1 :] = jdts[idx_first_new:]
                nc_obs[idx_last_old + 1 :] = df_new["observed"].values
                nc_prd[idx_last_old + 1 :] = df_new["predicted"].values

            else:

                nc_t[:] = jdts
                nc_obs[:] = df["observed"].values
                nc_prd[:] = df["predicted"].values

            # --------------------------------------------------------------
            # close .nc file

            ncds.close()
            success = True

        except Exception as e:
            print(e)
            os.remove(nc_fname)
            attempts += 1
            print("Download failed")
            print("\nTrying again (attempt #" + str(attempts) + ")")

    # end while loop over attempts

    # -----------------------------------------------------------------------

# end loop over stations

# ---------------------------------------------------------------------------
