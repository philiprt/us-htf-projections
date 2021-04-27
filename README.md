# Rapid increases and extreme months in projections of United States high-tide flooding
Thompson et al. (2021), *Nature Climate Change*, doi: [TBD](https://www.doi.org)

This repository contains the code and notebooks used to produce the high-tide-flooding (HTF) projections and associated analysis in the paper cited above. To keep the size of the repository manageable, most of the output from the various steps in the calculations are not included, but all information needed to reproduce figures from the paper is present. Output for each analysis step can be obtained by using the repository code in the order described below. See the Projections and Analysis sections for more information.

For best results, run this code in a virtual environment generated from the python_environment.txt file in the root directory of this repository.

## 1. Data and inputs

The data/ directory contains the NOAA SLR scenarios and hourly tide gauge data needed for the analysis. The tide gauge data is not included in this repository, but there is a script (data/tide_gauge/nos_hourly.py) that will download the necessary hourly TG data from the NOAA CO-OPS API.

The script data/stations.py extracts a list of stations and meta data from the NOAA SLR scenarios to be used in the following steps.

## 2. Ensembles of future monthly mean sea level variability

The first step towards HTF projections is to produce ensembles of monthly and annual mean sea level variability for the 21st century based on observed variability in tide gauges. The script projections/gp_monthly_means.py produces this ensemble for each station. The ensemble tidal predictions (see below) utilize these ensembles of mean sea level variability, which is why these must be produced first.

## 3. Ensemble tidal predictions

The next step toward the HTF projections is to produce ensemble tidal predictions that account for uncertainty in the constituent amplitudes and modulation cycles, as well as correlations between constituent amplitudes and mean sea level. First, calculate distributions of future tidal constituents by running tides/constituent_gp_regressions.py. Then produce the ensemble tidal predictions by running tides/probabilistic_tides.py. For additional plots showing results of tide calculations, run tides/analysis.py.

## 4. Statistical model for HTF frequency

The statistical model relating mean sea level, tides, and HTF frequency is generated for each station by running the script projections/thrsh_exprmnts_bbmodel_2gcdf.py. The statistical model is generated for each calendar month (all Januarys, all Februarys, etc.), giving twelve statistical models for each station.

## 5. Get flooding thresholds

Calculate NOAA derived flooding thresholds (Sweet et al., 2018) by running projections/thresholds.py. This script requires an internet connection to connect to the CO-OPS API.

## 6. Projections

The HTF projections for each permutation of location, threshold, and SLR scenario are produced by the script projections/projections.py. The result is a separate ensemble of projections for each calendar month. This format makes the ensemble somewhat unwieldy, but the necessary code to load these ensembles into a single matrix is included in this repository. See the Analysis section below for more information.

The ensemble itself is not included in this repository due to its size (>60 GB), but it can be generated using the code and steps outlined here.

## 7. Analysis

The code in the analysis/ directory performs calculations on the ensemble projections and creates the figures presented in the paper. Loading the ensembles for individual months into a single matrix can be done by importing analysis_module.py and utilizing the xdys_monthly_ensemble function.

The figures and tables in the paper are based on a set of calculations performed by first executing analysis/ensemble_stats.py. The results of these calculations are included in this repository in analysis/ensemble_stats/. Figure 2 regarding the nodal cycle requires first executing analysis/nodal_cycle.py. The results of these calculations are included in this repository in analysis/nodal_cycle/.

The figures and tables are created in the notebook analysis/us_htf_projections.ipynb, which imports code from files in analysis/figure_code. The figures and tables are saved to analysis/figures_and_tables/.
