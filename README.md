# GTFSTools

A tool to find the optimal route and theoretical time for transit runs.

## Setup

### On your local machine (recommended)
The code requires the use of Python, and the packages listed in `requirment.txt`. 
You can install the packages using pip:

```
pip install -r requirements.txt
``` 

To use this code, run the following code:
```
streamlit run gtfs_explorer_stable_v1.0.1.py
```

The code also requires the use of GTFS files, which you may download from: https://www.transit.land/feeds?search

More improvements and features will be added in the future.

## No setup
You may try using this version, no guarantee on the performance: https://gtfs-explorer.streamlit.app/


## An example run
Here is a screenshot of a result of a run, which show an optimal transit run in Vancouver's Skytrain.

![](<screenshots/Screenshot 2025-09-18 at 14.46.42.png>)

The current leaderboard time is 2hrs and 44 mins. As shown here: https://www.transitruns.org/vancouver/
