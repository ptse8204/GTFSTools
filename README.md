# GTFSTools

A tool to find the optimal route and theoretical time for transit speedruns.

## Current Features
- A transit network planner, capable of finding the best time to go through all the stops or all edges, or the maximum distance in the network, and so much more...
- A GTFS schedule viewer designed to quickly and easily digestible timetable information for planning your best run
- A network view just to see whether things load correctly
- A search function for all the stops and routes

More improvements and features will be added in the future.

## Notes on using the planner


## Setup

### On your local machine (recommended)
The tool is recommended to run on your local machine because the calculation itself is computationally expensive. To run the code, follow the following instructions:


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

## Out of the Box option
You may try using this version, no guarantees on the performance: https://gtfs-explorer.streamlit.app/


## An example run
Here is a screenshot of the result of a run, which shows an optimal transit run in Vancouver's Skytrain.

![](<screenshots/Screenshot 2025-09-18 at 14.46.42.png>)

The current leaderboard time is 2 hrs and 44 mins. As shown here: https://www.transitruns.org/vancouver/


