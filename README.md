# GTFSTools

A screenshot of the GTFS Explorer application displaying an optimal transit route in Vancouver's Skytrain system. The main interface shows a summary of the selected plan, including start time, total elapsed time, number of stops, and edges. A green notification indicates full coverage satisfied, listing the start time, station, total stops, edges, elapsed time, ride and wait times, and total distance. Below, a table details each segment of the trip with columns for departure time, origin station, trip ID, route,

The code requires the use of Python, and the packages listed in `requirment.txt`. 
You can install the packages using pip:

```
pip install -r requirements.txt
``` 

To use this code, run the following code:
```
streamlit run gtfs_explorer_stable_v1.0.1.py
```

More improvements and features will be added in the future.

## An example run
Here is a screenshot of a result of a run, which show an optimal transit run in Vancouver's Skytrain.

![](<screenshots/Screenshot 2025-09-18 at 14.46.42.png>)

The current leaderboard time is 2hrs and 44 mins. As shown here: https://www.transitruns.org/vancouver/
