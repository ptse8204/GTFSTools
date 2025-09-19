# gtfs_explorer.py
# Streamlit GTFS Explorer — Unified Coverage Planner + Robust Backtracking + Search (Fast + Smart Starts)
import io, zipfile, json, math, re, unicodedata, difflib, datetime as dt, os, hashlib, collections
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
import time

try:
    from queue import SimpleQueue, Empty
except Exception:
    from queue import Queue as SimpleQueue
    from queue import Empty

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from time import perf_counter

# Imports for parallelization
from joblib import Parallel, delayed
import contextlib
from tqdm import tqdm

# Streamlit-friendly tqdm context manager (module-level) -----------------------
@contextlib.contextmanager
def streamlit_tqdm(total):
    pbar = tqdm(total=total, leave=False, desc="Parallel Search")
    try:
        yield pbar
    finally:
        pbar.close()

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object  # for stable signatures
import streamlit as st
import pydeck as pdk

# Map style resolver (ensures we pass a valid Mapbox style URL to pydeck)
def _resolve_map_style(style_key):
    # Try user-provided MAP_STYLES dict first
    try:
        styles = globals().get("MAP_STYLES", {})
        if isinstance(styles, dict) and style_key in styles:
            return styles[style_key]
    except Exception:
        pass
    # Friendly defaults
    defaults = {
        "Light": "mapbox://styles/mapbox/light-v11",
        "Dark": "mapbox://styles/mapbox/dark-v11",
        "Outdoors": "mapbox://styles/mapbox/outdoors-v12",
        "Streets": "mapbox://styles/mapbox/streets-v12",
        "Satellite": "mapbox://styles/mapbox/satellite-streets-v12",
        "light": "mapbox://styles/mapbox/light-v11",
        "dark": "mapbox://styles/mapbox/dark-v11",
        "outdoors": "mapbox://styles/mapbox/outdoors-v12",
        "streets": "mapbox://styles/mapbox/streets-v12",
        "satellite": "mapbox://styles/mapbox/satellite-streets-v12",
    }
    return defaults.get(str(style_key), defaults["Light"])


# ---- Optional geometry for trimming
try:
    from shapely.geometry import shape, Point
    from shapely.ops import unary_union
    HAVE_SHAPELY = True
except Exception:
    HAVE_SHAPELY = False

# ---- Optional GPU (CuPy for CUDA, PyTorch for Apple Silicon)
try:
    import cupy as cp
    HAVE_CUPY = (cp.cuda.runtime.getDeviceCount() > 0)
except Exception:
    cp = None
    HAVE_CUPY = False

try:
    import torch
    HAVE_TORCH = True
    # Check for Apple Silicon MPS support
    HAVE_MPS = HAVE_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
except Exception:
    torch = None
    HAVE_TORCH = False
    HAVE_MPS = False

# This will be set by the UI. Default to 'cpu'.
ACCELERATOR: str = "cpu"

# Global (set by UI after edges are built)
CORRIDOR_OF_STOP = {}
CORRIDOR_NODES = {}
_ADJ_UNDIR = {}

# ---------- helpers ----------
def parse_time_to_seconds(s: str) -> Optional[int]:
    """Parse 'HH:MM:SS' with HH possibly >= 24. Return seconds or None."""
    if s is None or (isinstance(s, float) and np.isnan(s)): return None
    s = str(s).strip()
    try:
        parts = s.split(":")
        if len(parts) != 3:
            return None
        h, m, sec = parts
        return int(h)*3600 + int(m)*60 + int(sec)
    except Exception:
        return None

def seconds_to_hhmm(seconds: int) -> str:
    if seconds is None: return "-"
    h = seconds // 3600; m = (seconds % 3600) // 60
    return f"{h:02d}:{m:02d}"

def fmt_hms(seconds: int) -> str:
    if seconds is None: return "-"
    h = seconds // 3600; m = (seconds % 3600) // 60; s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def today_local_date() -> dt.date:
    return dt.datetime.now().date()

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))

def fmt_km(m): return f"{m/1000.0:,.2f} km"

def df_sig(df: Optional[pd.DataFrame], cols: Optional[List[str]]=None) -> int:
    """Stable 32-bit signature for caching."""
    if df is None or (hasattr(df, "empty") and df.empty):
        return 0
    try:
        if cols is None:
            cols = list(df.columns)
        h = int(hash_pandas_object(df[cols], index=True).sum()) & 0xFFFFFFFF
    except Exception:
        h = int(df.shape[0] * 1_000_003 + len(cols) * 97) & 0xFFFFFFFF
    return h

# For caching: make a quick signature of the edge set used in planning
def edges_signature(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    cols = [c for c in ["from_stop","to_stop","dep_s","arr_s","route_id","route_type"] if c in df.columns]
    try:
        h = int(hash_pandas_object(df[cols], index=False).sum()) & 0xFFFFFFFF
    except Exception:
        dep_sum = pd.to_numeric(df.get("dep_s", pd.Series(dtype="float64")), errors="coerce").fillna(0).sum()
        h = int(df.shape[0] * 1_000_003 + int(dep_sum) % 1_000_000_007) & 0xFFFFFFFF
    return h

# GPU/CPU unified helper for inc elapsed
def _inc_elapsed_vector(
    dep_arr_np: np.ndarray,
    arr_arr_np: np.ndarray,
    cur_time: int,
    prev_route_id: Optional[str],
    route_ids_series: pd.Series,
    zero_wait_same_route: bool,
    accelerator: str = "cpu",
) -> np.ndarray:
    """
    Return inc_elapsed (wait_eff + ride) as int32 numpy array.
    Uses CUDA or Apple Metal (MPS) if available and selected.
    """
    # CUDA / CuPy implementation
    if accelerator == "cuda" and HAVE_CUPY:
        d = cp.asarray(dep_arr_np, dtype=cp.int32)
        a = cp.asarray(arr_arr_np, dtype=cp.int32)
        wait = cp.maximum(d - int(cur_time), 0)
        if zero_wait_same_route and (prev_route_id is not None):
            same = cp.asarray(route_ids_series.astype(str).to_numpy() == str(prev_route_id))
            wait = cp.where(same, 0, wait)
        ride = cp.maximum(a - d, 0)
        inc = (wait + ride).astype(cp.int32)
        return cp.asnumpy(inc)

    # Apple Metal / PyTorch implementation
    if accelerator == "mps" and HAVE_MPS:
        device = torch.device("mps")
        d = torch.from_numpy(dep_arr_np.astype(np.int32)).to(device)
        a = torch.from_numpy(arr_arr_np.astype(np.int32)).to(device)
        wait = torch.clamp(d - int(cur_time), min=0)
        if zero_wait_same_route and (prev_route_id is not None):
            same_np = (route_ids_series.astype(str).to_numpy() == str(prev_route_id))
            same = torch.from_numpy(same_np).to(device)
            wait = torch.where(same, torch.tensor(0, device=device, dtype=torch.int32), wait)
        ride = torch.clamp(a - d, min=0)
        inc = (wait + ride)
        return inc.cpu().numpy()

    # CPU fallback (NumPy)
    dep = dep_arr_np.astype(np.int32)
    arr = arr_arr_np.astype(np.int32)
    wait_real = np.clip(dep - np.int32(cur_time), 0, None)
    if zero_wait_same_route and (prev_route_id is not None):
        same = (route_ids_series.astype(str).to_numpy() == str(prev_route_id))
        wait_eff = np.where(same, 0, wait_real)
    else: 
        wait_eff = wait_real
    ride = np.clip(arr - dep, 0, None)
    return (wait_eff + ride).astype(np.int32)

# ---------- GTFS ----------
@dataclass
class GTFSData:
    stops: pd.DataFrame
    stop_times: pd.DataFrame
    trips: pd.DataFrame
    routes: pd.DataFrame
    calendar: Optional[pd.DataFrame]
    calendar_dates: Optional[pd.DataFrame]
    gtfs_sig: int

def _zip_bytes_sig(zip_bytes: bytes) -> int:
    return int.from_bytes(hashlib.sha1(zip_bytes).digest()[:4], "big")

@st.cache_resource(show_spinner=False)
def load_gtfs_from_bytes(zip_bytes: bytes) -> GTFSData:
    """Cached loader: parses GTFS zip bytes."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        def read_csv(name, **kwargs):
            try:
                with z.open(name) as f: return pd.read_csv(f, **kwargs)
            except KeyError: return None
        stops = read_csv("stops.txt", dtype=str)
        stop_times = read_csv("stop_times.txt", dtype=str)
        trips = read_csv("trips.txt", dtype=str)
        routes = read_csv("routes.txt", dtype=str)
        calendar = read_csv("calendar.txt", dtype=str)
        calendar_dates = read_csv("calendar_dates.txt", dtype=str)

    for need, df in [("stops.txt",stops), ("stop_times.txt",stop_times), ("trips.txt",trips)]:
        if df is None: raise ValueError(f"GTFS missing {need}")

    for col in ["stop_id","stop_name","stop_lat","stop_lon"]:
        if col not in stops.columns: raise ValueError(f"stops.txt missing {col}")
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops = stops.dropna(subset=["stop_lat","stop_lon"]).drop_duplicates("stop_id").reset_index(drop=True)

    for col in ["trip_id","route_id"]:
        if col not in trips.columns: raise ValueError(f"trips.txt missing {col}")
    if "service_id" not in trips.columns: trips["service_id"] = "default_service"
    if "trip_headsign" not in trips.columns: trips["trip_headsign"] = ""


    if routes is None:
        routes = pd.DataFrame(columns=["route_id","route_short_name","route_long_name","route_type"])
    for c in ["route_id","route_short_name","route_long_name","route_type"]:
        if c not in routes.columns: routes[c] = ""
    routes["route_type"] = pd.to_numeric(routes["route_type"], errors="coerce").astype("Int64")

    for col in ["trip_id","stop_id","stop_sequence"]:
        if col not in stop_times.columns: raise ValueError(f"stop_times.txt missing {col}")
    if "arrival_time" not in stop_times.columns: stop_times["arrival_time"] = None
    if "departure_time" not in stop_times.columns: stop_times["departure_time"] = None

    # Preserve raw parsed seconds and build effective seconds used for edges
    stop_times["arr_s_raw"] = stop_times["arrival_time"].apply(parse_time_to_seconds)
    stop_times["dep_s_raw"] = stop_times["departure_time"].apply(parse_time_to_seconds)

    stop_times["arr_s"] = stop_times["arr_s_raw"]
    stop_times["dep_s"] = stop_times["dep_s_raw"]
    stop_times["arr_s"] = np.where(pd.isna(stop_times["arr_s"]), stop_times["dep_s"], stop_times["arr_s"])
    stop_times["dep_s"] = np.where(pd.isna(stop_times["dep_s"]), stop_times["arr_s"], stop_times["dep_s"])

    stop_times["stop_sequence"] = pd.to_numeric(stop_times["stop_sequence"], errors="coerce")
    stop_times = stop_times.dropna(subset=["arr_s","dep_s","stop_sequence"]).copy()
    stop_times["stop_sequence"] = stop_times["stop_sequence"].astype(int)

    sig = (
        df_sig(stops, ["stop_id","stop_name","stop_lat","stop_lon"]) ^
        df_sig(stop_times, ["trip_id","stop_id","stop_sequence","arr_s","dep_s"]) ^
        df_sig(trips, ["trip_id","route_id","service_id"]) ^
        df_sig(routes, ["route_id","route_type"])
    )
    return GTFSData(stops, stop_times, trips, routes, calendar, calendar_dates, sig)

def load_gtfs_from_zip(upload: io.BytesIO) -> GTFSData:
    return load_gtfs_from_bytes(upload.getvalue())

def service_ids_for_date(gtfs: GTFSData, day: dt.date) -> Optional[set]:
    active = set()
    if gtfs.calendar is not None and not gtfs.calendar.empty:
        cal = gtfs.calendar.copy()
        need = ["service_id","monday","tuesday","wednesday","thursday","friday","saturday","sunday","start_date","end_date"]
        if all(c in cal.columns for c in need):
            dnum = int(day.strftime("%Y%m%d"))
            wd_map = {0:"monday",1:"tuesday",2:"wednesday",3:"thursday",4:"friday",5:"saturday",6:"sunday"}
            wcol = wd_map[day.weekday()]
            cal[wcol] = pd.to_numeric(cal[wcol], errors="coerce").fillna(0).astype(int)
            cal["start_date"] = pd.to_numeric(cal["start_date"], errors="coerce")
            cal["end_date"] = pd.to_numeric(cal["end_date"], errors="coerce")
            mask = (cal[wcol]==1) & (cal["start_date"]<=dnum) & (cal["end_date"]>=dnum)
            active.update(cal.loc[mask,"service_id"].astype(str).tolist())
    if gtfs.calendar_dates is not None and not gtfs.calendar_dates.empty:
        cdx = gtfs.calendar_dates.copy()
        need = ["service_id","date","exception_type"]
        if all(c in cdx.columns for c in need):
            dnum = int(day.strftime("%Y%m%d"))
            cdx["date"] = pd.to_numeric(cdx["date"], errors="coerce")
            cdx["exception_type"] = pd.to_numeric(cdx["exception_type"], errors="coerce")
            add = cdx[(cdx["date"]==dnum)&(cdx["exception_type"]==1)]["service_id"].astype(str).tolist()
            rem = cdx[(cdx["date"]==dnum)&(cdx["exception_type"]==2)]["service_id"].astype(str).tolist()
            if gtfs.calendar is None or gtfs.calendar.empty: active=set()
            active.update(add); active.difference_update(rem)
    return active if active else None

def filter_trips_by_service(gtfs: GTFSData, service_ids: Optional[set]) -> pd.DataFrame:
    if service_ids is None: return gtfs.trips.copy()
    return gtfs.trips[gtfs.trips["service_id"].astype(str).isin({str(s) for s in service_ids})].copy()

@st.cache_data(show_spinner=False)
def build_time_edges_for_date_cached(stop_times: pd.DataFrame, trips_f: pd.DataFrame, routes: pd.DataFrame, sig_key: int) -> pd.DataFrame:
    """Cached edges builder keyed by GTFS + service selection signature."""
    st_df = stop_times.merge(trips_f[["trip_id","route_id","service_id"]], on="trip_id", how="inner")
    st_df = st_df.merge(routes[["route_id","route_type"]], on="route_id", how="left")
    st_df = st_df.sort_values(["trip_id","stop_sequence"])
    st_df["to_stop"] = st_df.groupby("trip_id")["stop_id"].shift(-1)

    # Correct times: depart current stop, arrive at next stop
    st_df["arr_next"] = st_df.groupby("trip_id")["arr_s"].shift(-1)   # arrival at NEXT stop
    st_df["dep_this"] = st_df["dep_s"]                                 # departure at CURRENT stop

    segs = st_df.dropna(subset=["to_stop","arr_next","dep_this"]).copy()
    segs["travel_s"] = (pd.to_numeric(segs["arr_next"]) - pd.to_numeric(segs["dep_this"]))
    segs = segs[segs["travel_s"] > 0]

    edges = segs[["stop_id","to_stop","dep_this","arr_next","trip_id","route_id","route_type","travel_s"]].rename(
        columns={"stop_id":"from_stop","dep_this":"dep_s","arr_next":"arr_s"}
    ).reset_index(drop=True)

    edges["dep_s"] = pd.to_numeric(edges["dep_s"], errors="coerce").fillna(0).astype(int)
    edges["arr_s"] = pd.to_numeric(edges["arr_s"], errors="coerce").fillna(0).astype(int)
    edges["travel_s"] = pd.to_numeric(edges["travel_s"], errors="coerce").fillna(0).astype(int)
    edges["route_type"]=pd.to_numeric(edges["route_type"], errors="coerce").astype("Int64")
    return edges

def build_time_edges_for_date(gtfs: GTFSData, service_trips: pd.DataFrame) -> pd.DataFrame:
    sig_key = gtfs.gtfs_sig ^ df_sig(service_trips, ["trip_id","route_id","service_id"])
    return build_time_edges_for_date_cached(gtfs.stop_times, service_trips, gtfs.routes, sig_key)

# ---------- planning ----------
@dataclass
class Leg:
    dep_s: int; arr_s: int; from_stop: str; to_stop: str; trip_id: str; route_id: str; route_type: Optional[int]

@dataclass
class PlanResult:
    legs: List[Leg]
    unique_stops: int
    unique_edges: int
    total_wait_s: int
    total_travel_s: int
    finished_s: int
    step_log: Optional[pd.DataFrame] = None
    meta: Optional[dict] = None
    seen_stop_ids: Optional[set] = None
    seen_edge_keys: Optional[set] = None
    search_details: Optional[List[Dict]] = None


# --- Live progress reporting -----------------------------------------------
from dataclasses import dataclass
import threading
import time

@dataclass
class RunProgress:
    job: int
    start_stop: str
    start_time_s: int
    iter_or_level: int      # greedy: step count; beam: level count
    best_elapsed_seen: int     # best finished_s - start_time_s seen so far
    stops_seen: int
    edges_seen: int
    targ_stops: int
    targ_edges: int
    full_hit: bool          # met coverage target?
    wall_s: float           # wall-clock seconds for this job

class LiveMonitor:
    """Thread-safe scoreboard for streaming progress from many threads."""
    def __init__(self, total_jobs: int):
        self.lock = threading.Lock()
        self.rows = {}    # job_id -> dict (display row)
        self.total = total_jobs

    def update(self, rp: RunProgress):
        """Thread-safe update of a single job's progress."""

        with self.lock:
            self.rows[rp.job] = {
                "job": rp.job,
                "start": f"{rp.start_stop}@{seconds_to_hhmm(rp.start_time_s)}",
                "iter/level": rp.iter_or_level,

                # RAW fields (used by the metric strip aggregator)
                "best_elapsed_seen": rp.best_elapsed_seen,
                "stops_seen": rp.stops_seen,
                "edges_seen": rp.edges_seen,
                "targ_stops": rp.targ_stops,
                "targ_edges": rp.targ_edges,
                "full_bool": bool(rp.full_hit),


                # DISPLAY fields (what the scoreboard table shows)
                "coverage": f"{rp.stops_seen}/{rp.targ_stops} stops | {rp.edges_seen}/{rp.targ_edges} edges",
                "full": "✅" if rp.full_hit else "—",
                "wall_s": rp.wall_s,
            }


    def table(self):
        """Thread-safe read-only snapshot of the scoreboard as a DataFrame."""

        import pandas as pd
        if not self.rows:
            return pd.DataFrame(columns=["job","start","iter/level","best_elapsed_seen","coverage","full","wall"])

        with self.lock:
            df = pd.DataFrame(list(self.rows.values()))
        df["best_elapsed_seen"] = df["best_elapsed_seen"].apply(fmt_hms)
        
        df["wall"] = df["wall_s"].map(lambda x: f"{x:,.1f}s")
        # Sort: full-hit first, then lowest elapsed, then iter
        df = df.sort_values(by=["full_bool","best_elapsed_seen","iter/level"], ascending=[False, True, True])
        return df[["job","start","iter/level","best_elapsed_seen","coverage","full","wall"]]
@st.cache_resource(show_spinner=False)
def build_departure_index(edges_df: pd.DataFrame, sig: int):
    return NextDepartureIndex(edges_df)

class NextDepartureIndex:
    """Fast per-stop index with cached NumPy dep arrays for searchsorted."""
    def __init__(self, edges_df: pd.DataFrame):
        self.by_stop: Dict[str, pd.DataFrame] = {}
        self.dep_arrays: Dict[str, np.ndarray] = {}
        # Keep only needed columns and ensure int dtype for fast arithmetic
        need_cols = ["from_stop","to_stop","dep_s","arr_s","trip_id","route_id","route_type","travel_s"]
        df = edges_df[need_cols].copy()
        df["dep_s"] = pd.to_numeric(df["dep_s"], errors="coerce").fillna(0).astype(np.int32)
        df["arr_s"] = pd.to_numeric(df["arr_s"], errors="coerce").fillna(0).astype(np.int32)
        df["travel_s"] = pd.to_numeric(df["travel_s"], errors="coerce").fillna(0).astype(np.int32)

        for sid, g in df.groupby("from_stop", sort=False):
            g2 = g.sort_values("dep_s").reset_index(drop=True)
            self.by_stop[sid] = g2
            self.dep_arrays[sid] = g2["dep_s"].to_numpy(np.int32)

    def candidates_from(self, stop_id: str, t0: int, k: int = 50) -> pd.DataFrame:
        g = self.by_stop.get(stop_id)
        if g is None or g.empty:
            return pd.DataFrame(columns=list(next(iter(self.by_stop.values())).columns) if self.by_stop else [])
        arr = self.dep_arrays[stop_id]
        idx = int(np.searchsorted(arr, np.int32(t0), side="left"))
        return g.iloc[idx: idx + k].copy()

def edge_key(u: str, v: str, undirected: bool) -> Tuple[str,str]:
    return tuple(sorted((u,v))) if undirected else (u,v)

def _coverage_hit(stops_seen: int, edges_seen: int, targ_stops: Optional[int], targ_edges: Optional[int]) -> bool:
    ok_stops = (targ_stops is None) or (stops_seen >= targ_stops)
    ok_edges = (targ_edges is None) or (edges_seen >= targ_edges)
    return ok_stops and ok_edges

def required_sets_from_edges(edges_df: pd.DataFrame, undirected: bool) -> Tuple[set, set]:
    """Compute the exact required coverage sets from the filtered network."""
    req_stops = set(edges_df["from_stop"]).union(set(edges_df["to_stop"]))
    if undirected:
        req_edges = {tuple(sorted(x)) for x in edges_df[["from_stop","to_stop"]].itertuples(index=False, name=None)}
    else:
        req_edges = {tuple(x) for x in edges_df[["from_stop","to_stop"]].itertuples(index=False, name=None)}
    return req_stops, req_edges

# ========= candidate ordering (fast) =========
def _prefer_new_sort_frame(
    cands: pd.DataFrame,
    cur_stop: str,
    seen_stops: set,
    seen_edges: set,
    undirected: bool,
    inc_elapsed: np.ndarray,
    prefer_new: bool,
    keep_internal_cols: bool = False,
) -> pd.DataFrame:
    """Primary key: inc_elapsed (wait+ride); soft prefer-new; then dep_s, travel_s. Vectorized for speed."""
    to_arr = cands["to_stop"].to_numpy()
    new_stop = ~np.isin(to_arr, list(seen_stops))
    if undirected:
        ek = [tuple(sorted((cur_stop, v))) for v in to_arr.tolist()]
    else:
        ek = [(cur_stop, v) for v in to_arr.tolist()]
    new_edge = ~pd.Series(ek).isin(seen_edges).to_numpy()
    new_any = np.logical_or(new_stop, new_edge)
    pen = (~new_any).astype(np.int16) if prefer_new else np.zeros(len(cands), dtype=np.int16)

    out = cands.copy()
    out["inc_elapsed"] = inc_elapsed
    # Augmented scoring with branch-aware lookahead & transfer discouragement
    try:
        use_la = bool(st.session_state.get("use_branch_lookahead", True))
        la_depth = int(st.session_state.get("lookahead_depth", 3))
        w_return = float(st.session_state.get("w_return", 1.0))
        w_coverage = float(st.session_state.get("w_coverage", 1.0))
        discour_tx = bool(st.session_state.get("discourage_transfers", True))
        per_tx_pen = int(st.session_state.get("transfer_penalty_s", 120))
    except Exception:
        use_la, la_depth, w_return, w_coverage, discour_tx, per_tx_pen = True, 3, 1.0, 1.0, True, 120
    if use_la and len(cands) <= 256:
        try:
            sig_local = edges_signature(edges_df)
            mean_edge_s = _mean_edge_travel_time(sig_local, edges_df)
            prev_route = prev_route_id
            ret_costs, cov_gains, tx_pens = [], [], []
            for _i, _row in cands.iterrows():
                rc, cg, tp = _candidate_return_and_coverage(
                    cur_stop, prev_route, _row["to_stop"], _row["route_id"], seen_stops,
                    CORRIDOR_OF_STOP, CORRIDOR_NODES, _ADJ_UNDIR, mean_edge_s, la_depth,
                    discour_tx, per_tx_pen
                )
                ret_costs.append(rc); cov_gains.append(cg); tx_pens.append(tp)
            out["return_cost_s"] = ret_costs
            out["coverage_gain_s"] = cov_gains
            out["transfer_pen_s"] = tx_pens
            # Final score = base elapsed + alpha * return_cost + transfer_penalty - gamma * coverage_gain
            out["score_aug"] = (
                out["inc_elapsed"].astype(float)
                + (w_return * out["return_cost_s"].astype(float))
                + out["transfer_pen_s"].astype(float)
                - (w_coverage * out["coverage_gain_s"].astype(float))
            )
            out["inc_elapsed"] = out["score_aug"]
        except Exception as _e_aug:
            pass

    out["_pen"] = pen
    out = out.sort_values(["inc_elapsed", "_pen", "dep_s", "travel_s"], ascending=[True, True, True, True])

    # Conditionally drop the internal '_pen' column
    return out if keep_internal_cols else out.drop(columns=["_pen"])

# ---------- Greedy ----------
def plan_elapsed_first_greedy(
    edges_df: pd.DataFrame, start_stop: str, start_time_s: int, horizon_s: int,
    undirected_edges: bool, k_candidates: int, n_starts: int, explore_eps: float,
    prefer_new_when_available: bool, ignore_deadline: bool,
    target_stops: Optional[int], target_edges: Optional[int],
    zero_wait_same_route: bool, strict_elapsed_first: bool,
    accelerator: str,
    idx: Optional[NextDepartureIndex]=None, log_steps: bool=False,
    *,
    progress_cb: Optional[callable] = None,
    job_id: Optional[int] = None
) -> PlanResult:
    """Greedy planner: repeatedly picks the trip leg that arrives soonest."""
    idx = idx or NextDepartureIndex(edges_df)

    t_wall0 = perf_counter()
    best: Optional[PlanResult] = None
    best_key: Optional[Tuple[int,int,int]] = None

    def run_one(jitter: int) -> PlanResult:
        t0 = start_time_s + jitter
        t_wall0 = perf_counter()
        deadline = t0 + horizon_s
        cur_time = t0; cur_stop = start_stop; prev_route_id = None
        legs: List[Leg] = []; seen_stops={start_stop}; seen_edges=set()
        total_wait=0; total_travel=0; last_arrival=t0
        rows=[]; steps=0

        while steps < 20000:
            steps += 1
            cands = idx.candidates_from(cur_stop, cur_time, k=k_candidates)
            if cands.empty: break

            # GPU-aware incremental elapsed computation
            dep_arr = cands["dep_s"].to_numpy(np.int32)
            arr_arr = cands["arr_s"].to_numpy(np.int32)
            inc_elapsed = _inc_elapsed_vector(
                dep_arr, arr_arr, cur_time, prev_route_id, cands["route_id"],
                zero_wait_same_route, accelerator
            )

            cands = _prefer_new_sort_frame(cands, cur_stop, seen_stops, seen_edges, undirected_edges, inc_elapsed, prefer_new_when_available)

            choice = cands.iloc[0] if strict_elapsed_first else (
                cands.head(min(3, len(cands))).sample(1).iloc[0] if (np.random.rand()<explore_eps and len(cands)>1) else cands.iloc[0]
            )

            if (not ignore_deadline) and int(choice["arr_s"]) > deadline: break

            l = Leg(int(choice["dep_s"]), int(choice["arr_s"]), str(choice["from_stop"]), str(choice["to_stop"]),
                    str(choice["trip_id"]), str(choice["route_id"]),
                    int(choice["route_type"]) if pd.notna(choice["route_type"]) else None)
            legs.append(l)
            w = max(0, l.dep_s - cur_time); t = max(0, l.arr_s - l.dep_s)
            total_wait += w; total_travel += t; last_arrival = l.arr_s

            if log_steps:
                rows.append({
                    "step": steps, "from": l.from_stop, "to": l.to_stop, "dep": seconds_to_hhmm(l.dep_s),
                    "arr": seconds_to_hhmm(l.arr_s), "inc_elapsed_s": int(l.arr_s - cur_time),
                    "wait_s": int(w), "ride_s": int(t),
                    "trip_id": l.trip_id, "route_id": l.route_id
                })

            seen_stops.add(l.to_stop); seen_edges.add(edge_key(l.from_stop, l.to_stop, undirected_edges))
            if progress_cb:
                try:
                    progress_cb(RunProgress(
                        job=(job_id if job_id is not None else -1),
                        start_stop=start_stop,
                        start_time_s=t0,
                        iter_or_level=steps, # steps count
                        best_elapsed_seen=max(0, int(l.arr_s - t0)),
                        stops_seen=len(seen_stops), edges_seen=len(seen_edges), # Corrected variable names for local scope use
                        targ_stops=int(target_stops or 0), targ_edges=int(target_edges or 0),
                        full_hit=_coverage_hit(len(seen_stops), len(seen_edges), target_stops, target_edges),
                        wall_s=float(perf_counter() - t_wall0)
                    ))
                except Exception as e: # Catch errors
                    pass
            if _coverage_hit(len(seen_stops), len(seen_edges), target_stops, target_edges): break
            cur_time = l.arr_s; cur_stop = l.to_stop; prev_route_id = l.route_id # Advance state

        step_log_df = pd.DataFrame(rows) if log_steps else None
        return PlanResult(
            legs, len(seen_stops), len(seen_edges),
            total_wait, total_travel, last_arrival,
            step_log_df, None, set(seen_stops), set(seen_edges)
        )

    # Fewer restarts for speed; jitter window smaller
    jitters = np.random.randint(-300, 300, size=max(1, n_starts))
    for j in jitters: # Restart loop
        res = run_one(int(j))
        elapsed_from_start = max(0, res.finished_s - (start_time_s + int(j)))
        cov_sum = res.unique_stops + res.unique_edges
        need_full = (target_stops is not None) or (target_edges is not None)
        full_ok = _coverage_hit(res.unique_stops, res.unique_edges, target_stops, target_edges)
        if need_full and (not full_ok):
            continue
        key = (elapsed_from_start, -cov_sum, res.total_wait_s)
        if best is None or key < best_key:
            best, best_key = res, key

    if best is not None:
        return best
    else:
        return PlanResult([], 1, 0, 0, 0, start_time_s, None, {"restarts": len(jitters)}, set(), set())

# ========= Robust Beam Backtracking (coverage-guided & adaptive) =========
@dataclass
class BeamState:
    cur_stop: str
    cur_time: int
    prev_route_id: Optional[str]
    legs: List[Leg]
    seen_stops: set
    seen_edges: set
    wait_s: int
    ride_s: int
    depth: int
    last_k: Tuple[str, ...]  # short history

def plan_beam_elapsed_first(
    edges_df: pd.DataFrame,
    start_stop: str,
    start_time_s: int,
    horizon_s: int,
    undirected_edges: bool,
    k_candidates: int,
    beam_width: int,
    per_parent: int,
    max_depth: int,
    prefer_new_when_available: bool,
    ignore_deadline: bool,
    target_stops: Optional[int],
    target_edges: Optional[int],
    zero_wait_same_route: bool,
    strict_elapsed_first: bool,
    accelerator: str,
    idx: Optional[NextDepartureIndex] = None,
    log_steps: bool = False,
    *,
    aggressive: bool = True,
    dom_slack_s: int = 0,
    widen_factor: float = 2.0,
    stagnation_patience: int = 12,
    lookahead_k: int = 12,
    stop_on_first_full: bool = True,
    progress_cb: Optional[callable] = None,
    job_id: Optional[int] = None
) -> PlanResult:
    """
    Coverage-guided, adaptive beam search with strict incremental elapsed-time ordering.
    It can automatically widen the beam and search deeper if coverage stagnates.
    """
    idx = idx or NextDepartureIndex(edges_df)

    all_stop_target = len(set(edges_df["from_stop"]).union(set(edges_df["to_stop"])))
    if undirected_edges:
        all_edge_target = len({tuple(sorted(x)) for x in edges_df[["from_stop","to_stop"]].itertuples(index=False, name=None)})
    else:
        all_edge_target = len({tuple(x) for x in edges_df[["from_stop","to_stop"]].itertuples(index=False, name=None)})

    LAST_K = 4
    TIME_SLACK = max(0, int(dom_slack_s))
    BASE_MIX = 0.6

    def cov_score(stops_count: int, edges_count: int) -> int:
        return stops_count + edges_count

    start = BeamState(
        cur_stop=start_stop, cur_time=start_time_s, prev_route_id=None,
        legs=[], seen_stops={start_stop}, seen_edges=set(),
        wait_s=0, ride_s=0, depth=0, last_k=(start_stop,)
    )
    beams = [start]

    best_seen_time: Dict[Tuple[str, int, int, Tuple[str, ...]], int] = {
        (start_stop, 1, 0, start.last_k): start_time_s
    }

    def beam_sort_key(s: BeamState) -> Tuple[int, int, int, int]:
        elapsed = s.cur_time - start_time_s
        return (-cov_score(len(s.seen_stops), len(s.seen_edges)), elapsed, s.wait_s, s.depth)

    expansions = pruned = levels = 0
    prev_best_cov = 1
    stagnation = 0

    bw_cur = int(beam_width)
    pp_cur = int(per_parent)
    depth_cur = int(max_depth)

    best: Optional[PlanResult] = None

    def maybe_update_best(s: BeamState):
        nonlocal best
        if _coverage_hit(len(s.seen_stops), len(s.seen_edges), target_stops, target_edges):
            pr = PlanResult(
                s.legs, len(s.seen_stops), len(s.seen_edges),
                s.wait_s, s.ride_s, s.cur_time,
                None, None, set(s.seen_stops), set(s.seen_edges)
            )
            if (best is None) or ((pr.finished_s - start_time_s) < (best.finished_s - start_time_s)):
                best = pr

    def local_potential(next_stop: str, arr_s: int, seen_stops: set, seen_edges: set) -> int:
        if lookahead_k <= 0: return 0
        g = idx.by_stop.get(next_stop)
        if g is None or g.empty: return 0
        arr = idx.dep_arrays.get(next_stop, g["dep_s"].to_numpy(np.int32))
        j = int(np.searchsorted(arr, np.int32(arr_s), side="left"))
        g2 = g.iloc[j:j+lookahead_k]
        new_s = (~g2["to_stop"].isin(seen_stops)).sum()
        if undirected_edges:
            ek = [tuple(sorted((next_stop, v))) for v in g2["to_stop"].tolist()]
        else:
            ek = [(next_stop, v) for v in g2["to_stop"].tolist()]
        new_e = (~pd.Series(ek).isin(seen_edges)).sum()
        return int(new_s + new_e)

    while beams and expansions < 800_000 and levels < 3000:
        levels += 1
        if progress_cb:
            try:
                if best is not None and best.legs: # Provide live updates
                    best_elapsed_seen = max(0, int(best.finished_s - start_time_s))
                    s_seen = int(best.unique_stops); e_seen = int(best.unique_edges)
                    full = _coverage_hit(best.unique_stops, best.unique_edges, target_stops, target_edges) # Track
                else:
                    b0 = beams[0] if beams else start
                    best_elapsed_seen = max(0, int(b0.cur_time - start_time_s))
                    s_seen = len(b0.seen_stops); e_seen = len(b0.seen_edges)
                    full = _coverage_hit(s_seen, e_seen, target_stops, target_edges)
                progress_cb(RunProgress(
                    job=(job_id if job_id is not None else -1),
                    start_stop=start_stop,
                    start_time_s=start_time_s,
                        iter_or_level=levels, # levels count
                    best_elapsed_seen=best_elapsed_seen,
                    stops_seen=s_seen, edges_seen=e_seen,
                    targ_stops=int(target_stops or 0), targ_edges=int(target_edges or 0),
                    full_hit=full,
                    wall_s=float(perf_counter() - t_wall0)
                ))
            except Exception as e:
                pass
        children: List[BeamState] = []

        for b in beams:
            cands_full = idx.candidates_from(b.cur_stop, b.cur_time, k=k_candidates)
            if cands_full.empty:
                continue

            dep_arr = cands_full["dep_s"].to_numpy(np.int32)
            arr_arr = cands_full["arr_s"].to_numpy(np.int32)
            inc_elapsed = _inc_elapsed_vector(
                dep_arr, arr_arr, b.cur_time, b.prev_route_id, cands_full["route_id"],
                zero_wait_same_route, accelerator
            )

            base_sorted = _prefer_new_sort_frame(
                cands_full, b.cur_stop, b.seen_stops, b.seen_edges, undirected_edges, inc_elapsed, prefer_new_when_available
            )

            if lookahead_k > 0:
                pots = []
                for _, row in base_sorted.iterrows():
                    pots.append(local_potential(str(row["to_stop"]), int(row["arr_s"]), b.seen_stops, b.seen_edges))
                base_sorted = base_sorted.assign(_pot=pots)
            else:
                base_sorted = base_sorted.assign(_pot=0)

            # Strict ordering: inc_elapsed → prefer-new penalty → dep_s → travel_s → potential(desc)
            if prefer_new_when_available:
                            
                new_stop_mask = ~base_sorted["to_stop"].isin(b.seen_stops)
                ek = [tuple(sorted((b.cur_stop, v))) if undirected_edges else (b.cur_stop, v) for v in base_sorted["to_stop"].tolist()]
                new_edge_mask = ~pd.Series(ek).isin(b.seen_edges).values
                new_any_mask = (new_stop_mask | new_edge_mask)
                base_sorted = base_sorted.assign(_pen=(~new_any_mask).astype(int))
            else:
                base_sorted = base_sorted.assign(_pen=0)

            base_sorted = base_sorted.sort_values(
                ["inc_elapsed", "_pen", "dep_s", "travel_s", "_pot"],
                ascending=[True, True, True, True, False]
            )
            c_take = base_sorted.head(pp_cur)

            for _, row in c_take.iterrows():
                if b.depth + 1 > depth_cur:
                    continue
                arr_s = int(row["arr_s"])
                if (not ignore_deadline) and arr_s > start_time_s + horizon_s:
                    continue

                dep_s = int(row["dep_s"])
                l = Leg(
                    dep_s=dep_s, arr_s=arr_s,
                    from_stop=str(row["from_stop"]), to_stop=str(row["to_stop"]),
                    trip_id=str(row["trip_id"]), route_id=str(row["route_id"]),
                    route_type=int(row["route_type"]) if pd.notna(row["route_type"]) else None,
                )

                w = max(0, dep_s - b.cur_time)
                r = max(0, arr_s - dep_s)

                seen_stops = set(b.seen_stops); seen_edges = set(b.seen_edges)
                seen_stops.add(l.to_stop)
                ekey = tuple(sorted((l.from_stop, l.to_stop))) if undirected_edges else (l.from_stop, l.to_stop)
                seen_edges.add(ekey)

                last_k = (b.last_k + (l.to_stop,))[-LAST_K:]
                dom_key = (l.to_stop, len(seen_stops), len(seen_edges), last_k)
                prev_best = best_seen_time.get(dom_key)

                if prev_best is not None and arr_s >= prev_best - TIME_SLACK:
                    pruned += 1
                    continue
                best_seen_time[dom_key] = arr_s

                child = BeamState(
                    cur_stop=l.to_stop, cur_time=arr_s, prev_route_id=l.route_id,
                    legs=b.legs + [l], seen_stops=seen_stops, seen_edges=seen_edges,
                    wait_s=b.wait_s + w, ride_s=b.ride_s + r, depth=b.depth + 1, last_k=last_k,
                )

                maybe_update_best(child)
                children.append(child)
                expansions += 1

                if stop_on_first_full and _coverage_hit(len(seen_stops), len(seen_edges), target_stops, target_edges):
                    beams = []
                    children = [child]
                    break

            if not beams:
                break

        if not children:
            break

        children.sort(key=beam_sort_key)
        beams = children[:bw_cur]

        # Adaptive escalation on stagnation
        level_best_cov = max(cov_score(len(x.seen_stops), len(x.seen_edges)) for x in beams)
        if level_best_cov <= prev_best_cov:
            stagnation += 1
            if aggressive and stagnation >= stagnation_patience:
                # When coverage stalls, widen the beam and search deeper
                bw_cur = min(40, int(math.ceil(bw_cur * min(widen_factor, 1.5))))
                pp_cur = min(10, int(math.ceil(pp_cur * min(widen_factor, 1.5))))
                depth_cur = min(20000, depth_cur + 200)
                stagnation = 0
        else:
            prev_best_cov = level_best_cov
            stagnation = 0

    if best is None:
        if beams:
            beams.sort(key=beam_sort_key)
            b = beams[0]
            best = PlanResult(
                b.legs, len(b.seen_stops), len(b.seen_edges),
                b.wait_s, b.ride_s, b.cur_time,
                None, None, set(b.seen_stops), set(b.seen_edges)
            )
        else:
            best = PlanResult([], 1, 0, 0, 0, start_time_s, None, None, set(), set())

    if log_steps and best and best.legs:
        rows = []
        cur_time = start_time_s
        for i, l in enumerate(best.legs, 1):
            w = max(0, l.dep_s - cur_time)
            r = max(0, l.arr_s - l.dep_s)
            rows.append(
                {
                    "step": i, "from": l.from_stop, "to": l.to_stop,
                    "dep": seconds_to_hhmm(l.dep_s), "arr": seconds_to_hhmm(l.arr_s),
                    "inc_elapsed_s": int(l.arr_s - cur_time),
                    "wait_s": int(w), "ride_s": int(r),
                    "trip_id": l.trip_id, "route_id": l.route_id,
                }
            )
            cur_time = l.arr_s
        best.step_log = pd.DataFrame(rows)
        best.meta = best.meta or {}
        best.meta.update({
            "beam_levels": levels, "beam_expansions": expansions, "beam_pruned": pruned,
            "bw_final": bw_cur, "pp_final": pp_cur, "depth_final": depth_cur,
            "aggressive": aggressive, "dom_slack_s": TIME_SLACK, "widen_factor": widen_factor,
            "stagnation_patience": stagnation_patience, "lookahead_k": lookahead_k,
        })
    return best

def generate_verbose_log_for_plan(
    plan: PlanResult,
    start_time_s: int,
    edges_df: pd.DataFrame,
    idx: NextDepartureIndex,
    params: dict,
) -> List[Dict]:
    """
    Replays a finished plan to generate a rich, step-by-step log of the
    decisions made, including all considered candidates and their scores.
    """
    if not plan.legs:
        return []

    details = []
    cur_time = start_time_s
    cur_stop = plan.legs[0].from_stop
    prev_route_id = None
    seen_stops = {cur_stop}
    seen_edges = set()

    # Unpack relevant parameters used in the search
    k_candidates = params.get("k_cands", 50)
    zero_wait_same_route = params.get("zero_wait_same_route", False)
    accelerator = params.get("accelerator", "cpu")
    undirected = params.get("undirected_edges", True)
    prefer_new = params.get("prefer_new", True)

    for i, leg in enumerate(plan.legs):
        step_info = {
            "step": i + 1,
            "cur_time": cur_time,
            "cur_stop": cur_stop,
            "seen_stops_count": len(seen_stops),
            "seen_edges_count": len(seen_edges),
        }

        # Get the exact same candidates the planner would have seen
        cands = idx.candidates_from(cur_stop, cur_time, k=k_candidates)

        if not cands.empty:
            # Re-compute the sorting metrics for these candidates
            dep_arr = cands["dep_s"].to_numpy(np.int32)
            arr_arr = cands["arr_s"].to_numpy(np.int32)
            inc_elapsed = _inc_elapsed_vector(
                dep_arr, arr_arr, cur_time, prev_route_id, cands["route_id"],
                zero_wait_same_route, accelerator
            )

            # Re-run the sorting logic, but KEEP the internal scoring columns
            sorted_cands = _prefer_new_sort_frame(
                cands, cur_stop, seen_stops, seen_edges, undirected, inc_elapsed, prefer_new,
                keep_internal_cols=True
            )

            # Mark the one that was actually chosen in the plan
            is_chosen_mask = (
                (sorted_cands["to_stop"] == leg.to_stop) &
                (sorted_cands["dep_s"] == leg.dep_s) &
                (sorted_cands["arr_s"] == leg.arr_s) &
                (sorted_cands["trip_id"] == leg.trip_id)
            )
            sorted_cands["is_chosen"] = is_chosen_mask

            # Clean up for display
            display_cols = [
                "is_chosen", "inc_elapsed", "from_stop", "to_stop", "dep_s", "arr_s",
                "route_id", "trip_id", "travel_s", "_pen"
            ]

            step_info["candidates"] = sorted_cands[display_cols].to_dict("records")
        else:
            step_info["candidates"] = []

        details.append(step_info)

        # Update state for the next iteration
        cur_time = leg.arr_s
        cur_stop = leg.to_stop
        prev_route_id = leg.route_id
        seen_stops.add(cur_stop)
        seen_edges.add(edge_key(leg.from_stop, leg.to_stop, undirected))

    return details


def plan_elapsed_components(plan: PlanResult, start_time_s: int) -> Tuple[int,int,int]:
    if not plan.legs: return 0,0,0
    elapsed = max(0, plan.finished_s - start_time_s)
    inv = plan.total_travel_s
    wait = elapsed - inv
    return elapsed, inv, wait

# ---------- grouping (improved + cached) ----------
def _strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s or "").encode("ascii","ignore").decode("ascii")

_STOP_NAME_JUNK = re.compile(r"\b(station|stn|platform|bay|stop|inbound|outbound|wb|eb|nb|sb)\b", re.I)

def normalize_stop_name(name: str) -> str:
    s=_strip_accents(name).lower()
    s=re.sub(r"[/\-–()&,\.]"," ",s)
    s=re.sub(r"\s+\d{1,3}$"," ",s)
    s=_STOP_NAME_JUNK.sub(" ",s)
    s=re.sub(r"\s+"," ",s).strip()
    return s

_CANON = {
    "st":"street","st.":"street",
    "ave":"avenue","av":"avenue","ave.":"avenue","av.":"avenue",
    "rd":"road","rd.":"road",
    "blvd":"boulevard","blvd.":"boulevard",
    "hwy":"highway","pkwy":"parkway","pkwy.":"parkway",
    "ctr":"center","ctr.":"center","ct":"court","ct.":"court",
    "ln":"lane","ln.":"lane","pl":"place","pl.":"place",
}
_STOPWORDS = {
    "and","at","the","of","to","for","near","opp","opposite","outside","inside","bay","stop",
    "platform","stand","standby"
}
_DIRWORDS = {"nb","sb","eb","wb","ne","nw","se","sw","inbound","outbound"}

def _name_tokens(name: str) -> List[str]:
    base = normalize_stop_name(name)
    toks = base.split()
    out = []
    for t in toks:
        if t in _DIRWORDS or t in _STOPWORDS:
            continue
        out.append(_CANON.get(t, t))
    return out

def _jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

class UnionFind:
    def __init__(self, n:int):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x:int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a:int, b:int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

@st.cache_data(show_spinner=False)
def group_stops_by_proximity_and_name_cached(
    stops_raw_filtered_sig: int,
    stops_frame: pd.DataFrame,
    dist_m: float,
    name_thr: float,
    jaccard_thr: float = 0.5,
    min_shared_tokens: int = 1,
    distance_override_m: float = 25.0,
    grid_expand: int = 1,
):
    """Cached grouping based on input signature + parameters."""
    if stops_frame.empty:
        return pd.DataFrame(columns=["group_id","stop_name","lat","lon","size"]), {}

    df = stops_frame.reset_index(drop=True)[["stop_id","stop_name","stop_lat","stop_lon"]].rename(columns={"stop_lat":"lat","stop_lon":"lon"})
    n = len(df)
    lat0 = float(df["lat"].median())
    deg_lat = dist_m / 111_320.0
    deg_lon = dist_m / (111_320.0 * max(0.1, math.cos(math.radians(lat0))))

    def cell_of(lat,lon):
        return (math.floor(lat/deg_lat) if deg_lat>0 else 0,
                math.floor(lon/deg_lon) if deg_lon>0 else 0)

    cells: Dict[Tuple[int,int], List[int]] = {}
    for i, r in df.iterrows():
        cells.setdefault(cell_of(r["lat"], r["lon"]), []).append(i)

    tok_sets = [set(_name_tokens(s)) for s in df["stop_name"].tolist()]
    norm_names = [normalize_stop_name(s) for s in df["stop_name"].tolist()]

    uf = UnionFind(n)
    for i, r in df.iterrows():
        ci, cj = cell_of(r["lat"], r["lon"])
        for di in range(-grid_expand, grid_expand+1):
            for dj in range(-grid_expand, grid_expand+1):
                neigh = cells.get((ci+di, cj+dj), [])
                for j in neigh:
                    if j <= i:
                        continue
                    s2 = df.iloc[j]
                    d_m = haversine_m(r["lat"], r["lon"], s2["lat"], s2["lon"])
                    if d_m > dist_m and d_m > distance_override_m:
                        continue

                    seq_sim = difflib.SequenceMatcher(None, norm_names[i], norm_names[j]).ratio()
                    jsim = _jaccard(tok_sets[i], tok_sets[j])
                    shared = len(tok_sets[i] & tok_sets[j])

                    name_ok = (seq_sim >= name_thr) or (jsim >= jaccard_thr and shared >= min_shared_tokens)
                    override_ok = (d_m <= distance_override_m)

                    if (d_m <= dist_m and name_ok) or override_ok:
                        uf.union(i, j)

    roots = [uf.find(i) for i in range(n)]
    by_root: Dict[int,List[int]] = {}
    for idx, r in enumerate(roots): by_root.setdefault(r, []).append(idx)

    groups=[]; mapping={}
    for _, members in by_root.items():
        sub = df.iloc[members]
        clat = float(sub["lat"].mean()); clon = float(sub["lon"].mean())
        rep_name = sub["stop_name"].mode().iloc[0] if not sub["stop_name"].mode().empty else sub["stop_name"].iloc[0]
        dists = sub.apply(lambda rr: haversine_m(clat,clon,rr["lat"],rr["lon"]), axis=1)
        rep_label = int(dists.idxmin())
        gid = f"grp_{rep_label}"
        groups.append({"group_id":gid,"stop_name":rep_name,"lat":clat,"lon":clon,"size":len(members)})
        for idx in members:
            mapping[df.iloc[idx]["stop_id"]] = gid

    return pd.DataFrame(groups), mapping

def group_stops_by_proximity_and_name(stops: pd.DataFrame, dist_m: float, name_thr: float,
                                      jaccard_thr: float = 0.5, min_shared_tokens: int = 1,
                                      distance_override_m: float = 25.0):
    sig = df_sig(stops, ["stop_id","stop_name","stop_lat","stop_lon"])
    return group_stops_by_proximity_and_name_cached(sig, stops, dist_m, name_thr, jaccard_thr, min_shared_tokens, distance_override_m)

def remap_edges_to_groups(edges_df: pd.DataFrame, mapping: Dict[str,str]) -> pd.DataFrame:
    if not mapping: return edges_df.copy()
    e=edges_df.copy()
    e["from_stop"]=e["from_stop"].map(lambda s: mapping.get(s,s))
    e["to_stop"]=e["to_stop"].map(lambda s: mapping.get(s,s))
    e=e[e["from_stop"]!=e["to_stop"]].reset_index(drop=True)
    return e

# ---- Rail/busiest label selection for grouped stops ----
@st.cache_data(show_spinner=False)
def compute_stop_service_metadata(gtfs: GTFSData, trips_f: pd.DataFrame) -> Tuple[Dict[str,int], Dict[str,bool]]:
    """Return (visits_per_stop, is_rail_stop) for the active service selection."""
    st_served = gtfs.stop_times.merge(trips_f[["trip_id","route_id"]], on="trip_id", how="inner")
    visits = st_served.groupby("stop_id").size().to_dict()
    rtypes = gtfs.routes.set_index("route_id")["route_type"].to_dict()
    st_served["route_type"] = st_served["route_id"].map(rtypes)
    def _is_rail(s: pd.Series) -> bool:
        vals = pd.to_numeric(s, errors="coerce").dropna().astype(int)
        return 2 in set(vals)  # GTFS route_type 2 = Rail
    rail = st_served.groupby("stop_id")["route_type"].apply(_is_rail).to_dict()
    return visits, rail

def relabel_groups_by_rule(
    groups_df: pd.DataFrame,
    mapping: Dict[str, str],
    gtfs: GTFSData,
    trips_f: pd.DataFrame,
) -> pd.DataFrame:
    """Rename group display names: prefer a rail stop's name; otherwise the member with most visits."""
    visits, rail = compute_stop_service_metadata(gtfs, trips_f)
    # invert mapping: group_id -> [member stop_ids]
    inv: Dict[str, List[str]] = {}
    for sid, gid in mapping.items():
        inv.setdefault(gid, []).append(sid)

    stops_lookup = gtfs.stops.set_index("stop_id")["stop_name"].to_dict()

    def pick_name(gid: str) -> str:
        members = inv.get(gid, [])
        if not members:
            return groups_df.loc[groups_df["group_id"] == gid, "stop_name"].iloc[0]
        rail_members = [s for s in members if rail.get(s, False)]
        if rail_members:
            chosen = rail_members[0]
        else:
            chosen = max(members, key=lambda s: visits.get(s, 0))
        return stops_lookup.get(chosen, groups_df.loc[groups_df["group_id"] == gid, "stop_name"].iloc[0])

    out = groups_df.copy()
    out["stop_name"] = out["group_id"].map(pick_name)
    return out

# ---------- Smart start recommendations (RIGOROUS) ----------
@st.cache_data(show_spinner=False)
def _compute_undirected_degree(edges_df_sig: int, edges_df: pd.DataFrame) -> Dict[str,int]:
    deg = collections.Counter()
    for a,b in edges_df[["from_stop","to_stop"]].itertuples(index=False, name=None):
        if a == b:
            continue
        deg[a]+=1; deg[b]+=1
    return dict(deg)

@st.cache_data(show_spinner=False)
def _compute_directed_degrees(edges_df_sig: int, edges_df: pd.DataFrame) -> Tuple[Dict[str,int], Dict[str,int]]:
    outdeg = collections.Counter()
    indeg = collections.Counter()
    for a,b in edges_df[["from_stop","to_stop"]].itertuples(index=False, name=None):
        if a == b:
            continue
        outdeg[a]+=1; indeg[b]+=1
    keys = set(outdeg.keys()) | set(indeg.keys())
    return ({k: outdeg.get(k,0) for k in keys}, {k: indeg.get(k,0) for k in keys})

@st.cache_data(show_spinner=False)
def _build_adj_undirected(edges_df_sig: int, edges_df: pd.DataFrame) -> Dict[str, set]:
    adj: Dict[str,set] = {}
    for a,b in edges_df[["from_stop","to_stop"]].itertuples(index=False, name=None):
        if a == b:
            continue
        adj.setdefault(a,set()).add(b)
        adj.setdefault(b,set()).add(a)
    return adj

@st.cache_data(show_spinner=False)
def _components_undirected(edges_df_sig: int, edges_df: pd.DataFrame) -> Dict[str,int]:
    """Map stop_id -> component_id for the undirected underlying graph."""
    adj = _build_adj_undirected(edges_df_sig, edges_df)
    comp = {}
    cid = 0
    from collections import deque
    for s in adj.keys():
        if s in comp:
            continue
        cid += 1
        dq = deque([s]); comp[s]=cid
        while dq:
            u=dq.popleft()
            for v in adj.get(u,()):
                if v not in comp:
                    comp[v]=cid; dq.append(v)
    return comp
@st.cache_data(show_spinner=False)
def _compute_corridors(edges_df_sig: int, edges_df: pd.DataFrame) -> Tuple[Dict[str,int], Dict[int,set]]:
    deg = _compute_undirected_degree(edges_df_sig, edges_df)
    adj = _build_adj_undirected(edges_df_sig, edges_df)
    chain_nodes = {n for n,d in deg.items() if d == 2}
    visited = set()
    corridor_of_stop: Dict[str,int] = {}
    corridor_nodes: Dict[int,set] = {}
    cid = 0
    for n in list(chain_nodes):
        if n in visited:
            continue
        comp = set()
        stack = [n]
        while stack:
            u = stack.pop()
            if u in visited or u not in chain_nodes:
                continue
            visited.add(u)
            comp.add(u)
            for v in adj.get(u, ()):
                if v in chain_nodes and v not in visited:
                    stack.append(v)
        if comp:
            corridor_nodes[cid] = comp
            for u in comp:
                corridor_of_stop[u] = cid
            cid += 1
    return corridor_of_stop, corridor_nodes


# ==== Branch-aware lookahead helpers ====
@st.cache_data(show_spinner=False)
def _mean_edge_travel_time(edges_df_sig: int, edges_df: pd.DataFrame) -> float:
    try:
        d = (edges_df["arr_s"] - edges_df["dep_s"]).clip(lower=0)
        return float(d.mean()) if len(d) else 60.0
    except Exception:
        return 60.0

def _min_hops_to_set(start: str, targets: set, adj: Dict[str,set], max_hops: int = 64) -> int:
    """
    BFS in the undirected stop graph to the nearest node in targets.
    Returns hop count (edges). If unreachable, returns max_hops+1.
    """
    if not targets or start in targets:
        return 0 if start in targets else max_hops + 1
    q = [(start, 0)]
    seen = {start}
    while q:
        u, h = q.pop(0)
        if h >= max_hops: 
            return max_hops + 1
        for v in adj.get(u, ()):
            if v in seen:
                continue
            if v in targets:
                return h + 1
            seen.add(v); q.append((v, h + 1))
    return max_hops + 1

def _corridor_walk_gain(from_stop: str, seen_stops: set, corridor_of_stop: Dict[str,int], corridor_nodes: Dict[int,set],
                        adj: Dict[str,set], max_steps: int = 3) -> int:
    """
    Starting at 'from_stop', count how many NEW corridor nodes we could cover by continuing
    along the same corridor greedily for up to max_steps. This estimates the "finish branch" payoff.
    """
    cid = corridor_of_stop.get(str(from_stop))
    if cid is None:
        return 0
    C = corridor_nodes.get(cid, set())
    # Only corridor neighbors
    step = 0
    covered = 0
    cur = str(from_stop)
    prev = None
    visited_local = set()
    while step < max_steps:
        # choose a corridor neighbor that's not yet globally seen and not the previous node
        nxt = None
        for v in adj.get(cur, ()):
            if v == prev: 
                continue
            if v in C and v not in seen_stops and v not in visited_local:
                nxt = v; break
        if nxt is None:
            break
        visited_local.add(nxt)
        covered += 1
        prev, cur = cur, nxt
        step += 1
    return covered

def _candidate_return_and_coverage(cur_stop: str, prev_route_id: str,
                                   cand_to_stop: str, cand_route_id: str,
                                   seen_stops: set,
                                   corridor_of_stop: Dict[str,int], corridor_nodes: Dict[int,set],
                                   adj: Dict[str,set],
                                   mean_edge_s: float,
                                   lookahead_depth: int,
                                   discourage_transfers: bool,
                                   per_transfer_penalty_s: int) -> tuple[float, float, float]:
    """
    For a single candidate move, estimate:
     - expected "return-to-branch" cost (seconds) if leaving a corridor with remaining unvisited nodes
     - coverage_gain_s: seconds of coverage you'd likely secure by continuing along the corridor if you stay
     - transfer_penalty_s: seconds penalty if we switch routes (optional)
    """
    return_cost_s = 0.0
    coverage_gain_s = 0.0
    transfer_pen_s = 0.0

    # If discourage_transfers is on, add fixed cost for route switches
    if discourage_transfers and str(cand_route_id) != str(prev_route_id) and prev_route_id is not None:
        transfer_pen_s = float(per_transfer_penalty_s)

    cid = corridor_of_stop.get(str(cur_stop))
    if cid is not None:
        # Any unvisited remaining in this corridor?
        remaining = {s for s in corridor_nodes.get(cid, set()) if s not in seen_stops}
        if remaining:
            # If candidate stays in corridor, estimate coverage gain by walking forward
            if corridor_of_stop.get(str(cand_to_stop)) == cid:
                new_nodes = _corridor_walk_gain(str(cand_to_stop), seen_stops, corridor_of_stop, corridor_nodes, adj, max_steps=int(lookahead_depth))
                coverage_gain_s = float(new_nodes) * float(mean_edge_s)
            else:
                # candidate leaves the corridor early; estimate hops to come back later
                hops = _min_hops_to_set(str(cand_to_stop), remaining, adj, max_hops=64)
                return_cost_s = float(hops) * float(mean_edge_s)

    return return_cost_s, coverage_gain_s, transfer_pen_s
def _branch_leaving_mask(cur_stop: str, to_series: pd.Series, seen_stops: set,
                         corridor_of_stop: Dict[str,int], corridor_nodes: Dict[int,set]) -> "np.ndarray":
    try:
        import numpy as np
    except Exception:
        return [0] * len(to_series)
    cid = corridor_of_stop.get(str(cur_stop), None)
    if cid is None:
        return np.zeros(len(to_series), dtype=np.int16)
    remaining = any((s not in seen_stops) for s in corridor_nodes.get(cid, set()))
    if not remaining:
        return np.zeros(len(to_series), dtype=np.int16)
    to_arr = to_series.astype(str).tolist()
    out = np.fromiter(((corridor_of_stop.get(t, None) != cid) for t in to_arr), dtype=np.int16, count=len(to_arr))
    return out


def _bfs_dists(adj: Dict[str,set], start: str) -> Dict[str,int]:
    from collections import deque
    dq = deque([start]); dist={start:0}
    while dq:
        u=dq.popleft()
        for v in adj.get(u,()):
            if v not in dist:
                dist[v]=dist[u]+1
                dq.append(v)
    return dist

def _bfs_farthest(adj: Dict[str,set], start: str) -> Tuple[str,int]:
    d = _bfs_dists(adj, start)
    if not d:
        return start, 0
    node = max(d, key=lambda k: d[k])
    return node, d[node]

@st.cache_data(show_spinner=False)
def _approx_diameter_pair(edges_df_sig: int, edges_df: pd.DataFrame, prefer_nodes: Optional[Tuple[str,...]]=None) -> Tuple[Optional[str], Optional[str]]:
    """Double-sweep heuristic to find a far pair (approx diameter)."""
    adj = _build_adj_undirected(edges_df_sig, edges_df)
    seeds = list(prefer_nodes) if prefer_nodes else (list(adj.keys())[:1] if adj else [])
    if not seeds:
        return (None, None)
    a, _ = _bfs_farthest(adj, seeds[0])
    b, _ = _bfs_farthest(adj, a)
    return (a, b)

@st.cache_data(show_spinner=False)
def _two_hop_degree(edges_df_sig: int, edges_df: pd.DataFrame) -> Dict[str,int]:
    """Approx centrality = |neighbors ∪ neighbors-of-neighbors|."""
    adj = _build_adj_undirected(edges_df_sig, edges_df)
    th = {}
    for u, nbrs in adj.items():
        two = set(nbrs)
        for v in nbrs:
            two |= adj.get(v, set())
        two.discard(u)
        th[u] = len(two)
    return th

def _availability_counts(edges_df: pd.DataFrame, t_list: List[int], window_s: int = 1800) -> Dict[str,int]:
    """For each stop, count how many times there is some departure within [t, t+window_s] across t_list."""
    if not t_list:
        return {}
    g = edges_df.groupby("from_stop")["dep_s"].apply(lambda s: np.sort(s.astype(int).to_numpy()))
    out={}
    for sid, arr in g.items():
        cnt = 0
        for t in t_list:
            j = int(np.searchsorted(arr, int(t), side="left"))
            if j < len(arr) and int(arr[j]) <= int(t) + int(window_s):
                cnt += 1
        out[sid] = cnt
    return out

def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = float(min(vals)), float(max(vals))
    if hi <= lo:
        return {k: 0.0 for k in scores}
    return {k: (float(v)-lo)/(hi-lo) for k,v in scores.items()}

def _unique_keep_order(seq: List[str]) -> List[str]:
    seen=set(); out=[]
    for x in seq:
        if x is None:
            continue
        if x in seen:
            continue
        seen.add(x); out.append(x)
    return out

def _suggest_end_for_start(adj: Dict[str,set], comp_pair_by_comp: Dict[int, Tuple[str,str]], comp_map: Dict[str,int], start: str) -> Optional[str]:
    """Pick one of the component's diameter endpoints farthest from 'start' as an end suggestion."""
    cid = comp_map.get(start)
    pair = comp_pair_by_comp.get(cid)
    if not pair:
        return None
    a, b = pair
    # no need to compute all distances; quick heuristic: whichever anchor is farther in hops from start
    dists = _bfs_dists(adj, start)
    da = dists.get(a, -1); db = dists.get(b, -1)
    if db > da:
        return b
    else:
        return a

def _score_candidates(
    edges_df: pd.DataFrame,
    startable_ids: set,
    require_all_edges: bool,
    undirected_edges: bool,
    t_list: Optional[List[int]]=None,
    horizon_s: Optional[int]=None,
    max_recs: int = 80,
) -> Tuple[List[str], List[str], str]:
    """
    Rigorous scoring combining:
      • Eulerian constraints (for edges goal),
      • periphery (approx eccentricity via distance to diameter ends),
      • leaf bonus,
      • anti-hub (two-hop degree),
      • departure availability in the user's time window.
    Also spreads picks across connected components.
    """
    sig = edges_signature(edges_df)
    adj = _build_adj_undirected(sig, edges_df)
    comp_map = _components_undirected(sig, edges_df)
    deg = _compute_undirected_degree(sig, edges_df)
    twohop = _two_hop_degree(sig, edges_df)
    # Per-component diameter endpoints a,b (via double sweep)
    comp_nodes: Dict[int, List[str]] = {}
    for n, c in comp_map.items():
        comp_nodes.setdefault(c, []).append(n)
    comp_pair_by_comp: Dict[int, Tuple[str,str]] = {}
    for c, nodes in comp_nodes.items():
        seed = nodes[0]
        a, _ = _bfs_farthest(adj, seed)
        b, _ = _bfs_farthest(adj, a)
        comp_pair_by_comp[c] = (a, b)

    # Availability (timetable-based), window = min(30m, horizon/6), bounded [10m..60m]
    avail = {}
    if t_list:
        win = 1800
        if horizon_s:
            win = max(600, min(3600, int(horizon_s // 6)))
        avail = _availability_counts(edges_df, t_list, window_s=win)

    # Normalize scores
    periphery: Dict[str, float] = {}
    for c, (a, b) in comp_pair_by_comp.items():
        da = _bfs_dists(adj, a)
        db = _bfs_dists(adj, b)
        ecc = {}
        for n in comp_nodes[c]:
            ecc[n] = max(da.get(n, 0), db.get(n, 0))
        # normalize within component to keep fairness
        ecc_norm = _normalize(ecc)
        periphery.update(ecc_norm)

    leaf_bonus = {n: 1.0 if deg.get(n,0)==1 else 0.0 for n in deg}
    anti_hub = _normalize({n: -float(twohop.get(n,0)) for n in deg})  # negative two-hop degree
    avail_norm = _normalize(avail) if avail else {}

    # Composite
    comp_score = {}
    for n in deg.keys():
        comp_score[n] = (
            3.0 * leaf_bonus.get(n,0.0) +
            2.0 * periphery.get(n,0.0) +
            1.0 * anti_hub.get(n,0.0) +
            1.0 * avail_norm.get(n,0.0)
        )

    # Eulerian constraints for edges-goal override
    euler_reason = ""
    euler_starts: List[str] = []
    euler_ends: List[str] = []
    if require_all_edges:
        if undirected_edges:
            odd = [n for n,dv in deg.items() if dv % 2 == 1]
            if len(odd) == 2:
                a,b = odd
                euler_starts = [a,b]
                euler_ends = [b,a]
                euler_reason = "Eulerian (undirected): exactly two odd-degree stations — must start/end there."
            elif len(odd) == 0:
                # prefer distant leaves or diameter ends
                euler_reason = "Eulerian (undirected, all even): circuit exists; prefer peripheral leaf/diameter endpoints."
            else:
                euler_reason = f"Eulerization needed (undirected): {len(odd)} odd-degree stations; prefer odd nodes as endpoints."
                # Prefer odd nodes strongly
                for n in odd:
                    comp_score[n] += 5.0
        else:
            outdeg, indeg = _compute_directed_degrees(sig, edges_df)
            start_cands = [n for n in outdeg if outdeg.get(n,0) == indeg.get(n,0) + 1]
            end_cands   = [n for n in indeg if indeg.get(n,0) == outdeg.get(n,0) + 1]
            if len(start_cands) == 1 and len(end_cands) == 1:
                euler_starts = start_cands
                euler_ends = end_cands
                euler_reason = "Eulerian (directed): start at out=in+1, end at in=out+1."
            else:
                # imbalance heuristic
                score = {n: outdeg.get(n,0) - indeg.get(n,0) for n in set(outdeg)|set(indeg)}
                if score:
                    s = max(score, key=lambda k: score[k]); e = min(score, key=lambda k: score[k])
                    euler_starts = [s]; euler_ends = [e]
                    euler_reason = "Directed (non-Euler): favored nodes with largest positive/negative (out-in)."
                # Bias imbalanced nodes
                for n, v in score.items():
                    comp_score[n] += 3.0 * abs(float(v))

    # Rank by score inside each component, then interleave components to diversify
    per_comp_sorted: Dict[int, List[str]] = {}
    for c, nodes in comp_nodes.items():
        nodes_f = [n for n in nodes if n in startable_ids]
        nodes_f.sort(key=lambda n: comp_score.get(n, -1e9), reverse=True)
        per_comp_sorted[c] = nodes_f

    # Round-robin pick across components
    total_nodes = len(startable_ids)
    cap = max(10, min(120, int(max_recs)))
    picks: List[str] = []
    round_idx = 0
    while len(picks) < cap:
        added_any = False
        for c, nodes in per_comp_sorted.items():
            if round_idx < len(nodes):
                picks.append(nodes[round_idx])
                added_any = True
                if len(picks) >= cap:
                    break
        if not added_any:
            break
        round_idx += 1

    # Always include diameter endpoints as backups
    for c, (a,b) in comp_pair_by_comp.items():
        for x in (a,b):
            if x in startable_ids:
                picks.append(x)

    # Place Eulerian endpoints (if any) right up front
    rec_starts = _unique_keep_order((euler_starts or []) + picks)
    rec_starts = [s for s in rec_starts if s in startable_ids][:cap]

    # Suggested ends: for each start, pick the farther diameter anchor in that component
    ends=[]
    for s in rec_starts[:max(2, min(40, len(rec_starts)))]:
        ends.append(_suggest_end_for_start(adj, comp_pair_by_comp, comp_map, s))
    rec_ends = _unique_keep_order((euler_ends or []) + [e for e in ends if e in startable_ids])

    why = []
    if euler_reason:
        why.append(euler_reason)
    why.append("Periphery favored via approximate eccentricity to component diameter ends; leaf bonus; anti-hub (two-hop degree); timetable availability within your time window.")
    return rec_starts, rec_ends, " ".join(why)

def _recommend_starts_and_ends(
    edges_df: pd.DataFrame,
    startable_ids: set,
    objective_nodes: bool,
    require_all_stops: bool,
    require_all_edges: bool,
    undirected_edges: bool,
    t_list: Optional[List[int]] = None,
    horizon_s: Optional[int] = None,
    max_recs: int = 80,
) -> Tuple[List[str], List[str], str]:
    """
    Wrapper that chooses the right scoring mode and returns (recommended_starts, suggested_ends, rationale).
    For node coverage, we still use the rigorous periphery/availability scoring.
    For edge coverage, Eulerian constraints get priority, then scoring handles ties.
    """
    starts, ends, why = _score_candidates(
        edges_df=edges_df,
        startable_ids=startable_ids,
        require_all_edges=require_all_edges,
        undirected_edges=undirected_edges,
        t_list=t_list,
        horizon_s=horizon_s,
        max_recs=max_recs,
    )
    # If user explicitly wants "all stations" and there are few leaves, ensure distant endpoints are present:
    if (objective_nodes or require_all_stops) and len(starts) < max(10, max_recs//3):
        sig = edges_signature(edges_df)
        a,b = _approx_diameter_pair(sig, edges_df, None)
        starts = _unique_keep_order(([a,b] if a and b else []) + starts)
    return starts, ends, why

# ---------- geojson ----------
def geojson_to_geom(gj: dict):
    if not HAVE_SHAPELY:
        raise RuntimeError("Shapely is required for GeoJSON trimming. pip install shapely")
    gtype = gj.get("type", "").lower()
    if gtype == "featurecollection":
        feats = gj.get("features", [])
        geoms = [shape(f["geometry"]) for f in feats if f.get("geometry")]
        if not geoms: raise ValueError("Empty FeatureCollection.")
        return unary_union(geoms)
    elif gtype == "feature":
        geom = gj.get("geometry")
        if geom is None: raise ValueError("Feature has no geometry.")
        return shape(geom)
    elif gtype == "geometrycollection":
        geoms = [shape(g) for g in gj.get("geometries", []) if g]
        if not geoms: raise ValueError("Empty GeometryCollection.")
        return unary_union(geoms)
    else:
        return shape(gj)

# ---------- UI helpers ----------
def render_spatial_trimmer_ui(stops_df: pd.DataFrame, key_prefix: str) -> Optional[set]:
    """
    Renders UI for spatial trimming and returns a set of stop_ids to keep.
    Returns None if trimming is disabled or fails, indicating no change.
    """
    keep_ids: Optional[set] = None
    enable_trim =         st.checkbox("Enable trimming", value=False, key=f"{key_prefix}_trim_enable")
    if not enable_trim:
        return None

    trim_mode = st.radio("Trim by", ["Bounding box", "Viewport-like box", "GeoJSON polygon"],
                         horizontal=True, key=f"{key_prefix}_trim_by")
    lat_min0, lat_max0 = float(stops_df["lat"].min()), float(stops_df["lat"].max())
    lon_min0, lon_max0 = float(stops_df["lon"].min()), float(stops_df["lon"].max())

    if trim_mode == "Bounding box":
        c1, c2, c3, c4 = st.columns(4)
        with c1: min_lat =         st.number_input("Min lat", value=lat_min0, step=0.001, format="%.6f", key=f"{key_prefix}_min_lat")
        with c2: max_lat =         st.number_input("Max lat", value=lat_max0, step=0.001, format="%.6f", key=f"{key_prefix}_max_lat")
        with c3: min_lon =         st.number_input("Min lon", value=lon_min0, step=0.001, format="%.6f", key=f"{key_prefix}_min_lon")
        with c4: max_lon =         st.number_input("Max lon", value=lon_max0, step=0.001, format="%.6f", key=f"{key_prefix}_max_lon")
        keep_mask = (stops_df["lat"] >= min_lat) & (stops_df["lat"] <= max_lat) & \
                    (stops_df["lon"] >= min_lon) & (stops_df["lon"] <= max_lon)
        keep_ids = set(stops_df.loc[keep_mask, "stop_id"])

    elif trim_mode == "Viewport-like box":
        c1, c2, c3 = st.columns(3)
        with c1: center_lat =         st.number_input("Center lat", value=float(stops_df["lat"].median()), step=0.001, format="%.6f", key=f"{key_prefix}_center_lat")
        with c2: center_lon =         st.number_input("Center longitude", value=float(stops_df["lon"].median()), step=0.001, format="%.6f", key=f"{key_prefix}_center_lon")
        with c3: span_km =         st.slider("Span (km)", 1, 200, 20, 1, key=f"{key_prefix}_span_km")
        dlat = (span_km * 0.5) / 111.32
        dlon = (span_km * 0.5) / (111.32 * max(0.1, math.cos(math.radians(center_lat))))
        min_lat, max_lat = center_lat - dlat, center_lat + dlat
        min_lon, max_lon = center_lon - dlon, center_lon + dlon
        keep_mask = (stops_df["lat"] >= min_lat) & (stops_df["lat"] <= max_lat) & \
                    (stops_df["lon"] >= min_lon) & (stops_df["lon"] <= max_lon)
        keep_ids = set(stops_df.loc[keep_mask, "stop_id"])

    else:  # GeoJSON
        gj_file = st.file_uploader("GeoJSON (.geojson/.json)", type=["geojson", "json"], key=f"{key_prefix}_gj")
        keep_inside =         st.selectbox("Keep stops", ["inside", "outside"], index=0, key=f"{key_prefix}_keep_inside") == "inside"
        if gj_file and HAVE_SHAPELY:
            gj = json.load(gj_file)
            try:
                geom = geojson_to_geom(gj)
                if geom is not None:
                    mask = stops_df.apply(lambda r: (geom.contains(Point(r["lon"], r["lat"])) or geom.touches(Point(r["lon"], r["lat"]))), axis=1)
                    if not keep_inside: mask = ~mask
                    keep_ids = set(stops_df.loc[mask, "stop_id"])
            except Exception as e:
                st.error(f"GeoJSON error: {e}")
        elif gj_file and not HAVE_SHAPELY:
            st.error("Install shapely to use GeoJSON trimming: pip install shapely")

    return keep_ids


# ---------- UI ----------
st.set_page_config(page_title="🚌 GTFS Explorer — Unified Coverage Planner (Fast + Smart Starts)", layout="wide")
st.title("🚌 GTFS Explorer — Unified Coverage Planner")

uploaded = st.file_uploader("Upload a GTFS .zip", type=["zip"])
if uploaded is None:
    st.info("Upload a GTFS .zip to get started.")
    st.stop()

try:
    gtfs = load_gtfs_from_zip(uploaded)
except Exception as e:
    st.error(f"Failed to parse GTFS: {e}")
    st.stop()

# Sidebar: service + map theme
st.sidebar.header("Service / Filters")
service_date = st.sidebar.date_input("Service date", value=today_local_date())
service_ids = service_ids_for_date(gtfs, service_date)
trips_f = filter_trips_by_service(gtfs, service_ids)
st.sidebar.write(f"Active trips: **{len(trips_f):,}** {'(all trips)' if service_ids is None else ''}")

# Performance toggle
st.sidebar.header("Performance")
accelerator_options = ["CPU (NumPy)"]
if HAVE_CUPY:
    accelerator_options.append("GPU (CUDA / CuPy)")
if HAVE_MPS:
    accelerator_options.append("GPU (Apple Metal / PyTorch)")

default_ix = 0
if HAVE_CUPY:
    default_ix = accelerator_options.index("GPU (CUDA / CuPy)")
elif HAVE_MPS:
    default_ix = accelerator_options.index("GPU (Apple Metal / PyTorch)")

selected_accelerator_label = st.sidebar.radio(
    "Computation device",
    options=accelerator_options,
    index=default_ix,
    help=("Select the hardware for acceleration. "
          "CUDA requires an NVIDIA GPU. Apple Metal requires an M1/M2/M3 Mac."),
)
if "CUDA" in selected_accelerator_label:
    ACCELERATOR = "cuda"
elif "Apple Metal" in selected_accelerator_label:
    ACCELERATOR = "mps"
else:
    ACCELERATOR = "cpu"


# ---- Basemap styles with OSM fallback ----
st.sidebar.header("Map Theme")

MAPBOX_TOKEN = ""
try:
    MAPBOX_TOKEN = st.secrets.get("MAPBOX_API_KEY", "")
except Exception:
    pass
if not MAPBOX_TOKEN:
    MAPBOX_TOKEN = os.environ.get("MAPBOX_API_KEY", "") or os.environ.get("MAPBOX_TOKEN", "")

if MAPBOX_TOKEN:
    pdk.settings.mapbox_key = MAPBOX_TOKEN
    MAP_STYLES = {
        "OpenStreetMap (Light)": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        "OpenStreetMap (Dark)":  "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        "Mapbox Light":          "mapbox://styles/mapbox/light-v10",
        "Mapbox Dark":           "mapbox://styles/mapbox/dark-v10",
        "Mapbox Streets":        "mapbox://styles/mapbox/streets-v12",
        "Mapbox Satellite":      "mapbox://styles/mapbox/satellite-streets-v12",
    }
    default_style = "OpenStreetMap (Light)"
else:
    MAP_STYLES = {
        "OpenStreetMap (Light)": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        "OpenStreetMap (Dark)":  "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    }
    default_style = "OpenStreetMap (Light)"

theme = st.sidebar.selectbox("Basemap style", list(MAP_STYLES.keys()),
                             index=list(MAP_STYLES.keys()).index(default_style), key="map_theme")
map_style=_resolve_map_style(MAP_STYLES[theme])

ROUTE_TYPE_LABELS = {0:"Tram/Light rail",1:"Subway/Metro",2:"Rail",3:"Bus",4:"Ferry",5:"Cable tram",6:"Aerial lift",7:"Funicular",11:"Trolleybus",12:"Monorail"}

# Option to exclude stations with no service on selected date
exclude_no_service = st.sidebar.checkbox("Exclude stops without service on selected date", value=True)

# ---------- Build base edges/stops (serve-only) ----------
edges_all = build_time_edges_for_date(gtfs, trips_f)
# Precompute corridor groups for branch heuristic
try:
    _sig = edges_signature(edges_all)
    CORRIDOR_OF_STOP, CORRIDOR_NODES = _compute_corridors(_sig, edges_all)
    try:
        _ADJ_UNDIR = _build_adj_undirected(_sig, edges_all)
    except Exception:
        _ADJ_UNDIR = {}
except Exception as _e:
    CORRIDOR_OF_STOP, CORRIDOR_NODES = {}, {}


# Stops served by at least one *active* trip (after service-date filter)
if exclude_no_service:
    served_ids_all = set(
        gtfs.stop_times.merge(trips_f[["trip_id"]], on="trip_id", how="inner")["stop_id"].unique()
    )
else:
    served_ids_all = set(gtfs.stops["stop_id"].unique())

# Keep only stops served (global filter)
stops_raw_filtered = gtfs.stops.loc[
    gtfs.stops["stop_id"].isin(served_ids_all),
    ["stop_id", "stop_name", "stop_lat", "stop_lon"]
].copy()

# UI copy (lat/lon names)
stops_all = stops_raw_filtered.rename(
    columns={"stop_lat": "lat", "stop_lon": "lon"}
).reset_index(drop=True)

# Tabs — Planner first, plus Schedule Viewer
tab_plan, tab_sched, tab_net, tab_search = st.tabs(["Planner", "Schedule Viewer", "Network View", "Search GTFS"])

# ---- Schedule Viewer ----
# Helper utilities for the viewers
def _hex_to_rgb_list(hx: str):
    try:
        hx = str(hx).strip().lstrip("#")
        if len(hx) != 6:
            return [80, 80, 80]
        return [int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)]
    except Exception:
        return [80, 80, 80]

@st.cache_data(show_spinner=False)
def _service_trips_for_day(gtfs: GTFSData, day):
    srv = service_ids_for_date(gtfs, day)
    return filter_trips_by_service(gtfs, srv)

@st.cache_data(show_spinner=False)
def _trip_ends_for_trips(gtfs: GTFSData, trips_df: pd.DataFrame):
    stimes = gtfs.stop_times.merge(trips_df[["trip_id","route_id","service_id"]], on="trip_id", how="inner")
    smin = stimes.sort_values(["trip_id","stop_sequence"]).groupby("trip_id").first().reset_index()
    smax = stimes.sort_values(["trip_id","stop_sequence"]).groupby("trip_id").last().reset_index()
    def _to_sec(x):
        try:
            hh,mm,ss = str(x).split(":"); return int(hh)*3600+int(mm)*60+int(ss)
        except Exception:
            return None
    smin["dep_s0"] = smin["departure_time"].map(_to_sec)
    smax["arr_s1"] = smax["arrival_time"].map(_to_sec)
    ends = smin[["trip_id","stop_id","dep_s0"]].merge(
        smax[["trip_id","stop_id","arr_s1"]], on="trip_id", how="inner", suffixes=("_first","_last")
    )
    ends = ends.merge(trips_df[["trip_id","route_id"]], on="trip_id", how="left")
    ends["elapsed_s"] = (pd.to_numeric(ends["arr_s1"]) - pd.to_numeric(ends["dep_s0"])).astype("Int64")
    slookup = gtfs.stops.set_index("stop_id")["stop_name"].to_dict()
    ends["first_name"] = ends["stop_id_first"].map(lambda s: slookup.get(s, str(s)))
    ends["last_name"]  = ends["stop_id_last"].map(lambda s: slookup.get(s, str(s)))
    return ends

def _hhmm(total_seconds: int):
    try:
        if total_seconds is None: return "--:--"
        total_seconds = int(total_seconds)
        hh = total_seconds // 3600
        mm = (total_seconds % 3600) // 60
        return f"{hh:02d}:{mm:02d}"
    except Exception:
        return "--:--"

def _format_timerange(a_s: int, b_s: int):
    return f"{_hhmm(a_s)} - {_hhmm(b_s)}" if a_s is not None and b_s is not None else "-"

def _compute_departure_seconds_for_origin(gtfs: GTFSData, trips_subset: pd.DataFrame):
    stimes = gtfs.stop_times.merge(trips_subset[["trip_id","stop_id_first"]], left_on=["trip_id","stop_id"], right_on=["trip_id","stop_id_first"], how="inner")
    def _to_sec(x):
        try:
            hh,mm,ss = str(x).split(":"); return int(hh)*3600+int(mm)*60+int(ss)
        except Exception:
            return None
    deps = [ _to_sec(x) for x in stimes["departure_time"].dropna().tolist() ]
    return sorted([d for d in deps if d is not None])

def _build_headway_segments(dep_seconds, tolerance_min=1, min_seg_minutes=30):
    if not dep_seconds or len(dep_seconds) < 2:
        return []
    gaps = [ (dep_seconds[i+1] - dep_seconds[i]) // 60 for i in range(len(dep_seconds)-1) ]
    segments = []
    seg_start_idx = 0
    def headway_label(vals):
        uniq = sorted(set(vals))
        if len(uniq) == 1:
            return str(uniq[0])
        if len(uniq) == 2 and abs(uniq[0]-uniq[1]) <= max(1, tolerance_min):
            return f"{uniq[0]}／{uniq[1]}"
        import statistics
        return str(int(round(statistics.median(vals))))
    for i in range(1, len(gaps)):
        if abs(gaps[i] - gaps[i-1]) <= tolerance_min:
            continue
        vals = gaps[seg_start_idx:i]
        segments.append({
            "start_s": dep_seconds[seg_start_idx],
            "end_s": dep_seconds[i],
            "label": headway_label(vals),
            "duration_min": (dep_seconds[i] - dep_seconds[seg_start_idx]) // 60
        })
        seg_start_idx = i
    vals = gaps[seg_start_idx:]
    segments.append({
        "start_s": dep_seconds[seg_start_idx],
        "end_s": dep_seconds[-1],
        "label": headway_label(vals),
        "duration_min": (dep_seconds[-1] - dep_seconds[seg_start_idx]) // 60
    })
    merged = []
    for seg in segments:
        if merged and (seg["duration_min"] < min_seg_minutes or seg["label"] == merged[-1]["label"]):
            merged[-1]["end_s"] = seg["end_s"]
            merged[-1]["duration_min"] = (merged[-1]["end_s"] - merged[-1]["start_s"]) // 60
        else:
            merged.append(seg)
    return merged

def _route_color_lookup(gtfs: GTFSData):
    if "route_color" in gtfs.routes.columns:
        return gtfs.routes.set_index("route_id")["route_color"].to_dict()
    return {}

with tab_sched:
    st.subheader("Schedule Viewer")
    sub_route, sub_stop = st.tabs(["Route Viewer", "Stop Viewer"])

    # --------------- ROUTE VIEWER ---------------
    with sub_route:
        day_r = st.date_input("Service date", value=dt.date.today(), key="route_date")
        trips_day = _service_trips_for_day(gtfs, day_r)
        routes_df = gtfs.routes.copy()
        fmt_route = lambda r: f"{r.get('route_short_name','')} — {r.get('route_long_name','')}" if r.get('route_long_name') else r.get('route_short_name', r.get('route_id'))
        route_options = routes_df.to_dict("records") if not routes_df.empty else []
        sel_route = st.selectbox("Route", options=route_options, format_func=fmt_route, key="route_sel")
        if sel_route:
            rid = sel_route.get("route_id")
            rtrips = trips_day[trips_day["route_id"].astype(str)==str(rid)].copy()
            if "direction_id" not in rtrips.columns:
                rtrips["direction_id"] = pd.NA
            ends = _trip_ends_for_trips(gtfs, rtrips)
            ends = ends.merge(rtrips[["trip_id","trip_headsign","direction_id"]], on="trip_id", how="left")

            use_group = st.checkbox("Group multiple termini in the same direction (treat branches as one)", value=True, key="group_termini")
            if "direction_id" in ends.columns and ends["direction_id"].notna().any() and use_group:
                sample_labels = ends.groupby("direction_id").apply(lambda g: (g["first_name"].mode().iloc[0] if not g["first_name"].mode().empty else "Origin") + " → " + (g["last_name"].mode().iloc[0] if not g["last_name"].mode().empty else "Destination")).to_dict()
                dir_choices = [d for d in sorted(ends["direction_id"].dropna().unique().tolist())]
                dir_label = lambda d: f"Direction {int(d)} — {sample_labels.get(d,'')}" if pd.notna(d) else "Unknown"
                sel_dir = st.selectbox("Direction", options=dir_choices, format_func=dir_label, key="route_dirid")
                chosen = ends[ends["direction_id"]==sel_dir]
                dir_name_display = sample_labels.get(sel_dir, "")
            else:
                ends["pair"] = ends["first_name"] + " → " + ends["last_name"]
                pairs = sorted(ends["pair"].unique().tolist())
                sel_pair = st.selectbox("Direction / Terminus", options=pairs, key="route_dirpair")
                a,b = sel_pair.split(" → ")
                chosen = ends[(ends["first_name"]==a) & (ends["last_name"]==b)]
                dir_name_display = sel_pair

            dep_seconds = _compute_departure_seconds_for_origin(gtfs, chosen)
            detail_level = st.radio("Headway detail", options=["Concise","Detailed"], horizontal=True, key="head_detail")
            min_seg = 30 if detail_level=="Concise" else 10
            segments = _build_headway_segments(dep_seconds, tolerance_min=1, min_seg_minutes=min_seg)
            if segments:
                head_df = pd.DataFrame([{"Service Hours": _format_timerange(s["start_s"], s["end_s"]), "Headway (Minutes)": s["label"], "Terminus": dir_name_display.split(" → ")[-1] if " → " in dir_name_display else dir_name_display} for s in segments])
                st.markdown("**Headway summary**")
                st.dataframe(head_df, use_container_width=True, hide_index=True)
            else:
                st.info("No departures found to compute headways for this selection.")

            rtr_nonnull = chosen.dropna(subset=["elapsed_s"]).sort_values("elapsed_s")
            if not rtr_nonnull.empty:
                fastest_min = int(rtr_nonnull["elapsed_s"].min() // 60)
                longest_min = int(rtr_nonnull["elapsed_s"].max() // 60)
                avg_min = int(round(rtr_nonnull["elapsed_s"].mean() / 60.0))
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fastest trip", f"{fastest_min} min")
                with col2:
                    st.metric("Average trip", f"{avg_min} min")
                with col3:
                    st.metric("Longest trip", f"{longest_min} min")

            with st.expander("Detailed timetable by origin (grouped by hour)"):
                for origin_id, sub in chosen.groupby("stop_id_first"):
                    oname = sub.iloc[0]["first_name"]
                    st.markdown(f"**Departures from {oname}**")
                    tids = sub["trip_id"].tolist()
                    dep_rows = gtfs.stop_times[gtfs.stop_times["trip_id"].isin(tids) & (gtfs.stop_times["stop_id"].astype(str)==str(origin_id))][["trip_id","departure_time"]].copy()
                    dep_rows = dep_rows.merge(chosen[["trip_id","trip_headsign"]], on="trip_id", how="left").sort_values("departure_time")
                    # build hour -> list of minutes
                    by_hour = {}
                    for t in dep_rows["departure_time"].dropna().tolist():
                        try:
                            hh,mm,_ = t.split(":")
                            by_hour.setdefault(hh, []).append(mm)
                        except Exception:
                            pass
                    if by_hour:
                        # show a table instead of code pad
                        import pandas as _pd
                        data = [{"Hour": hh, "Minutes": ", ".join(sorted(set(by_hour[hh])))} for hh in sorted(by_hour.keys())]
                        st.dataframe(_pd.DataFrame(data), use_container_width=True, hide_index=True)
                    # tabular view for reference
                    dep_rows = dep_rows.rename(columns={"departure_time":"Departure","trip_headsign":"Headsign"})
                    st.dataframe(dep_rows, use_container_width=True, hide_index=True)

            st.markdown("**Route map**")
            sample = chosen.sort_values("elapsed_s").head(3)["trip_id"].tolist() if len(chosen)>=3 else chosen["trip_id"].tolist()
            seq = gtfs.stop_times[gtfs.stop_times["trip_id"].isin(sample)].sort_values(["trip_id","stop_sequence"])
            coords = gtfs.stops[["stop_id","stop_lat","stop_lon"]].rename(columns={"stop_lat":"lat","stop_lon":"lon"})
            seq = seq.merge(coords, on="stop_id", how="left")
            seg_rows = []
            for tid, grp in seq.groupby("trip_id"):
                grp = grp.reset_index(drop=True)
                for i in range(len(grp)-1):
                    seg_rows.append({"from_lat": float(grp.loc[i,"lat"]), "from_lon": float(grp.loc[i,"lon"]), "to_lat": float(grp.loc[i+1,"lat"]), "to_lon": float(grp.loc[i+1]["lon"]), "trip_id": tid})
            seg_df = pd.DataFrame(seg_rows)
            rcolor = _route_color_lookup(gtfs).get(rid, None)
            col = _hex_to_rgb_list(rcolor) if rcolor else [24, 119, 242]
            st.pydeck_chart(pdk.Deck(
                map_style=_resolve_map_style(map_style),
                initial_view_state=pdk.ViewState(latitude=coords["lat"].median() if not coords.empty else 0.0,
                                                 longitude=coords["lon"].median() if not coords.empty else 0.0,
                                                 zoom=11),
                layers=[
                    pdk.Layer("LineLayer", data=seg_df,
                              get_source_position="[from_lon,from_lat]",
                              get_target_position="[to_lon,to_lat]",
                              get_width=3,
                              get_color=col,
                              pickable=False),
                ],
            ))

    # --------------- STOP VIEWER ---------------
    with sub_stop:
        day_s = st.date_input("Service date", value=dt.date.today(), key="stop_date")
        trips_day_s = _service_trips_for_day(gtfs, day_s)

        use_group = st.checkbox("Group nearby / similar-named stops (like the planner)", value=False, key="stop_group")
        if use_group:
            groups_df, mapping = group_stops_by_proximity_and_name(
                stops_raw_filtered, dist_m=50, name_thr=0.3, jaccard_thr=0.5, min_shared_tokens=1, distance_override_m=25.0
            )
            stops_view = groups_df.rename(columns={"group_id":"stop_id"})
            stop_name_lookup = stops_view.set_index('stop_id')['stop_name'].to_dict()
            options = stops_view["stop_id"].tolist()
            fmt_stop = lambda sid: f"{sid} — {stop_name_lookup.get(sid, '')}"
        else:
            options = stops_all["stop_id"].tolist()
            stop_name_lookup = stops_all.set_index('stop_id')['stop_name'].to_dict()
            fmt_stop = lambda sid: f"{sid} — {stop_name_lookup.get(sid, '')}"

        sel_stop = st.selectbox("Stop", options=options, format_func=fmt_stop, key="stop_sel")
        sid_list = []
        if use_group and sel_stop:
            gmap = mapping
            inv = {}
            for s, g in gmap.items():
                inv.setdefault(g, []).append(s)
            sid_list = inv.get(sel_stop, [])
        else:
            sid_list = [sel_stop] if sel_stop else []

        if sel_stop:
            stimes = gtfs.stop_times.merge(trips_day_s[["trip_id","route_id","service_id"]], on="trip_id", how="inner")
            stimes = stimes[stimes["stop_id"].astype(str).isin([str(s) for s in sid_list])].copy()
            if "route_short_name" in gtfs.routes.columns:
                stimes = stimes.merge(gtfs.routes[["route_id","route_short_name","route_long_name"]], on="route_id", how="left")
            def _to_sec(x):
                try:
                    hh,mm,ss = str(x).split(":"); return int(hh)*3600+int(mm)*60+int(ss)
                except Exception:
                    return None
            if "departure_time" in stimes.columns:
                stimes["dep_s"] = stimes["departure_time"].map(_to_sec)

            ends_all = _trip_ends_for_trips(gtfs, trips_day_s)
            if "direction_id" in trips_day_s.columns:
                ends_all = ends_all.merge(trips_day_s[["trip_id","direction_id"]], on="trip_id", how="left")
                key_cols = ["route_id","direction_id"]
            else:
                key_cols = ["route_id","first_name","last_name"]
            fastest_by_key = ends_all.sort_values("elapsed_s").dropna(subset=["elapsed_s"]).groupby(key_cols).first().reset_index()
            fastest_ids = set(fastest_by_key["trip_id"].tolist())

            stimes["fastest_for_route"] = stimes["trip_id"].map(lambda t: t in fastest_ids)
            show_cols = [c for c in ["departure_time","route_short_name","route_long_name","trip_id"] if c in stimes.columns]
            show_cols += [c for c in ["stop_sequence","trip_headsign"] if c in stimes.columns]
            if "fastest_for_route" in stimes.columns:
                show_cols.append("fastest_for_route")
            st.markdown(f"**Departures at {stop_name_lookup.get(sel_stop, sel_stop)}**")
            st.dataframe(stimes.sort_values("dep_s")[show_cols], use_container_width=True, hide_index=True)

            rids = sorted(stimes["route_id"].dropna().unique().tolist())
            if rids:
                st.markdown("**Map: routes through this stop**")
                edges = edges_all[edges_all["route_id"].isin(rids)][["from_stop","to_stop","route_id"]].drop_duplicates().copy()
                coord = stops_all.set_index("stop_id")[["lat","lon"]]
                edges = edges.merge(coord, left_on="from_stop", right_index=True, how="left")
                edges = edges.merge(coord, left_on="to_stop", right_index=True, how="left", suffixes=("_from","_to"))
                rcolor = _route_color_lookup(gtfs)
                edges["color"] = edges["route_id"].map(lambda r: _hex_to_rgb_list(rcolor.get(r, None)) if rcolor else [24,119,242])
                stops_sel = stops_all[stops_all["stop_id"].astype(str).isin([str(s) for s in sid_list])][["lat","lon"]].copy()
                st.pydeck_chart(pdk.Deck(
                    map_style=_resolve_map_style(map_style),
                    initial_view_state=pdk.ViewState(latitude=stops_sel["lat"].mean() if not stops_sel.empty else (stops_all["lat"].median() if not stops_all.empty else 0.0),
                                                     longitude=stops_sel["lon"].mean() if not stops_sel.empty else (stops_all["lon"].median() if not stops_all.empty else 0.0),
                                                     zoom=12),
                    layers=[
                        pdk.Layer("LineLayer", data=edges,
                                  get_source_position="[lon_from,lat_from]",
                                  get_target_position="[lon_to,lat_to]",
                                  get_width=3,
                                  get_color="color",
                                  pickable=False),
                        pdk.Layer("ScatterplotLayer", data=stops_sel.assign(size=12),
                                  get_position="[lon,lat]",
                                  get_radius=30,
                                  get_fill_color=[255,0,0],
                                  pickable=False),
                    ],
                ))
# ---- Network View ----
with tab_net:
    st.subheader("Network View")
    MAP_STYLES = globals().get("MAP_STYLES", {})
    theme = next(iter(MAP_STYLES.keys())) if MAP_STYLES else "Light"
    map_style = st.selectbox("Map style", options=list(MAP_STYLES.keys()) if MAP_STYLES else ["Light","Dark","Outdoors","Streets","Satellite"], index=0, key="net_mapstyle")
    # Filter by GTFS route_type
    avail_types_net = sorted(gtfs.routes["route_type"].dropna().unique().tolist()) if "route_type" in gtfs.routes.columns else []
    sel_types_net = st.multiselect("Route types to show (empty = all)", options=avail_types_net, default=[], key="net_modes")
    if sel_types_net:
        routes_show = gtfs.routes[gtfs.routes["route_type"].astype(int).isin(sel_types_net)][["route_id"]]
        edges_net = edges_all.merge(routes_show, on="route_id", how="inner")
    else:
        edges_net = edges_all.copy()
    try:
        _edges = edges_net[["from_stop","to_stop","route_id"]].drop_duplicates().copy()
        coord = stops_all.set_index("stop_id")[["lat","lon"]]
        _edges = _edges.merge(coord, left_on="from_stop", right_index=True, how="left")
        _edges = _edges.merge(coord, left_on="to_stop", right_index=True, how="left", suffixes=("_from","_to"))
        # color-code by route if available
        def _hex_to_rgb_list(hx: str):
            try:
                hx = str(hx).strip().lstrip("#")
                if len(hx) != 6: return [24,119,242]
                return [int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)]
            except Exception:
                return [24,119,242]
        rcolor = gtfs.routes.set_index("route_id")["route_color"].to_dict() if "route_color" in gtfs.routes.columns else {}
        _edges["color"] = _edges["route_id"].map(lambda r: _hex_to_rgb_list(rcolor.get(r, None)) if rcolor else [24,119,242])
        # Stop points
        _stops = stops_all[["lat","lon"]].rename(columns={"lat":"lat","lon":"lon"})
        st.pydeck_chart(pdk.Deck(
            map_style=_resolve_map_style(map_style),
            initial_view_state=pdk.ViewState(latitude=stops_all["lat"].median() if not stops_all.empty else 0.0,
                                             longitude=stops_all["lon"].median() if not stops_all.empty else 0.0,
                                             zoom=11),
            layers=[
                pdk.Layer("LineLayer", data=_edges,
                          get_source_position="[lon_from,lat_from]",
                          get_target_position="[lon_to,lat_to]",
                          get_width=2,
                          get_color="color",
                          pickable=False),
                pdk.Layer("ScatterplotLayer", data=_stops,
                          get_position="[lon,lat]",
                          get_radius=40,
                          pickable=True),
            ]
        ))
    except Exception as _e:
        st.caption(f"Network map unavailable: {_e}")
# ---- Search GTFS ----
with tab_search:
    st.subheader("Search GTFS")
    # Search options
    col_s1, col_s2, col_s3 = st.columns([2,1,1])
    with col_s1:
        q = st.text_input("Search by stop / route / trip / headsign", key="search_q")
    with col_s2:
        mode = st.selectbox("Match mode", ["All words", "Any word", "Exact phrase"], index=0, help="How to match your query against text fields.")
    with col_s3:
        use_fuzzy = st.checkbox("Fuzzy (typo tolerant)", value=True, help="Includes close matches using simple string similarity.")

    def _norm(x):
        import unicodedata
        s = str(x).lower().strip()
        try:
            s = unicodedata.normalize('NFKD', s).encode('ascii','ignore').decode('ascii')
        except Exception:
            pass
        return s

    def _tokenize(q):
        return [t for t in re.split(r"\s+", _norm(q)) if t]

    def _match_series(series, q):
        s = series.astype(str).map(_norm)
        if mode == "Exact phrase":
            pat = re.escape(_norm(q))
            mask = s.str.contains(pat, regex=True, na=False)
        else:
            toks = _tokenize(q)
            if not toks:
                return s.index[:0]
            if mode == "All words":
                mask = True
                for t in toks:
                    mask = mask & s.str.contains(re.escape(t), regex=True, na=False)
            else:  # Any word
                mask = False
                for t in toks:
                    mask = mask | s.str.contains(re.escape(t), regex=True, na=False)
        idx = s.index[mask] if hasattr(mask, "__iter__") else s.index[s == True]
        if use_fuzzy and len(idx) < 10 and len(s) > 0 and len(_norm(q)) >= 3:
            # bring in a few fuzzy matches by SequenceMatcher ratio > 0.8
            import difflib, pandas as _pd
            candidates = s.sample(min(1000, len(s)), random_state=0) if len(s) > 1000 else s
            ratios = candidates.map(lambda v: difflib.SequenceMatcher(None, v, _norm(q)).ratio())
            fuzzy_idx = ratios[ratios >= 0.8].index
            idx = _pd.Index(sorted(set(idx).union(set(fuzzy_idx))))
        return idx

    if q:
        # Stops
        stops_cols = [c for c in ["stop_id","stop_name","stop_desc"] if c in gtfs.stops.columns]
        stop_idx = _match_series(gtfs.stops.get("stop_name", gtfs.stops.get("stop_id")), q)
        res_stops = gtfs.stops.loc[stop_idx, stops_cols].copy().head(200) if len(stop_idx)>0 else gtfs.stops.head(0)
        # Routes
        route_cols = [c for c in ["route_id","route_short_name","route_long_name","route_desc","route_type"] if c in gtfs.routes.columns]
        route_idx = _match_series(gtfs.routes.get("route_long_name", gtfs.routes.get("route_short_name", gtfs.routes.get("route_id"))), q)
        res_routes = gtfs.routes.loc[route_idx, route_cols].copy().head(200) if len(route_idx)>0 else gtfs.routes.head(0)
        # Trips & Headsigns
        trips_cols = [c for c in ["trip_id","route_id","trip_headsign","direction_id","service_id"] if c in gtfs.trips.columns]
        trip_target = gtfs.trips.get("trip_headsign", gtfs.trips.get("trip_id"))
        trip_idx = _match_series(trip_target, q)
        res_trips = gtfs.trips.loc[trip_idx, trips_cols].copy().head(200) if len(trip_idx)>0 else gtfs.trips.head(0)

        st.markdown("#### Stops")
        st.dataframe(res_stops, use_container_width=True, hide_index=True)
        st.markdown("#### Routes")
        st.dataframe(res_routes, use_container_width=True, hide_index=True)
        st.markdown("#### Trips")
        st.dataframe(res_trips, use_container_width=True, hide_index=True)
    else:
        st.info("Enter a query to search stops, routes, and trips. Tip: toggle **Match mode** and **Fuzzy** for broader results.")
# ---- Planner ----
with tab_plan:
    with st.expander('Planner Settings', expanded=True):
        st.subheader("Planner — Elapsed Time First (with unified coverage constraints)")


    with st.expander("Heuristics & Transfers"):
        use_branch_lookahead =         st.checkbox("Enable branch-aware lookahead", value=True,
                                           help="Look ahead a few steps to avoid leaving a branch unfinished.")
        st.session_state["use_branch_lookahead"] = bool(use_branch_lookahead)
        colA, colB, colC = st.columns(3)
        with colA:
            lookahead_depth =         st.slider("Lookahead depth (steps)", 0, 6, 3)
            st.session_state["lookahead_depth"] = int(lookahead_depth)
        with colB:
            w_return =         st.number_input("Weight: return-to-branch (α)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            st.session_state["w_return"] = float(w_return)
        with colC:
            w_coverage =         st.number_input("Weight: coverage lookahead (γ)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            st.session_state["w_coverage"] = float(w_coverage)
        discourage_transfers =         st.checkbox("Discourage unnecessary route transfers", value=True,
                                           help="Add a cost for switching routes unless it helps coverage/time.")
        st.session_state["discourage_transfers"] = bool(discourage_transfers)
        transfer_penalty_s =         st.slider("Per-transfer penalty (β seconds)", 0, 600, 120, 10)
        st.session_state["transfer_penalty_s"] = int(transfer_penalty_s)
    # Fast mode & early stop
    fast_mode =         st.checkbox("⚡ Fast mode (auto-tune search for speed)", value=True, help="Auto-scales beam & candidates by graph size.", key="plan_fast")
    stop_on_first_full =         st.checkbox("Stop search on first full-coverage plan", value=True, key="plan_stop_first_full")

    # Mode filter
    avail_types_plan = sorted(gtfs.routes["route_type"].dropna().astype(int).unique().tolist())
    sel_types_plan =         st.multiselect("Allowed modes (empty = all)", options=avail_types_plan, default=[],
                                    format_func=lambda t: f"{t} — {ROUTE_TYPE_LABELS.get(t,'Unknown')}", key="plan_modes")
    edges_df = edges_all[edges_all["route_type"].isin(sel_types_plan)].reset_index(drop=True) if sel_types_plan else edges_all.copy()
    stops_view = stops_all.copy()
    stop_name_lookup = stops_all.set_index('stop_id')['stop_name'].to_dict()

    # Grouping for planning
    group_plan =         st.checkbox("Group stops for planning", value=False, key="plan_group_merge")
    groups_df = None
    mapping = {}
    if group_plan:
        dist_m =         st.slider("Merge distance (m)", 50, 800, 200, 25, key="plan_merge_dist")
        name_thr =         st.slider("Name similarity (SequenceMatcher)", 0.5, 1.0, 0.80, 0.01, key="plan_merge_name")

        with st.expander("Advanced matching"):
            jac_thr =         st.slider("Token Jaccard threshold", 0.0, 1.0, 0.50, 0.05, key="plan_jac_thr")
            min_shared =         st.number_input("Minimum shared tokens", 0, 5, 1, 1, key="plan_min_shared")
            dist_override =         st.slider("Distance-only override (m)", 0, 100, 25, 5, key="plan_dist_override")

        groups_df, mapping = group_stops_by_proximity_and_name(
            stops_raw_filtered, dist_m, name_thr,
            jaccard_thr=float(jac_thr), min_shared_tokens=int(min_shared), distance_override_m=float(dist_override)
        )
        groups_df = relabel_groups_by_rule(groups_df, mapping, gtfs, trips_f)
        stops_view = groups_df.rename(columns={"group_id":"stop_id"})
        edges_df = remap_edges_to_groups(edges_df, mapping)
        stop_name_lookup = stops_view.set_index('stop_id')['stop_name'].to_dict()

        # ---- Grouped stops viewer (only builds when opened) ----
        with st.expander("Grouped stops (viewer)"):
            inv = {}
            for sid, gid in mapping.items():
                inv.setdefault(gid, []).append(sid)
            def members_for(gid: str) -> List[Tuple[str,str]]:
                sids = inv.get(gid, [])
                sub = stops_raw_filtered[stops_raw_filtered["stop_id"].isin(sids)][["stop_id","stop_name"]]
                return list(sub.itertuples(index=False, name=None))
            gv = groups_df.copy()
            gv["members_count"] = gv["group_id"].map(lambda gid: len(inv.get(gid, [])))
            gv["member_stop_ids"] = gv["group_id"].map(lambda gid: ", ".join(inv.get(gid, [])[:20]) + (" ..." if len(inv.get(gid, []))>20 else ""))
            gv["member_stop_names"] = gv["group_id"].map(lambda gid: ", ".join([nm for (_sid,nm) in members_for(gid)][:10]) + (" ..." if len(inv.get(gid, []))>10 else ""))

            group_display_names = gv.set_index('group_id')['stop_name'].to_dict()
            group_options = ["(All)"] + gv["group_id"].tolist()
            sel_gid =         st.selectbox(
                "Select a group to inspect",
                options=group_options,
                format_func=lambda gid: f"{group_display_names.get(gid, gid)}" if gid != "(All)" else "(All)",
                key="plan_group_viewer_gid"
            )

            if sel_gid == "(All)":
                st.dataframe(gv[["group_id","stop_name","size","members_count","member_stop_ids","member_stop_names"]],
                             use_container_width=True, hide_index=True)
            else:
                rows = [{"stop_id": sid, "stop_name": nm} for (sid,nm) in members_for(sel_gid)]
                left = gv[gv["group_id"]==sel_gid][["group_id","stop_name","size","members_count"]]
                st.dataframe(left, use_container_width=True, hide_index=True)
                st.markdown("**Members**")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Trimmer (affects planning)
    with st.expander("Spatial Trimmer (affects planning)"):
        keep_ids = render_spatial_trimmer_ui(stops_view, "plan")
        if keep_ids is not None:
            stops_view = stops_view[stops_view["stop_id"].isin(keep_ids)].copy()
            edges_df = edges_df[(edges_df["from_stop"].isin(keep_ids)) & (edges_df["to_stop"].isin(keep_ids))].reset_index(drop=True)


    # Keep only stops that appear in the current planning graph (after all planner filters)
    _used_plan_ids = set(edges_df["from_stop"]).union(set(edges_df["to_stop"]))
    stops_view = stops_view[stops_view["stop_id"].isin(_used_plan_ids)].reset_index(drop=True)

    # ---- Objective & Coverage ----
    st.markdown("### Objective & Coverage")
    obj_choice = st.radio("Optimize coverage for", ["Stops","Edges"], index=0, horizontal=True, key="obj_choice")
    obj_is_stops = (obj_choice == "Stops")
    undirected_edges =         st.checkbox("Treat edges as undirected for counting", value=True, disabled=obj_is_stops, key="plan_undirected")

    require_all_stops =         st.checkbox("Require visiting ALL stops", value=False, key="req_all_stops")
    require_all_edges =         st.checkbox("Require traversing ALL edges", value=False, key="req_all_edges")

    # Search behavior
    st.markdown("### Search behavior")
    prefer_new =         st.checkbox("Prefer legs that add new coverage (soft preference)", value=True,
                             help="Still allows revisits when needed (enables true backtracking).", key="prefer_new_soft")
    zero_wait_same_route =         st.checkbox("0 connection wait if next leg is same route (selection only)", value=False, key="zero_wait_same_route")
    strict_elapsed =         st.checkbox("Strictly minimize incremental elapsed time at each step", value=True,
                                 help="Primary sort = (wait + ride). New coverage acts only as a soft tie-breaker.", key="strict_elapsed_step")

    # Start time controls
    st.markdown("### Start time / Budget")
    optimize_time =         st.checkbox("Optimize start time across a window", value=True, key="plan_optimize_time")
    if optimize_time:
        c1,c2,c3 = st.columns(3)
        with c1: win_from_h =         st.number_input("Window start hour", 0, 47, 6, key="plan_win_from")
        with c2: win_to_h   =         st.number_input("Window end hour", 0, 47, 10, key="plan_win_to")
        with c3: step_min   =         st.number_input("Step (minutes)", 5, 60, 15, 5, key="plan_step_min")

        # Option to use exact departures instead of fixed step minutes
        scan_exact_departures =         st.checkbox(
            "Scan all departures in window (exact)",
            value=st.session_state.get("plan_scan_departures", True),
            key="plan_scan_departures",
            help="Use every scheduled departure observed at the chosen start stops within this window."
        )
    else:
        c1,c2 = st.columns(2)
        with c1: start_time_h =         st.number_input("Start hour", 0, 47, 8, key="plan_start_h")
        with c2: start_time_m =         st.number_input("Start minute", 0, 59, 0, key="plan_start_m")
        start_time_s = start_time_h*3600 + start_time_m*60

    horizon_min =         st.slider("Time budget (minutes)", 60, 24*60, 12*60, 30,
                            disabled=(require_all_stops or require_all_edges), key="plan_horizon_min")

    # Build t_list and window info (used for availability-aware smart starts)
    if optimize_time:
        t_list=[]; base=int(win_from_h)*3600; end=int(win_to_h)*3600; step=int(step_min)*60; t=base
        while t<=end: t_list.append(int(t)); t+=step
    else:
        t_list=[int(start_time_s)]
    ignore_deadline = require_all_stops or require_all_edges
    horizon_s = (14*24*3600) if ignore_deadline else int(horizon_min*60)

    # Start selection (with Smart Starts)
    
    # Coverage selections — required stops (optional)
    with st.expander("Coverage selections", expanded=False):
        if "stop_id" in stops_all.columns and "stop_name" in stops_all.columns:
            _stop_opts = stops_all[["stop_id","stop_name"]].copy()
            _stop_opts["label"] = _stop_opts["stop_id"].astype(str) + " — " + _stop_opts["stop_name"].astype(str)
            _sel_labels = st.multiselect("Stops you need to visit (optional)", options=_stop_opts["label"].tolist(), default=[], key="coverage_required_stops")
            # Save required stop_ids in session_state for planner logic to consume
            _lab2id = dict(zip(_stop_opts["label"], _stop_opts["stop_id"]))
            st.session_state["coverage_required_stop_ids"] = [ _lab2id.get(lbl) for lbl in _sel_labels ]
        else:
            st.info("Stops table missing 'stop_id' / 'stop_name' columns; cannot select required stops.")
    st.markdown("### Start selection")
    startable_ids = set(edges_df["from_stop"].unique()).union(set(edges_df["to_stop"].unique()))
    smart_starts = st.checkbox("Use smart start recommendations (peripheral, Eulerian if needed, availability-aware)", value=True, key="plan_smart_starts")

    from_loc = st.checkbox("Start from a location (lat/lon)", value=False, key="plan_from_loc")
    loc_candidates=[]
    if from_loc:
        c1,c2,c3 = st.columns(3)
        with c1: lat0 =         st.number_input("Start latitude", value=float(stops_view["lat"].median()), format="%.6f", key="plan_lat0")
        with c2: lon0 =         st.number_input("Start longitude", value=float(stops_view["lon"].median()), format="%.6f", key="plan_lon0")
        with c3: k_near =         st.number_input("Use nearest N stops", min_value=1, max_value=50, value=5, step=1, key="plan_k_near")
        sv = stops_view.copy()
        sv["dist_m"] = sv.apply(lambda r: haversine_m(lat0, lon0, r["lat"], r["lon"]), axis=1)
        sv = sv.sort_values("dist_m")
        loc_candidates = [sid for sid in sv["stop_id"].tolist() if sid in startable_ids][:int(k_near)]
        st.caption(f"Using {len(loc_candidates)} nearest candidates.")
    manual_starts =         st.multiselect(
        "Manual start stops (optional)",
        options=sorted(list(startable_ids)),
        default=[],
        key="plan_manual_starts",
        format_func=lambda sid: f"{stop_name_lookup.get(sid, sid)} ({sid})"
    )
    use_manual = (len(manual_starts) > 0) and (not from_loc)

    # Heuristics tuning
    st.markdown("### Heuristics")
    c1,c2,c3 = st.columns(3)
    with c1: greedy_restarts =         st.slider("Greedy restarts", 3, 60, 20, 1, key="plan_restarts")
    with c2: k_cands =         st.slider("Candidates per step", 10, 200, 50, 5, key="plan_k_cands")
    with c3: explore =         st.slider("Exploration (random %)", 0.0, 0.2, 0.01, 0.01, key="plan_explore")

    st.markdown("### Backtracking (Beam)")
    enable_beam =         st.checkbox("Enable backtracking / beam search", value=True, key="plan_enable_beam")
    if enable_beam:
        c1,c2,c3 = st.columns(3)
        with c1: beam_width =         st.number_input("Beam width", 1, 40, 10, 1, key="plan_beam_width")
        with c2: per_parent =         st.number_input("Branches per parent", 1, 10, 4, 1, key="plan_per_parent")
        with c3: base_depth =         st.number_input("Base max steps", 50, 20000, 700, 50, key="plan_base_depth")

        with st.expander("Advanced backtracking tuning"):
            st.caption("Aggressive mode widens/deepens automatically when coverage stalls.")
            cA,cB,cC = st.columns(3)
            with cA: aggressive_backtrack =         st.checkbox("Coverage-guided (aggressive)", value=True, key="plan_aggr")
            with cB: dom_slack_s =         st.number_input("Dominance pruning slack (s)", 0, 600, 0, 10, key="plan_dom_slack")
            with cC: widen_factor =         st.slider("Max widen factor (×)", 1.0, 4.0, 2.0, 0.5, key="plan_widen_factor")
            cD,cE = st.columns(2)
            with cD: stagnation_patience =         st.number_input("Stagnation patience (levels)", 3, 50, 10, 1, key="plan_stag_pat")
            with cE: lookahead_k =         st.number_input("Coverage look-ahead (K edges)", 0, 60, 10, 1, key="plan_lookahead_k")
    else:
        beam_width, per_parent, base_depth = 10, 4, 700
        aggressive_backtrack = True
        dom_slack_s = 0
        widen_factor = 2.0
        stagnation_patience = 10
        lookahead_k = 10


    # Auto-tune effective params in fast mode
    eff_beam = int(beam_width)
    eff_pp = int(per_parent)
    eff_k = int(k_cands)
    eff_depth = int(base_depth)
    if fast_mode:
        n_edges = len(edges_df)
        scale = 1.0 if n_edges <= 150_000 else max(0.35, min(1.0, 150_000 / max(1, n_edges)))
        eff_beam = max(4, int(round(beam_width * scale)))
        eff_pp   = max(2, int(round(per_parent * (0.6 + 0.4*scale))))
        eff_k    = max(20, int(round(k_cands * (0.6 + 0.4*scale))))
        if not (require_all_stops or require_all_edges):
            eff_depth = max(300, int(round(base_depth * (0.6 + 0.4*scale))))
        st.caption(f"Fast mode: using beam={eff_beam}, per_parent={eff_pp}, k={eff_k}, depth={eff_depth} (edges={n_edges:,})")

    verbose =         st.checkbox("Show detailed search logs", value=False, key="plan_verbose")

    st.markdown("### Execution")
    use_single_thread =         st.checkbox("Single-thread mode (no parallel workers)", value=False, key="single_thread_mode", help="Run search sequentially in the main thread. Slower, but useful for debugging.")

    # ---- Build a signature of current planner settings for caching ----
    def current_planner_params_signature() -> dict:
        trim_enabled_val = False
        try:
            if 'plan_trim_enable' in st.session_state:
                trim_enabled_val = bool(st.session_state['plan_trim_enable'])
        except Exception: # Handle cases where session_state might not be ready
            pass

        sig = {
            "optimize_time": bool(optimize_time),
            "win_from_h": int(win_from_h) if optimize_time else None,
            "win_to_h": int(win_to_h) if optimize_time else None,
            "step_min": int(step_min) if optimize_time else None,
            "start_time_s": int(start_time_s) if not optimize_time else None,
            "horizon_min": int(horizon_min),
            "require_all_stops": bool(require_all_stops),
            "require_all_edges": bool(require_all_edges),
            "undirected_edges": bool(undirected_edges),
            "prefer_new": bool(prefer_new),
            "zero_wait_same_route": bool(zero_wait_same_route),
            "strict_elapsed": bool(strict_elapsed),
            "enable_beam": bool(enable_beam),
            "beam_width": eff_beam,
            "per_parent": eff_pp,
            "base_depth": eff_depth,
            "k_cands": eff_k,
            "greedy_restarts": int(greedy_restarts),
            "explore": float(explore),
            "mode_filter": tuple(int(x) for x in sel_types_plan) if sel_types_plan else (),
            "group_plan": bool(group_plan),
            "trim_enabled": trim_enabled_val,
            "edges_sig": edges_signature(edges_df),
            "from_loc": bool(from_loc),
            "loc_candidates": tuple(loc_candidates) if loc_candidates else (),
            "manual_starts": tuple(manual_starts),
            "service_date": str(service_date),
            "aggressive_backtrack": bool(aggressive_backtrack),
            "dom_slack_s": int(dom_slack_s),
            "widen_factor": float(widen_factor),
            "stagnation_patience": int(stagnation_patience),
            "lookahead_k": int(lookahead_k),
            "fast_mode": bool(fast_mode),
            "stop_on_first_full": bool(stop_on_first_full),
            "smart_starts": bool(smart_starts),
            "accelerator": ACCELERATOR,
        }
        return sig

    params_now = current_planner_params_signature()

    # Prepare targets (counts) and exact required sets — based on CURRENT planning graph (group-aware)
    usable_ids = set(edges_df["from_stop"]).union(set(edges_df["to_stop"]))
    targ_stops = len(usable_ids) if require_all_stops else None

    if require_all_edges:
        if undirected_edges:
            pairs = set(tuple(sorted(x)) for x in edges_df[["from_stop","to_stop"]].itertuples(index=False, name=None))
        else:
            pairs = set(tuple(x) for x in edges_df[["from_stop","to_stop"]].itertuples(index=False, name=None))
        targ_edges = len(pairs)
    else:
        targ_edges = None

    # Exact required sets (post-filtering/grouping/trim)
    req_stop_ids, req_edge_keys = required_sets_from_edges(edges_df, undirected_edges)

    # Smart start recommendations (now rigorous + bigger pool)
    suggested_endpoints_text = ""
    recommended_starts: List[str] = []
    if smart_starts and not from_loc and not use_manual:
        n_nodes = len(usable_ids)
        max_recs = min(120, max(20, n_nodes))  # bigger pool
        rec_starts, rec_ends, why = _recommend_starts_and_ends(
            edges_df, usable_ids, objective_nodes=obj_is_stops,
            require_all_stops=require_all_stops, require_all_edges=require_all_edges,
            undirected_edges=undirected_edges, t_list=t_list, horizon_s=horizon_s, max_recs=int(max_recs)
        )
        recommended_starts = [s for s in rec_starts if s in usable_ids]
        suggested_endpoints_text = why
        if recommended_starts:
            show_n = min(8, len(recommended_starts))
            rec_names = [f"{stop_name_lookup.get(sid, sid)}" for sid in recommended_starts[:show_n]]
            st.info(f"Smart starts (top {show_n} of {len(recommended_starts)}): {', '.join(rec_names)}")
            if why:
                st.caption(why)

    startable_ids_all = sorted(list(usable_ids))
    if from_loc and loc_candidates:
        starts_list = loc_candidates
    elif use_manual:
        starts_list = [s for s in manual_starts if s in startable_ids]
    elif smart_starts and recommended_starts:
        # Avoid slider edge case when too few
        nrec = len(recommended_starts)
        if nrec <= 2:
            max_try = nrec
            st.caption(f"Trying {nrec} smart start{'s' if nrec!=1 else ''}.")
        else:
            max_try =         st.slider(
                "Max smart starts to try",
                min_value=2,
                max_value=min(120, nrec),
                value=min(20, nrec),
                step=1,
                key="plan_max_smart"
            )
        starts_list = recommended_starts[:max_try]
    else:
        starts_list = startable_ids_all

    # ---- cached departure index per edge set ----
    idx = build_departure_index(edges_df, edges_signature(edges_df))

    # Override t_list with exact scheduled departures within the window (if requested)
    try:
        if optimize_time and scan_exact_departures:
            base = int(win_from_h) * 3600
            end  = int(win_to_h) * 3600
            exact_times = set()
            for sid in starts_list:
                arr = idx.dep_arrays.get(sid)
                if arr is None or len(arr) == 0:
                    continue
                i0 = int(np.searchsorted(arr, np.int32(base), side="left"))
                i1 = int(np.searchsorted(arr, np.int32(end),  side="right"))
                # Clip safely
                i0 = max(0, min(i0, len(arr)))
                i1 = max(0, min(i1, len(arr)))
                for v in arr[i0:i1].tolist():
                    try:
                        exact_times.add(int(v))
                    except Exception:
                        pass
            t_list = sorted(exact_times)
            st.caption(f"Scanning {len(t_list):,} exact departures within the selected window.")
    except Exception as e:
        st.caption(f"Exact-departure scan unavailable: {e}")

    # Helper to test if a stop has a departure after t (within window unless ignore_deadline)
    def has_departure_after(sid: str, t: int, horizon_s: int, ignore_deadline: bool, idx: NextDepartureIndex) -> bool:
        g = idx.by_stop.get(sid)
        if g is None or g.empty:
            return False
        arr = idx.dep_arrays[sid]
        j = int(np.searchsorted(arr, np.int32(t), side="left"))
        if j >= len(arr):
            return False
        return True if ignore_deadline else (int(arr[j]) <= t + horizon_s)

    # Compute button
    run_plan = st.button("Compute Itinerary", key="plan_compute")

    # Execution backend controls

    with st.expander("Execution options", expanded=False):

        colB1, colB2, colB3 = st.columns([1,1,1])
        use_joblib = colB1.toggle(
            "Use joblib (batch parallel)",
            value=st.session_state.get("plan_use_joblib", False),
            key="plan_use_joblib",
            help="Threads = live metrics, Joblib = batch updates"
        )
        colB1.caption("Threads = live metrics, Joblib = batch updates")
        max_workers = colB2.slider(
            "Max workers", 1, 32,
            st.session_state.get("plan_threads", max(2, min(16, (os.cpu_count() or 4)))),
            1, key="plan_threads"
        )
        remember_choice = colB3.checkbox(
            "Remember my choice",
            value=st.session_state.get("plan_remember", True),
            key="plan_remember"
        )
    if "plan_cache" not in st.session_state:
        st.session_state["plan_cache"] = None

    # fallback-safe status panel
    def _mk_status():
        if hasattr(st, "status"):
            s = st.status("Planning…", expanded=False)
            return {
                "write": lambda msg: s.write(msg),
                "update": lambda label=None, state=None: s.update(label=label, state=state),
                "complete": lambda : s.update(label="Search complete.", state="complete"),
            }
        else:
            ph = st.empty()
            return {
                "write": lambda msg: ph.info(msg),
                "update": lambda label=None, state=None: ph.info(label or ""),
                "complete": lambda : ph.success("Search complete."),
            }

    # Helper: execute planning and write cache







    def compute_and_cache_results():
        """Parallelize the network search and cache a rich result catalog, with snappy progress + ETA + joblib iteration status."""
        import os, contextlib

        # UI: show progress immediately (avoid initial overhead blank time)
        progress = st.progress(0.0)
        prog_text = st.empty()
        try:
            progress.progress(0.001)
            prog_text.text("Preparing… building fast index")
        except Exception:
            pass

        # Build fast departure index early
        idx = NextDepartureIndex(edges_df)

        # --- Metrics strip (archived style) ---
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        m_runs = mcol1.empty(); m_elapsed = mcol2.empty(); m_best = mcol3.empty(); m_cov = mcol4.empty(); m_eta = mcol5.empty()
        best_elapsed_seen = None
        best_cov_stops = 0; best_cov_edges = 0
        found_full_any = False
        done_so_far = 0
        t_wall0_local = perf_counter()

        # --- Work unit ---
        def run_single_plan(t, sid):
            if not has_departure_after(sid, int(t), horizon_s, ignore_deadline, idx):
                return None
            if enable_beam:
                plan = plan_beam_elapsed_first(
                    edges_df, sid, int(t), horizon_s, undirected_edges, int(eff_k),
                    int(eff_beam), int(eff_pp), int(eff_depth), bool(prefer_new),
                    ignore_deadline, targ_stops, targ_edges, bool(zero_wait_same_route),
                    bool(strict_elapsed),
                    accelerator=ACCELERATOR,
                    idx=idx, log_steps=False,
                    aggressive=bool(aggressive_backtrack),
                    dom_slack_s=int(dom_slack_s),
                    widen_factor=float(widen_factor),
                    stagnation_patience=int(stagnation_patience),
                    lookahead_k=int(lookahead_k),
                    stop_on_first_full=bool(stop_on_first_full),
                )
            else:
                plan = plan_elapsed_first_greedy(
                    edges_df, sid, int(t), horizon_s, undirected_edges, int(eff_k),
                    int(greedy_restarts), float(explore), bool(prefer_new),
                    ignore_deadline, targ_stops, targ_edges, bool(zero_wait_same_route),
                    bool(strict_elapsed),
                    accelerator=ACCELERATOR,
                    idx=idx, log_steps=False,
                    progress_cb=None, job_id=None
                )
            if plan is None:
                return None
            elapsed, inv, wait = plan_elapsed_components(plan, int(t))
            full_hit = True
            if require_all_stops:
                full_hit &= (plan.seen_stop_ids is not None) and req_stop_ids.issubset(plan.seen_stop_ids)
            if require_all_edges:
                full_hit &= (plan.seen_edge_keys is not None) and req_edge_keys.issubset(plan.seen_edge_keys)
            return {"plan": plan, "start_time_s": int(t), "start_stop": str(sid), "elapsed": int(elapsed), "wait": int(wait), "full_hit": bool(full_hit)}

        # --- Jobs ---
        all_jobs_args = [(t, sid) for t in t_list for sid in starts_list]
        total_jobs = len(all_jobs_args)
        if total_jobs == 0:
            st.warning("No valid start times or locations to search."); return

        # Set a small baseline so the bar appears instantly
        try:
            progress.progress(0.02)
            prog_text.text(f"Queued {total_jobs} search tasks…")
        except Exception:
            pass

        S = _mk_status()
        S["write"](f"Preparing {total_jobs} searches …")  # concise

        # --- Helpers: ETA + progress wrapper ---
        def fmt_eta():
            if done_so_far <= 0:
                return "—"
            spent = max(0.001, perf_counter() - t_wall0_local)
            per = spent / float(done_so_far)
            remain = max(0.0, (total_jobs - done_so_far) * per)
            try:
                return fmt_hms(int(remain))
            except Exception:
                return f"~{int(remain)}s"

        @contextlib.contextmanager
        def streamlit_pbar(total):
            class _P:
                def __init__(self, total): self.total=total; self.n=0
                def update(self, k=1): self.n += k
                def close(self): pass
            p = _P(total); 
            try: yield p
            finally: p.close()

        def update_metrics_ui(current_best):
            # Coverage targets for display (if any)
            try:
                st_targ = targ_stops if (targ_stops is not None) else len(req_stop_ids)
            except Exception:
                st_targ = 0
            try:
                ed_targ = targ_edges if (targ_edges is not None) else len(req_edge_keys)
            except Exception:
                ed_targ = 0

            m_runs.metric("Runs", f"{done_so_far}/{total_jobs}")
            m_elapsed.metric("Wall", fmt_hms(int(perf_counter() - t_wall0_local)))
            m_best.metric("Best elapsed", fmt_hms(int(current_best)) if current_best is not None else "—", delta=("✅ full" if found_full_any else None))
            m_cov.metric("Coverage", f"stops {best_cov_stops}/{st_targ} | edges {best_cov_edges}/{ed_targ}" if st_targ or ed_targ else f"stops {best_cov_stops} | edges {best_cov_edges}")
            m_eta.metric("ETA", fmt_eta())

        # Storage
        results = []
        best_elapsed_s_local = None

        # Choose backend per UI
        use_joblib = bool(st.session_state.get("plan_use_joblib", False))
        max_workers = int(st.session_state.get("plan_threads", 0)) or max(2, min(16, (os.cpu_count() or 4)))

        # --- joblib path with batching (shows iteration batches) ---
        if use_joblib:
            try:
                from joblib import Parallel, delayed
                n_jobs = max(1, min(max_workers, total_jobs))
                # Determine batches to provide progress heartbeats
                num_batches = min(20, max(1, total_jobs // max(1, n_jobs)))
                batch_size = max(1, (total_jobs + num_batches - 1) // num_batches)
                with streamlit_pbar(total=total_jobs) as pbar:
                    for bi in range(0, total_jobs, batch_size):
                        batch = all_jobs_args[bi:bi+batch_size]
                        b_idx = bi // batch_size + 1
                        b_tot = (total_jobs + batch_size - 1) // batch_size
                        try:
                            prog_text.text(f"Joblib: running batch {b_idx}/{b_tot} ({len(batch)} tasks)…")
                        except Exception:
                            pass
                        batch_results = Parallel(n_jobs=n_jobs)(delayed(run_single_plan)(t, sid) for (t, sid) in batch)
                        for res in batch_results:
                            done_so_far += 1
                            if res:
                                results.append(res)
                                p = res["plan"]
                                best_cov_stops = max(best_cov_stops, int(getattr(p, "unique_stops", 0)))
                                best_cov_edges = max(best_cov_edges, int(getattr(p, "unique_edges", 0)))
                                if (best_elapsed_s_local is None) or (res["elapsed"] < best_elapsed_s_local):
                                    best_elapsed_s_local = res["elapsed"]; best_elapsed_seen = best_elapsed_s_local
                                if res.get("full_hit"): found_full_any = True
                            try:
                                pbar.update(1)
                                progress.progress(min(1.0, pbar.n/total_jobs))
                                update_metrics_ui(best_elapsed_s_local)
                            except Exception:
                                pass
            except Exception:
                use_joblib = False

        # --- Threaded path (live updates per task) ---
        if not use_joblib:
            try:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                n_workers = max(1, min(max_workers, total_jobs))
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futs = [ex.submit(run_single_plan, t, sid) for t, sid in all_jobs_args]
                    for fut in as_completed(futs):
                        res = None
                        try: res = fut.result()
                        except Exception: res = None
                        done_so_far += 1
                        if res:
                            results.append(res)
                            p = res["plan"]
                            best_cov_stops = max(best_cov_stops, int(getattr(p, "unique_stops", 0)))
                            best_cov_edges = max(best_cov_edges, int(getattr(p, "unique_edges", 0)))
                            if (best_elapsed_s_local is None) or (res["elapsed"] < best_elapsed_s_local):
                                best_elapsed_s_local = res["elapsed"]; best_elapsed_seen = best_elapsed_s_local
                            if res.get("full_hit"): found_full_any = True
                        try:
                            progress.progress(min(1.0, done_so_far/max(1,total_jobs)))
                            update_metrics_ui(best_elapsed_s_local)
                            prog_text.text(f"Threads: processed {done_so_far}/{total_jobs} tasks…")
                        except Exception: pass
            except Exception:
                pass

        # --- Summarize results ---
        S["update"](label=f"Processing {len(results)} completed plans…", state="running")
        catalog = {}; winners = []
        for res in results:
            plan = res["plan"]
            catalog[(res["start_time_s"], res["start_stop"])] = plan
            winners.append((res["elapsed"], -int(getattr(plan, "unique_stops", 0)), -int(getattr(plan, "unique_edges", 0)), res["wait"], res["start_time_s"], res["start_stop"], plan, bool(res["full_hit"])))
        winners.sort(key=lambda w: (not w[7], w[0], w[1], w[2], w[3], w[4], w[5]))
        progress.progress(1.0); S["complete"]()

        if not winners:
            st.error("No feasible itineraries found for these settings."); st.session_state["plan_cache"] = None; return

        try:
            stops_snap = stops_view[["stop_id","stop_name","lat","lon"]].copy()
        except Exception:
            stops_snap = None

        st.session_state["plan_cache"] = {"params": params_now, "catalog": catalog, "winners": winners, "targ_stops": targ_stops, "targ_edges": targ_edges, "map_style": map_style, "stops_snapshot": stops_snap, "req_stop_ids": req_stop_ids, "req_edge_keys": req_edge_keys, "require_all_stops": require_all_stops, "require_all_edges": require_all_edges}
        return True

    if run_plan:
        compute_and_cache_results()

    # Compare current params to cached
    cache = st.session_state.get("plan_cache", None)
    same_params = cache is not None and cache.get("params") == params_now

    if (cache is None) or (not same_params):
        if cache is None:
            st.info("Set your options, then click **Compute Itinerary**.")
        else:
            st.warning("Settings changed since the last run. Click **Compute Itinerary** to recompute results.")
    else:
        catalog = cache["catalog"]; winners = cache["winners"]
        targ_stops = cache["targ_stops"]; targ_edges = cache["targ_edges"]
        stops_snap = cache["stops_snapshot"]
        if not winners:
            st.error("No feasible itineraries for these settings."); st.stop()
        feasible = [w for w in winners if w[-1]]
        best_pool = feasible if feasible else winners
        best_elapsed, _, _, _, best_time_s, best_sid, best_plan, full_flag = best_pool[0]
        if full_flag: st.success("✅ Previewing a **FULL COVERAGE** plan.")
        else: st.info("◻ Previewing best partial coverage plan.")
        st.subheader("Alternatives (global)")
        name_lut = stops_snap.set_index("stop_id")["stop_name"].to_dict()
        show_max =         st.slider("Max rows to show", 50, 2000, 400, 50, key="lb_max_rows")
        lb_rows=[]
        for elapsed, negS, negE, wait, t, sid, p, full in winners[:show_max]:
            lb_rows.append({"start_time_s": int(t), "start_time": seconds_to_hhmm(int(t)), "start_stop": sid, "start_name": name_lut.get(sid, sid), "elapsed_s": int(elapsed), "elapsed": fmt_hms(int(elapsed)), "coverage_stops": int(p.unique_stops), "coverage_edges": int(p.unique_edges), "waiting_s": int(wait), "waiting": fmt_hms(int(wait)), "in_vehicle_s": int(p.total_travel_s), "in_vehicle": fmt_hms(int(p.total_travel_s)), "full_coverage": bool(full)})
        lb_df = pd.DataFrame(lb_rows).sort_values(["full_coverage","elapsed_s","coverage_stops","coverage_edges","waiting_s"], ascending=[False,True,False,False,True])
        st.dataframe(lb_df[["start_time","start_stop","start_name","elapsed","coverage_stops","coverage_edges","waiting","in_vehicle","full_coverage"]], use_container_width=True, hide_index=True)
        tested_times = sorted({int(r[4]) for r in winners})
        default_time_idx = tested_times.index(int(best_time_s)) if int(best_time_s) in tested_times else 0
        show_time_s =         st.selectbox("View alternatives for start time", options=tested_times, index=default_time_idx, format_func=lambda t: seconds_to_hhmm(int(t)), key="plan_alt_time")
        rows=[]
        for sid in sorted({sid for (t,sid) in catalog.keys() if int(t)==int(show_time_s)}):
            p = catalog[(int(show_time_s), sid)]
            el, inv, wa = plan_elapsed_components(p, int(show_time_s))
            rows.append({"start_stop":sid,"start_name":name_lut.get(sid,sid),"elapsed_s":int(el),"elapsed":fmt_hms(int(el)),"coverage_stops":int(p.unique_stops),"coverage_edges":int(p.unique_edges),"waiting_s":int(wa),"waiting":fmt_hms(int(wa)),"in_vehicle_s":int(inv),"in_vehicle":fmt_hms(int(inv))})
        if rows:
            by_time_df = pd.DataFrame(rows).sort_values(["elapsed_s","coverage_stops","coverage_edges","waiting_s"], ascending=[True,False,False,True])
            st.dataframe(by_time_df.drop(columns=["elapsed_s","waiting_s","in_vehicle_s"]), use_container_width=True, hide_index=True)
        st.subheader("Preview")
        pick_opts = [("best", int(best_time_s), str(best_sid))]; seen_pairs=set()
        for _, r in lb_df.iterrows():
            pair=(int(r["start_time_s"]), str(r["start_stop"])) 
            if pair in seen_pairs: continue
            seen_pairs.add(pair); pick_opts.append(("alt", int(r["start_time_s"]), str(r["start_stop"])))
        def pick_label(kind, t, sid):
            tag = "Best" if kind=="best" else "Alt"; nm = name_lut.get(sid, sid); p = catalog.get((int(t), str(sid)))
            if p is None: return f"{tag} — {seconds_to_hhmm(int(t))} @ {nm} (unavailable)"
            el, _, _ = plan_elapsed_components(p, int(t)); return f"{tag} — {seconds_to_hhmm(int(t))} @ {nm} | elapsed {fmt_hms(int(el))} | stops {p.unique_stops} | edges {p.unique_edges}"
        pick_idx =         st.selectbox("Choose plan", options=list(range(len(pick_opts))), format_func=lambda i: pick_label(*pick_opts[i]), key="plan_pick_idx")
        _, preview_time_s, preview_sid = pick_opts[pick_idx]; preview_plan = catalog.get((int(preview_time_s), str(preview_sid)))
        if (not preview_plan) or (not getattr(preview_plan, "legs", None)): st.info("No legs in this plan.")
        else:
            coord = stops_snap.set_index("stop_id")[ ["lat","lon"] ].to_dict(orient="index")
            total_dist=0.0; georows=[]; leg_rows=[]; stop_name_lut = stops_snap.set_index("stop_id")["stop_name"].to_dict()
            try: route_short_lut = gtfs.routes.set_index("route_id")["route_short_name"].astype(str).to_dict()
            except Exception: route_short_lut = {{}}
            for l in preview_plan.legs:
                a,b = coord.get(l.from_stop), coord.get(l.to_stop)
                if (not a) or (not b): continue
                d = haversine_m(a["lat"],a["lon"],b["lat"],b["lon"]); total_dist+=d
                leg_rows.append({"Depart": seconds_to_hhmm(l.dep_s),"From": f"{stop_name_lut.get(l.from_stop, l.from_stop)} ({l.from_stop})","Trip": l.trip_id,"Route": route_short_lut.get(l.route_id, l.route_id),"Arrive": seconds_to_hhmm(l.arr_s),"To": f"{stop_name_lut.get(l.to_stop, l.to_stop)} ({l.to_stop})","Segment (min)": int(max(0,(l.arr_s-l.dep_s)//60)),"Mode": ROUTE_TYPE_LABELS.get(int(l.route_type) if l.route_type is not None else -1, str(l.route_type))})
                georows.append({"from_lat":float(a["lat"]), "from_lon":float(a["lon"]),"to_lat":float(b["lat"]), "to_lon":float(b["lon"]),"tooltip": f"{stop_name_lut.get(l.from_stop,l.from_stop)} → {stop_name_lut.get(l.to_stop,l.to_stop)}\n{seconds_to_hhmm(l.dep_s)} → {seconds_to_hhmm(l.arr_s)}"})
            elapsed, inv, wait = plan_elapsed_components(preview_plan, int(preview_time_s))
            visited_stops = preview_plan.seen_stop_ids or set(); visited_edges = preview_plan.seen_edge_keys or set()
            ok_stops = True if not cache["require_all_stops"] else cache["req_stop_ids"].issubset(visited_stops)
            ok_edges = True if not cache["require_all_edges"] else cache["req_edge_keys"].issubset(visited_edges)
            full_ok = ok_stops and ok_edges
            need_stops = len(cache["req_stop_ids"]); need_edges = len(cache["req_edge_keys"])
            have_stops = len(visited_stops); have_edges = len(visited_edges)
            badge = "✅ Full coverage satisfied" if full_ok else "◻ Partial coverage"
            st.success(f"{badge}  |  Start **{seconds_to_hhmm(int(preview_time_s))}** @ **{name_lut.get(preview_sid, preview_sid)}**  |  Stops: **{have_stops}/{need_stops}**  Edges: **{have_edges}/{need_edges}**  |  Elapsed: **{fmt_hms(int(elapsed))}** (Ride {fmt_hms(int(inv))} + Wait {fmt_hms(int(wait))})  |  Distance: **{fmt_km(total_dist)}" )
            st.dataframe(pd.DataFrame(leg_rows), use_container_width=True, hide_index=True)
            try:
                geo_df = pd.DataFrame(georows)
                st.pydeck_chart(pdk.Deck(map_style=_resolve_map_style(map_style), initial_view_state=pdk.ViewState(latitude=float(stops_snap["lat"].median()) if not stops_snap.empty else 0.0, longitude=float(stops_snap["lon"].median()) if not stops_snap.empty else 0.0, zoom=10), layers=[pdk.Layer("ScatterplotLayer", data=stops_snap.rename(columns={"stop_name":"tooltip"}), get_position="[lon,lat]", get_radius=6, pickable=True), pdk.Layer("LineLayer", data=geo_df, get_source_position="[from_lon,from_lat]", get_target_position="[to_lon,to_lat]", get_width=2, pickable=True)], tooltip={"text":"{tooltip}"} ))
            except Exception as e:
                st.caption(f"Map preview unavailable: {e}")
            # ---- Export options ----
            try:
                import pandas as _pd
                import json as _json
                _legs_df = _pd.DataFrame(leg_rows)
                _csv = _legs_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Export legs (CSV)", _csv, file_name="plan_legs.csv", mime="text/csv", key="export_csv")

                # GeoJSON FeatureCollection of leg segments
                _feat = []
                coord_idx = stops_snap.set_index("stop_id")[["lat","lon"]].to_dict(orient="index")
                for i, l in enumerate(preview_plan.legs):
                    a = coord_idx.get(l.from_stop); b = coord_idx.get(l.to_stop)
                    if not a or not b: continue
                    _feat.append({
                        "type":"Feature",
                        "properties": {
                            "index": i,
                            "from_stop": l.from_stop, "to_stop": l.to_stop,
                            "trip_id": l.trip_id, "route_id": l.route_id, "route_type": l.route_type,
                            "dep_s": int(l.dep_s), "arr_s": int(l.arr_s)
                        },
                        "geometry": {
                            "type":"LineString",
                            "coordinates": [[float(a["lon"]), float(a["lat"])],[float(b["lon"]), float(b["lat"])]]
                        }
                    })
                _gj = {"type":"FeatureCollection","features":_feat}
                st.download_button("⬇️ Export legs (GeoJSON)", _json.dumps(_gj).encode("utf-8"),
                                   file_name="plan_legs.geojson", mime="application/geo+json", key="export_geojson")
                st.download_button("⬇️ Export legs (JSON)", _json.dumps({"legs": [l.__dict__ for l in preview_plan.legs]}).encode("utf-8"),
                                   file_name="plan_legs.json", mime="application/json", key="export_json")
            except Exception as _ex:
                st.caption(f"Export unavailable: {_ex}")

            # ---- Playback (chronological) ----
            with st.expander("▶ Playback"):
                _n = len(preview_plan.legs)

                # Playback state (separate from slider key)
                if "_playing" not in st.session_state:
                    st.session_state["_playing"] = False
                if "play_cursor" not in st.session_state:
                    st.session_state["play_cursor"] = 0

                play_col1, play_col2 = st.columns([3,1])
                play_idx = play_col1.slider("Step", 0, _n, st.session_state.get("play_cursor", 0),
                                            key="play_idx", help="Advance through legs in order")
                col_p1, col_p2 = play_col2.columns(2)
                if col_p1.button("▶", key="play_go"):
                    st.session_state["_playing"] = True
                if col_p2.button("⏸", key="play_pause"):
                    st.session_state["_playing"] = False

                # Auto-advance using play_cursor; do not touch the slider's key
                effective_idx = play_idx
                if st.session_state["_playing"]:
                    st.session_state["play_cursor"] = min(st.session_state.get("play_cursor", 0) + 1, _n)
                    effective_idx = st.session_state["play_cursor"]
                    time.sleep(0.5)
                    st.experimental_rerun()
                else:
                    st.session_state["play_cursor"] = play_idx
                    effective_idx = play_idx

                # Draw up to effective_idx
                try:
                    _rows = []
                    for i, l in enumerate(preview_plan.legs[:effective_idx]):
                        a = coord_idx.get(l.from_stop); b = coord_idx.get(l.to_stop)
                        if not a or not b: continue
                        _rows.append({"from_lat": float(a["lat"]), "from_lon": float(a["lon"]),
                                      "to_lat": float(b["lat"]), "to_lon": float(b["lon"])})
                    _gdf = _pd.DataFrame(_rows)
                    st.pydeck_chart(pdk.Deck(
                        map_style=_resolve_map_style(map_style),
                        initial_view_state=pdk.ViewState(latitude=stops_snap["lat"].median() if not stops_snap.empty else 0.0,
                                                         longitude=stops_snap["lon"].median() if not stops_snap.empty else 0.0,
                                                         zoom=11),
                        layers=[
                            pdk.Layer("LineLayer", data=_gdf, get_source_position="[from_lon,from_lat]",
                                      get_target_position="[to_lon,to_lat]", get_width=3, pickable=False),
                        ]
                    ))
                except Exception as _e2:
                    st.caption(f"Playback preview unavailable: {_e2}")
