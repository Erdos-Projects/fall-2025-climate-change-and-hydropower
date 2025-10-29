#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import math
import pandas as pd
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
import urllib.request

# ----------------------------------------------------------------------
# 1. Load parameter descriptions 
# ----------------------------------------------------------------------
params = {}

def add_params(fname):
    try:
        with open(fname, 'r', encoding='utf-8') as f:
            obj = json.loads(f.read())
            for feat in obj['features']:
                pid = int(feat['id'])
                desc = feat['properties']['parameter_description']
                params[pid] = desc
    except FileNotFoundError:
        print(f"Warning: {fname} not found – using generic names.")
    except Exception as e:
        print(f"Error loading {fname}: {e}")

add_params('params.json')
add_params('params_2.json')

# ----------------------------------------------------------------------
# 2. USGS API helpers
# ----------------------------------------------------------------------
API_BASE = 'https://api.waterdata.usgs.gov/ogcapi/v0/collections/'

def get_json(url):
    time.sleep(0.21)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return {}

def find_closest_feature(lat, lon, features):
    if not features:
        return None
    best = features[0]
    min_dist = float('inf')
    for f in features:
        coords = f['geometry']['coordinates']
        if len(coords) < 2:
            continue
        d = (lat - coords[1]) ** 2 + (lon - coords[0]) ** 2
        if d < min_dist:
            min_dist = d
            best = f
    return best

# ----------------------------------------------------------------------
# 3. Find the *closest* station that reports a given code
# ----------------------------------------------------------------------
def nearest_station_for_code(lat, lon, code, max_dist_km=100):
    """
    Returns (station_id, coordinates) for the nearest station that
    currently reports `code`.  Returns (None, None) if none found.
    """
    max_deg = max_dist_km / 111.0
    for eps in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        if eps > max_deg:
            break
        bbox = [lon - eps, lat - eps, lon + eps, lat + eps]
        query = {
            'f': 'json',
            'lang': 'en-US',
            'bbox': ','.join(map(str, bbox)),
            'limit': 1000,
            'properties': 'id,monitoring_location_id,parameter_code',
            'parameter_code': code
        }
        url = f"{API_BASE}latest-continuous/items?{urlencode(query)}"
        resp = get_json(url)
        if resp.get('numberReturned', 0):
            feat = find_closest_feature(lat, lon, resp['features'])
            station = feat['properties']['monitoring_location_id']
            if not station.startswith('USGS-'):
                station = 'USGS-' + station
            print(f"   → {code} → {station}  (~{eps*111:.1f} km)")
            return station, feat['geometry']['coordinates']
    return None, None

# ----------------------------------------------------------------------
# 4. Historical data (daily mean) for ONE station + ONE code
# ----------------------------------------------------------------------
def get_historical_values(station, code, years=30, statistic_id='00003'):
    if not station:
        return pd.DataFrame()

    if not station.startswith('USGS-'):
        station = 'USGS-' + station

    end   = datetime.now(timezone.utc).date()
    start = end - timedelta(days=years * 365.25)
    date_range = f"{start:%Y-%m-%d}/{end:%Y-%m-%d}"

    base = (
        f'{API_BASE}daily/items?f=json&lang=en-US&limit=1000&skipGeometry=true'
        f'&monitoring_location_id={station}&parameter_code={code}'
        f'&datetime={date_range}'
    )
    if statistic_id:
        base += f'&statistic_id={statistic_id}'

    records = []
    offset = 0
    while True:
        url = f'{base}&offset={offset}'
        data = get_json(url)

        if 'code' in data:
            print(f"   API error for {code}: {data.get('title','')}")
            break

        feats = data.get('features', [])
        if not feats:
            break

        for f in feats:
            p = f['properties']
            records.append({
                'datetime'   : p.get('time'),
                'value'      : p.get('value'),
                'units'      : p.get('unit_of_measure'),
                'approval'   : p.get('approval_status') or p.get('approvals_status'),
                'statistic'  : p.get('statistic_id')
            })

        fetched = len(feats)
        print(f"   [{code}] offset {offset:,} → +{fetched:,} records")
        if fetched < 1000:
            break
        offset += 1000

    df = pd.DataFrame(records)
    if not df.empty:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.sort_values('datetime').reset_index(drop=True)
    return df

# ----------------------------------------------------------------------
# 5. Nice column names
# ----------------------------------------------------------------------
def nice_column(code, unit):
    name_map = {
        '00060': 'discharge_cfs',
        '00010': 'temp_C',
        '00065': 'stage_ft',
        '00045': 'precip_in',
        '00095': 'cond_uScm',
        '00530': 'sediment_mg_per_L',
        '80154': 'sediment_inst_mg_per_L',
        '80155': 'sediment_load_tons_per_day',
        '00076': 'turbidity_NTU',
        '00197': 'evaporation_in',
        '00035': 'wind_mph',
        '72019': 'reservoir_elev_ft'
    }
    base = name_map.get(code, f'param_{code}')
    return f'{base}_{unit.replace(" ", "_").replace("/", "_per_")}'

# ----------------------------------------------------------------------
# 6. MAIN – ONE STATION PER PARAMETER
# ----------------------------------------------------------------------
if __name__ == '__main__':
    LAT, LON = 44.0521, -123.0897          # change as needed
    YEARS    = 30
    STAT_ID  = '00003'                     # daily mean

    # 13 hydropower-relevant parameters
    TARGET_PARAMS = [
        '00060',  # Discharge (cfs)
        '00010',  # Water temperature
        '00065',  # Gage height (ft)
        '00045',  # Precipitation (in)
        '00095',  # Specific conductance
        '00530',  # Suspended sediment concentration
        '80154',  # Suspended sediment (instant)
        '80155',  # Suspended sediment discharge
        '00076',  # Turbidity
        '00197',  # Evaporation
        '00035',  # Wind speed
        '72019',  # Reservoir elevation
    ]

    print(f"\n=== Searching nearest station for each of {len(TARGET_PARAMS)} parameters (≤100 km) ===\n")

    master_dfs = {}

    for code in TARGET_PARAMS:
        print(f"Looking for parameter {code} ({params.get(int(code), 'unknown')}) …")
        station, coords = nearest_station_for_code(LAT, LON, code, max_dist_km=100)

        if not station:
            print(f"   No station within 100 km for {code}\n")
            continue

        # ---- fetch historical data ---------------------------------
        if code == '00065':                     # stage may not have daily mean
            df = get_historical_values(station, code, years=YEARS, statistic_id=None)
            if not df.empty:
                if '00003' in df['statistic'].values:
                    df = df[df['statistic'] == '00003']
                else:
                    df = df[df['statistic'] == '00011']   # instantaneous
        else:
            df = get_historical_values(station, code, years=YEARS, statistic_id=STAT_ID)

        if df.empty:
            print(f"   No historical data for {code} at {station}\n")
            continue

        unit = df['units'].iloc[0] if 'units' in df else 'unknown'
        col  = nice_column(code, unit)
        df   = df[['datetime', 'value']].rename(columns={'value': col})
        df   = df.set_index('datetime')
        master_dfs[col] = df
        print(f"   Saved {len(df):,} rows for {col}\n")

    # ------------------------------------------------------------------
    # 7. Merge everything on datetime
    # ------------------------------------------------------------------
    if not master_dfs:
        raise SystemExit("No data retrieved for any parameter – nothing to save.")

    combined = pd.concat(master_dfs.values(), axis=1, join='outer')
    combined = combined.sort_index().reset_index()

    out_file = f"usgs_hydropower_multi_station_{YEARS}years.csv"
    combined.to_csv(out_file, index=False)

    print(f"\nFinished! All parameters merged → {out_file}")
    print(f"   Rows   : {len(combined):,}")
    print(f"   Columns: {list(combined.columns)}")
