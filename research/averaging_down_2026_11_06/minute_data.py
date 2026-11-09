"""Shared parsing and strictly backward-looking intraday stabilization flag."""
import numpy as np
import pandas as pd


def regular_frame(doc):
    d=pd.DataFrame(doc['bars'])
    if d.empty:return d
    d.index=pd.to_datetime(d.t,unit='ms',utc=True).dt.tz_convert('America/New_York')
    d=d.sort_index();minutes=d.index.hour*60+d.index.minute
    return d[(minutes>=570)&(minutes<960)]


def stability_flags(d):
    """Close above prior 15 wall-clock minutes' highs; at least 10 prior bars."""
    times=d.index.as_unit('s').asi8.astype(float);h=d.h.to_numpy();c=d.c.to_numpy()
    flags=np.zeros(len(d),dtype=bool)
    for i in range(1,len(d)):
        start=np.searchsorted(times,times[i]-900)
        if i-start>=10 and c[i]>np.max(h[start:i]):flags[i]=True
    return flags
