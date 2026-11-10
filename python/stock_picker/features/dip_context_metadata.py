"""Descriptions, formulas and examples for ordinary morning dip inputs.

All formulas are already aligned to entry date t. They describe earlier
completed sessions plus the current Open where explicitly stated.
The snapshot adapter and dataset alignment produce these decision-time values.
"""
from dataclasses import dataclass

from stock_picker.features.dip_context import DIP_COLUMNS


@dataclass(frozen=True)
class DipFeatureMetadata:
    name: str
    description: str
    computation: str
    example: str

_DETAILS = {
    "dip_close_to_low_252d": (
        "Yesterday's close relative to the lowest completed low in 252 sessions.",
        "C[t-1] / min(L[t-252:t]) - 1; require 252 valid earlier bars.",
        "Yesterday closed at 51, the prior low was 50: 0.02, or 2% above it.",
    ),
    "dip_open_to_low_252d": (
        "Current opening distance above or below the prior 252-session low.",
        "Open[t] / min(L[t-252:t]) - 1; only earlier lows enter the floor.",
        "Opening at 49 with a prior low of 50 gives -0.02, a fresh breach.",
    ),
    "dip_low_age_252d": (
        "Sessions since the most recent occurrence of the prior 252-session low.",
        "Reverse the 252 earlier lows; argmin + 1 selects the most recent tied minimum.",
        "A tied minimum yesterday and ten sessions ago has age 1.",
    ),
    "dip_prior_new_low_252d": (
        "Whether yesterday made or tied the low of the prior 252-session window.",
        "1 if L[t-1] <= min(L[t-252:t-1]); unknown without 252 valid earlier bars.",
        "Yesterday's low 49 versus the preceding 251-bar low 50 gives 1.",
    ),
    "dip_open_below_low_252d": (
        "Whether the observed open breaks below the prior 252-session low.",
        "1 if Open[t] < min(L[t-252:t]); unknown if Open or earlier history is missing.",
        "An open of 49 versus a prior low of 50 gives 1; an open of 50 gives 0.",
    ),
    "dip_low_history_valid": (
        "Availability of the complete prior window needed for long-term low context.",
        "1 if all 252 earlier session bars are finite, positive and OHLC-consistent; otherwise 0.",
        "251 valid earlier bars give 0, while low-dependent measurements remain unknown.",
    ),
    "dip_loss_2d_sigma": (
        "Two-day closing return scaled by volatility measured before the selloff.",
        "(C[t-1]/C[t-3]-1) / (std20(earlier returns ending t-3, ddof=1) * sqrt(2)).",
        "A -4% two-day return with 2% daily pre-shock volatility gives about -1.414.",
    ),
    "dip_recovery_fraction": (
        "Fraction of the preceding two-day closing loss recovered by the current open.",
        "(Open[t]-C[t-1]) / (C[t-3]-C[t-1]); earlier closing loss must be positive.",
        "Close fell from 100 to 80 and opens at 82: 0.10 of the closing loss recovered.",
    ),
    "dip_recovery_valid": (
        "Whether the prior closing loss and current open define a recovery fraction.",
        "1 if earlier closes and current Open are valid and C[t-3] > C[t-1]; otherwise 0.",
        "An unchanged two-day close gives 0 and an unknown recovery fraction, not zero recovery.",
    ),
    "dip_deep_unrecovered": (
        "Exact candidate flag for a two-day selloff near a long-term low with little recovery.",
        "Both earlier close returns <= -2%; prior close 0–2% above low252; current-open recovery fraction 0–25%.",
        "Two -3% days, close 1% above the low, and 10% of the loss recovered give 1.",
    ),
    "dip_loss_change_sigma": (
        "Change in absolute daily return size, normalized by pre-selloff volatility.",
        "(abs(r[t-1])-abs(r[t-2])) / std20(earlier returns ending t-3, ddof=1).",
        "Losses shrink from 4% to 2% with 1% pre-shock volatility: -2.0.",
    ),
    "dip_range_change_1d": (
        "Yesterday's true range relative to the preceding day's true range.",
        "TR[t-1]/TR[t-2]-1; TR=max(H-L,abs(H-Cprev),abs(L-Cprev)) on earlier completed bars.",
        "True range falls from 4 to 3: -0.25, or 25% contraction.",
    ),
    "dip_volume_change_1d": (
        "Yesterday's volume change relative to the first selloff day.",
        "V[t-1]/V[t-2]-1, using earlier completed bars with positive volume.",
        "Volume falls from 2 million to 1.5 million: -0.25.",
    ),
    "dip_pressure_easing": (
        "Exact candidate flag for two down days with smaller loss, lower volume and an upper-half close.",
        "Earlier r1<0,r2<0,abs(r1)<abs(r2),V[t-1]<V[t-2],(C-L)/(H-L)[t-1]>=0.5.",
        "Loss shrinks from 4% to 2%, volume contracts, and close location is 0.7: 1.",
    ),
    "dip_gap_previous_volume_zscore": (
        "Current opening gap interacted with yesterday's abnormal selling-volume context.",
        "(Open[t]/C[t-1]-1) * volume_zscore20[t-1]; z-score uses earlier completed volume only.",
        "A +1% gap after yesterday's volume z-score of 2 gives 0.02.",
    ),
    "dip_support_break_depth_20d": (
        "Depth of yesterday's low below a support level fixed before that session.",
        "(S-L[t-1])/S, where S=min(L[t-21:t-1]) uses 20 earlier lows before yesterday.",
        "Old support is 50 and yesterday's low 49: 0.02 below support.",
    ),
    "dip_support_close_distance_20d": (
        "Yesterday's closing distance from support established before its attempted break.",
        "C[t-1]/S-1, where S=min(L[t-21:t-1]) uses earlier lows before yesterday.",
        "Closing at 51 after testing support at 50 gives +0.02.",
    ),
    "dip_support_open_distance_20d": (
        "Current opening distance from the support reference fixed before yesterday.",
        "Open[t]/S-1, where S=min(L[t-21:t-1]) uses earlier lows before yesterday.",
        "Opening at 50.5 relative to support 50 gives +0.01.",
    ),
    "dip_support_reclaim_20d": (
        "Exact candidate flag for an attempted breakdown that reclaimed support and holds at the open.",
        "Earlier L[t-1]<S and C[t-1]>=S, then Open[t]>=S; S=min(L[t-21:t-1]).",
        "Support 50, yesterday low 49 and close 51, current open 50.5: 1.",
    ),
    "dip_sma50_slope_10d": (
        "Ten-session change in the completed 50-session moving average.",
        "SMA50[t-1]/SMA50[t-11]-1, using earlier completed closes only.",
        "The 50-session average rises from 100 to 102 over ten sessions: +0.02.",
    ),
    "dip_trend_pullback": (
        "Exact candidate flag for a two-day pullback while remaining above a rising longer-term average.",
        "Earlier C[t-1]/C[t-3]-1 <= -2%, C[t-1]>SMA50[t-1], and SMA50[t-1]>SMA50[t-11].",
        "A 3% two-day decline still above a rising 50-session average gives 1.",
    ),
}

DIP_FEATURES = tuple(
    DipFeatureMetadata(name=name, description=_DETAILS[name][0], computation=_DETAILS[name][1], example=_DETAILS[name][2])
    for name in DIP_COLUMNS
)
DIP_FEATURE_METADATA = {feature.name: feature for feature in DIP_FEATURES}
