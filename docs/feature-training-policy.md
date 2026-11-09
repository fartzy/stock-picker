# Feature training policy

Persisted pipeline inputs are ordinary model features. They are included by
default, subject to the normal per-run selection and explicit prune list.
There is no additional gate for newly created features.

Ordinary Yahoo daily-price history covers two years with the existing explicit
`auto_adjust=False` convention; benchmark/SPY/VIX history uses the same horizon.
Every feature family computes on that full history for warmup. Feature Store
snapshots then retain only the latest completed calendar year, for every column
together, preserving the existing training span and avoiding an accidental
doubling of SVM training rows. Missing history remains unknown for new listings
or incomplete provider coverage. The isolated five-year research cache is not
copied into production prices and does not set the ordinary training horizon.
Two calendar years do not guarantee 252 valid preceding sessions for every
earliest retained row, new listing, or history with missing bars; those values
remain unknown with explicit validity indicators.

The 19 price-pattern inputs are stored in the usual feature snapshots:
11 volatility inputs, four breadth inputs, and four trajectory inputs. They
appear in those Feature Store categories with the standard selection and
prune controls.

The 21 dip inputs are also ordinary persisted features, grouped into low
context, recovery, pressure, and support/trend families. Their setup indicators
use 1 when a condition holds, 0 when valid inputs show it does not, and missing
values when the required inputs are unavailable or invalid. Separate history
and recovery validity indicators record input availability. These inputs use
the same per-run selection and ordinary pruning as every other feature.

Production training first evaluates chronological folds, then refits Fit on
all eligible training dates through the latest close. The SVM-derived Fit
inputs remain earlier-date out-of-fold predictions during LightGBM fitting;
the companion SVM estimators are refitted on the full training-ticker history
for serving. Held-out tickers never enter either fit. Rank also trains on
the full training-ticker history.

The highest recorded importance is retained across saved models, comparison
fits, and values recorded in historical prune reasons. Pruning or removing
a feature from a later model does not erase its previous importance.
Each value is a normalized share within its source fit; the per-feature
maxima can come from different fits and do not sum to 100%.
