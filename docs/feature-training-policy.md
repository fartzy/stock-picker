# Feature training policy

Persisted pipeline inputs are ordinary model features. They are included by
default, subject to the normal per-run selection and explicit prune list.
There is no additional gate for newly created features.

The 19 price-pattern inputs are stored in the usual feature snapshots:
11 volatility inputs, four breadth inputs, and four trajectory inputs. They
appear in those Feature Store categories with the standard selection and
prune controls.

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
