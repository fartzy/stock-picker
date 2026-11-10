"""Train only the small Intraday Entry helper from cached research inputs."""
import json

from stock_picker.training.intraday_entry_job import run_intraday_entry_training


if __name__ == "__main__":
    result = run_intraday_entry_training(phase=lambda value: print(value, flush=True))
    print(json.dumps({"trained_through": result["trained_through"],
                      "evaluated_through": result["evaluated_through"],
                      "feature_count": result["feature_count"],
                      "validation": result["validation"]}, indent=2))
