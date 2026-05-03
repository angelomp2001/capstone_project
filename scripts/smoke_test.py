from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generate_sample_df import generate_sample_df
from operations import apply_operations
from text_parser import llm_parses_to_ops


def main() -> None:
    ''' A simple smoke test to verify that the core components of the app are working as expected.'''
    df = generate_sample_df(n_rows=10)

    drop_ops = llm_parses_to_ops("drop columns gender and categories", df)
    assert drop_ops == [{"op": "drop_columns", "params": {"columns": ["gender", "categories"]}}]
    dropped_df = apply_operations(df, drop_ops)
    assert "gender" not in dropped_df.columns
    assert "categories" not in dropped_df.columns

    fill_ops = llm_parses_to_ops("fill missing normal with 0", df)
    assert fill_ops == [{"op": "fillna", "params": {"column": "normal", "strategy": "constant", "value": 0}}]

    replace_ops = llm_parses_to_ops("replace med category in risk_level with 1", df)
    assert replace_ops == [{
        "op": "replace_value",
        "params": {"column": "risk_level", "old_value": "med", "new_value": 1},
    }]
    replaced_df = apply_operations(df, replace_ops)
    assert "med" not in replaced_df["risk_level"].astype(str).tolist()

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
