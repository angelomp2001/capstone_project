from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generate_sample_df import generate_sample_df
from src.cleaning_operations import apply_operations
from src.llm_cleaning import parse_instruction_to_ops


def main() -> None:
    df = generate_sample_df(n_rows=10)

    drop_ops = parse_instruction_to_ops("drop columns gender and categories", df)
    assert drop_ops == [{"op": "drop_columns", "params": {"columns": ["gender", "categories"]}}]
    dropped_df = apply_operations(df, drop_ops)
    assert "gender" not in dropped_df.columns
    assert "categories" not in dropped_df.columns

    fill_ops = parse_instruction_to_ops("fill missing normal with 0", df)
    assert fill_ops == [{"op": "fillna", "params": {"column": "normal", "strategy": "constant", "value": 0}}]

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
