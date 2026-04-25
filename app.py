import streamlit as st
import pandas as pd

from generate_sample_df import generate_sample_df
from src.cleaning_operations import apply_operations
from src.llm_cleaning import parse_instruction_to_ops


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "df" not in st.session_state:
        st.session_state.df = generate_sample_df()
    if "operations" not in st.session_state:
        st.session_state.operations = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def main():
    st.set_page_config(page_title="LLM Data Cleaning POC", layout="wide")
    st.title("LLM-Driven Data Cleaning (POC)")

    init_session_state()

    # --- Data Preview --------------------------------------------------------
    st.subheader("Sample Data (first 15 rows)")
    st.dataframe(st.session_state.df.head(15))

    st.markdown("**Columns and dtypes:**")
    dtypes = {col: str(dtype) for col, dtype in st.session_state.df.dtypes.items()}
    st.json(dtypes)

    st.markdown("---")

    # --- Cleaning Chat -------------------------------------------------------
    st.subheader("Cleaning Chat")

    # Show previous messages
    for role, text in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Assistant:** {text}")

    user_input = st.text_input(
        "Describe a cleaning action (e.g., 'drop rows with any missing values', "
        "'fill missing age with median', 'drop columns gender and city')."
    )

    if st.button("Apply instruction") and user_input:
        # Log user message
        st.session_state.chat_history.append(("user", user_input))

        # Ask LLM to translate instruction into operations
        ops = parse_instruction_to_ops(user_input, st.session_state.df)

        if not ops:
            msg = (
                "I couldn't parse that into valid operations. "
                "Try being more explicit (e.g., name the column or describe the strategy)."
            )
            st.session_state.chat_history.append(("assistant", msg))
        else:
            # Apply operations
            new_df = apply_operations(st.session_state.df, ops)

            # Update state
            st.session_state.df = new_df
            st.session_state.operations.extend(ops)

            msg = (
                f"I applied {len(ops)} operation(s). "
                "The DataFrame preview has been updated."
            )
            st.session_state.chat_history.append(("assistant", msg))

    # --- Reset button --------------------------------------------------------
    if st.button("Reset data"):
        st.session_state.df = generate_sample_df()
        st.session_state.operations = []
        st.session_state.chat_history = []

    st.markdown("---")

    # --- Operations Log ------------------------------------------------------
    st.subheader("Operations Applied So Far")
    st.json(st.session_state.operations)


if __name__ == "__main__":
    main()