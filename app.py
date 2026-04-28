import streamlit as st
import pandas as pd

from src.generate_sample_df import generate_sample_df
from src.cleaning_operations import apply_operations
from src.llm_cleaning import parse_instruction_to_ops
from src.llm_utils import is_ollama_available


def create_initial_df() -> pd.DataFrame:
    """Generate display-friendly starter data for the app."""
    df = generate_sample_df()
    for col in df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns:
        df[col] = df[col].astype(str)
    return df


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "df" not in st.session_state:
        st.session_state.df = create_initial_df()
    if "operations" not in st.session_state:
        st.session_state.operations = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def main():
    st.set_page_config(page_title="LLM Data Cleaning POC", layout="wide")
    st.title("LLM-Driven Data Cleaning (POC)")

    init_session_state()
    if is_ollama_available():
        st.success("Connected to Ollama. Natural-language instructions will use the LLM parser.")
    else:
        st.info("Ollama was not detected. The app is running in built-in POC mode with simple instruction parsing.")

    # --- Data Preview --------------------------------------------------------
    st.subheader("Sample Data (first 15 rows)")
    preview_df = st.session_state.df.head(8).copy()
    st.dataframe(preview_df, width='stretch')

    st.markdown("**Columns and dtypes:**")
    dtypes = {col: str(dtype) for col, dtype in st.session_state.df.dtypes.items()}
    st.json(dtypes)

    st.markdown("---")

    # --- Chat -------------------------------------------------------
    st.subheader("Chat")

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

    apply_clicked = st.button("Apply instruction")

    if apply_clicked:
        if not user_input:
            st.warning("Please enter an instruction.")
        else:
            # Log user message
            st.session_state.chat_history.append(("user", user_input))

            # Translate instruction into operations
            ops = parse_instruction_to_ops(user_input, st.session_state.df)

            if not ops or ops[0].get("op") == "error":
                msg = (
                    ops[0]["params"]["message"]
                    if ops
                    else "I couldn't map that request to one of the supported POC operations yet."
                )
                st.session_state.chat_history.append(("assistant", msg))
            else:
                # Apply operations
                new_df = apply_operations(st.session_state.df, ops)

                # Update state
                st.session_state.df = new_df
                st.session_state.operations.extend(ops)

                msg = f"I applied {len(ops)} operation(s):\n{ops}"
                st.session_state.chat_history.append(("assistant", msg))

    # --- Reset button --------------------------------------------------------
    if st.button("Reset data"):
        st.session_state.df = create_initial_df()
        st.session_state.operations = []
        st.session_state.chat_history = []

    st.markdown("---")

    # --- Operations Log ------------------------------------------------------
    st.subheader("Operations Applied So Far")
    st.json(st.session_state.operations)


if __name__ == "__main__":
    main()
