import streamlit as st
import pandas as pd
import logging

from src.generate_sample_df import generate_sample_df
from src.cleaning_operations import apply_operations
from src.llm_cleaning import parse_instruction_to_ops
from src.llm_utils import (
    is_ollama_available,
    setup_logging,
    get_log_locations,
    append_trace,
)


setup_logging()
logger = logging.getLogger(__name__)


def create_initial_df() -> pd.DataFrame:
    """Generate display-friendly starter data for the app."""
    df = generate_sample_df()
    for col in df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns:
        df[col] = df[col].astype(str)
    logger.info("Created initial DataFrame with shape %s", df.shape)
    append_trace(f"DATAFRAME INIT shape={df.shape!r} preview={df.head(3).to_dict(orient='records')!r}")
    return df


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "df" not in st.session_state:
        st.session_state.df = create_initial_df()
    if "operations" not in st.session_state:
        st.session_state.operations = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "session_log_started" not in st.session_state:
        st.session_state.session_log_started = True
        logger.info("Started new Streamlit session.")
        append_trace("SESSION START")


def main():
    st.set_page_config(page_title="LLM Data Cleaning POC", layout="wide")
    st.title("LLM-Driven Data Cleaning (POC)")

    init_session_state()
    log_locations = get_log_locations()
    logger.info("App rendered. Current DataFrame shape=%s", st.session_state.df.shape)
    if is_ollama_available():
        st.success("Connected to Ollama. Natural-language instructions will use the LLM parser.")
    else:
        st.info("Ollama was not detected. The app is running in built-in POC mode with simple instruction parsing.")

    with st.expander("Where the logs are saved"):
        st.code(
            f"App log:   {log_locations['app_log']}\n"
            f"Trace log: {log_locations['trace_log']}",
            language="text",
        )

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
            logger.warning("User clicked apply with empty input.")
            append_trace("USER INPUT empty")
            st.warning("Please enter an instruction.")
        else:
            # Log user message
            st.session_state.chat_history.append(("user", user_input))
            logger.info("User instruction received: %s", user_input)
            append_trace(f"USER INPUT text={user_input!r}")

            # Translate instruction into operations
            ops = parse_instruction_to_ops(user_input, st.session_state.df)
            logger.info("Parsed operations from instruction: %s", ops)

            if not ops or ops[0].get("op") == "error":
                msg = (
                    ops[0]["params"]["message"]
                    if ops
                    else "I couldn't map that request to one of the supported POC operations yet."
                )
                logger.warning("Instruction could not be applied. Message=%s", msg)
                append_trace(f"USER INPUT FAILED text={user_input!r} message={msg!r}")
                st.session_state.chat_history.append(("assistant", msg))
            else:
                # Apply operations
                new_df = apply_operations(st.session_state.df, ops)

                # Update state
                st.session_state.df = new_df
                st.session_state.operations.extend(ops)

                msg = f"I applied {len(ops)} operation(s):\n{ops}"
                logger.info("Instruction applied successfully. Operations=%s new_shape=%s", ops, new_df.shape)
                append_trace(f"USER INPUT SUCCESS ops={ops!r} new_shape={new_df.shape!r}")
                st.session_state.chat_history.append(("assistant", msg))

    # --- Reset button --------------------------------------------------------
    if st.button("Reset data"):
        st.session_state.df = create_initial_df()
        st.session_state.operations = []
        st.session_state.chat_history = []
        logger.info("User reset the app state.")
        append_trace("SESSION RESET")

    st.markdown("---")

    # --- Operations Log ------------------------------------------------------
    st.subheader("Operations Applied So Far")
    st.json(st.session_state.operations)


if __name__ == "__main__":
    main()
