import streamlit as st
import pandas as pd
import logging
import os
from src.generate_sample_df import generate_sample_df
from src.llm.operations import apply_operations
from src.llm.text_parser import llm_parses_to_ops
from src.llm.llm_utils import (
    is_llm_available,
    setup_logging,
    get_log_locations,
)

# setup logging
from dotenv import load_dotenv
load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)
trace = logging.getLogger("trace")


def create_initial_df() -> pd.DataFrame:
    """Generate display-friendly starter data for the app. if there is no data in the data/raw dir"""
    if os.path.exists("data/raw/titanic.csv"):
        df = pd.read_csv("data/raw/titanic.csv")
    else:
        # wrapper for this function
        df = generate_sample_df()
        # clean df
        for col in df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns:
            df[col] = df[col].astype(str)
        # log 
        logger.info("Created initial DataFrame with shape %s", df.shape)
        trace.info("DATAFRAME INIT shape=%r preview=%r", df.shape, df.head(3).to_dict(orient="records"))
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
        trace.info("SESSION START")


def main():
    st.set_page_config(page_title="LLM Data Cleaning POC", layout="wide")
    st.title("LLM-Driven Data Cleaning (POC)") # ui component

    init_session_state()
    log_locations = get_log_locations()
    # log status
    logger.info("App rendered. Current DataFrame shape=%s", st.session_state.df.shape)
    if is_llm_available():
        st.success("Connected to LLM API. Natural-language instructions will use the LLM parser.")  # ui component
    else:
        st.error("LLM API was not detected. Please configure the LLM API to use this app.")  # ui component

    # with st.expander("Where the logs are saved"): # ui component
    #     st.code( # ui component
    #         f"App log:   {log_locations['app_log']}\n"
    #         f"Trace log: {log_locations['trace_log']}",
    #         language="text",
    #     )

    # --- Data Preview --------------------------------------------------------
    st.subheader("Sample Data (first 15 rows)") # ui component
    preview_df = st.session_state.df.head(8).copy()
    st.dataframe(preview_df, use_container_width=True) # width='stretch' ui component

    st.markdown("**Columns and dtypes:**") # ui component
    dtypes = {col: str(dtype) for col, dtype in st.session_state.df.dtypes.items()}
    st.json(dtypes) # ui component

    st.markdown("---") # ui component (divider)

    # --- Chat -------------------------------------------------------
    st.subheader("Chat") # ui component

    # Show previous messages
    for role, text in st.session_state.chat_history: # ui component
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Assistant:** {text}")

    user_input = st.text_input( # ui component
        "Describe a cleaning action (e.g., 'drop rows with any missing values', "
        "'fill missing age with median', 'drop columns gender and city')."
    )

    apply_clicked = st.button("Apply instruction") # ui component

    if apply_clicked:
        if not user_input:
            logger.warning("User clicked apply with empty input.")
            trace.info("USER INPUT empty")
            st.warning("Please enter an instruction.")
        else:
            # Log user message
            st.session_state.chat_history.append(("user", user_input))
            logger.info("User instruction received: %s", user_input)
            trace.info("USER INPUT text=%r", user_input)

            # Translate instruction into operations
            ops_json = llm_parses_to_ops(user_input, st.session_state.df)
            logger.info("Parsed operations from instruction: %s", ops_json)

            if not ops_json or ops_json[0].get("op") == "error":
                msg = (
                    ops_json[0]["params"]["message"]
                    if ops_json
                    else "I couldn't map that request to one of the supported POC operations yet."
                )
                logger.warning("Instruction could not be applied. Message=%s", msg)
                trace.info("USER INPUT FAILED text=%r message=%r", user_input, msg)
                st.session_state.chat_history.append(("assistant", msg))
                st.error(msg)
            else:
                # Apply operations
                new_df = apply_operations(st.session_state.df, ops_json)

                # Update state
                st.session_state.df = new_df
                st.session_state.operations.extend(ops_json)

                msg = f"I applied {len(ops_json)} operation(s):\n{ops_json}"
                logger.info("Instruction applied successfully. Operations=%s new_shape=%s", ops_json, new_df.shape)
                trace.info("USER INPUT SUCCESS ops=%r new_shape=%r", ops_json, new_df.shape)
                st.session_state.chat_history.append(("assistant", msg))
                st.rerun()

    # --- Reset button --------------------------------------------------------
    if st.button("Reset data"):
        st.session_state.df = create_initial_df()
        st.session_state.operations = []
        st.session_state.chat_history = []
        logger.info("User reset the app state.")
        trace.info("SESSION RESET")
        st.rerun()

    st.markdown("---")

    # --- Operations Log ------------------------------------------------------
    st.subheader("Operations Applied So Far")
    st.json(st.session_state.operations)


if __name__ == "__main__":
    main()
