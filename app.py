from __future__ import annotations

import json
from datetime import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.data_load import load_personal_data
from src.feature_engineering import get_latest_state
from src.forecast import compare_forecasts, load_model_bundle, simulate_forecast
from src.train import train_model
from src.viz import build_forecast_figure, build_importance_figure

load_dotenv()  # Load COHERE_API_KEY from .env if present

st.set_page_config(page_title="Personal Energy Forecast Planner", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_personal_dataframe() -> tuple[pd.DataFrame, dict]:
    return load_personal_data("data/oura_personal.csv")


@st.cache_resource(show_spinner=False)
def get_model() -> dict | None:
    return load_model_bundle("models/energy_model.pkl")


def load_metadata(path: str = "models/metadata.json") -> dict:
    meta_path = Path(path)
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Time / hour helpers
# ---------------------------------------------------------------------------

def model_hour_to_time(hour_value: float) -> time:
    normalized = float(hour_value) % 24
    hour_int = int(normalized)
    minute_int = int(round((normalized - hour_int) * 60))
    if minute_int == 60:
        hour_int = (hour_int + 1) % 24
        minute_int = 0
    return time(hour=hour_int, minute=minute_int)


def model_hour_to_label(hour_value: float) -> str:
    t = model_hour_to_time(hour_value)
    return f"{t.hour:02d}:{t.minute:02d}"


def time_to_model_hour(value: time) -> float:
    hour_value = value.hour + value.minute / 60.0
    if hour_value < 12:
        hour_value += 24
    return hour_value


def circular_hour_diff(target_hour: float, baseline_hour: float) -> float:
    diff = (target_hour % 24) - (baseline_hour % 24)
    if diff > 12:
        diff -= 24
    if diff < -12:
        diff += 24
    return diff


def clock_duration_minutes(bed_time: time, wake_time: time) -> int:
    bed_min = bed_time.hour * 60 + bed_time.minute
    wake_min = wake_time.hour * 60 + wake_time.minute
    duration = wake_min - bed_min
    if duration <= 0:
        duration += 24 * 60
    return int(duration)


def duration_label(minutes: float) -> str:
    rounded = int(round(minutes))
    h = rounded // 60
    m = rounded % 60
    return f"{h}h {m:02d}m"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    if "scenario_params" not in st.session_state:
        st.session_state.scenario_params = {
            "name": "Current Scenario",
            "bedtime_target_hour": None,
            "wake_target_hour": None,
            "bedtime_shift_hours": 0.0,
            "sleep_delta_min": 0,
            "training_load_delta": 0,
            "caffeine_cutoff_hour": 15,
            "alcohol": False,
            "late_meal": False,
        }

    if "experiment_log" not in st.session_state:
        log_path = Path("data/experiment_log.csv")
        if log_path.exists():
            log_df = pd.read_csv(log_path)
        else:
            log_df = pd.DataFrame(
                columns=["created_at", "scenario_name", "tags", "day1_delta", "notes"]
            )
        st.session_state.experiment_log = log_df

    # State for LLM features
    if "llm_parsed_params" not in st.session_state:
        st.session_state.llm_parsed_params = None
    if "optimizer_result" not in st.session_state:
        st.session_state.optimizer_result = None


def persist_experiment_log() -> None:
    path = Path("data/experiment_log.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    st.session_state.experiment_log.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Apply LLM scenario to main session state
# ---------------------------------------------------------------------------

def apply_llm_scenario_to_state(
    params: dict,
    baseline_bedtime_hour: float,
    baseline_sleep_duration_min: float,
    name: str = "AI Scenario",
) -> None:
    """Convert LLM-extracted shift values to absolute target hours and apply."""
    shift = float(params.get("bedtime_shift_hours", 0.0))
    sleep_delta = int(params.get("sleep_delta_min", 0))
    baseline_wake_hour = baseline_bedtime_hour + baseline_sleep_duration_min / 60.0

    new_bedtime_hour = baseline_bedtime_hour + shift
    new_wake_hour = baseline_wake_hour + shift + sleep_delta / 60.0

    st.session_state.scenario_params = {
        "name": name,
        "bedtime_target_hour": new_bedtime_hour,
        "wake_target_hour": new_wake_hour % 24,
        "bedtime_shift_hours": shift,
        "sleep_delta_min": sleep_delta,
        "training_load_delta": int(params.get("training_load_delta", 0)),
        "caffeine_cutoff_hour": int(params.get("caffeine_cutoff_hour", 15)),
        "alcohol": bool(params.get("alcohol", False)),
        "late_meal": bool(params.get("late_meal", False)),
    }


# ---------------------------------------------------------------------------
# Training block
# ---------------------------------------------------------------------------

def training_block() -> dict | None:
    model_bundle = get_model()
    if model_bundle is not None:
        return model_bundle

    st.warning("No trained model found. Please run training once.")
    if st.button("Train model now", type="primary"):
        with st.spinner("Training in progress..."):
            train_model(
                personal_csv="data/oura_personal.csv",
                synthetic_csv="data/synthetic.csv",
                model_out="models/energy_model.pkl",
                metadata_out="models/metadata.json",
                n_synth_days=730,
                random_state=42,
            )
        st.cache_resource.clear()
        st.success("Training completed. Reloading app.")
        st.rerun()

    return None


# ---------------------------------------------------------------------------
# UI helpers for LLM parameter display
# ---------------------------------------------------------------------------

def _render_param_grid(params: dict) -> None:
    """Render a compact grid of scenario parameters extracted by the LLM."""
    c1, c2, c3 = st.columns(3)
    shift = params.get("bedtime_shift_hours", 0.0)
    c1.metric(
        "Bedtime shift",
        f"{shift:+.1f}h",
        delta=f"{'earlier' if shift < 0 else 'later' if shift > 0 else 'unchanged'}",
        delta_color="normal" if shift <= 0 else "inverse",
    )
    sleep_delta = params.get("sleep_delta_min", 0)
    c2.metric(
        "Sleep change",
        f"{sleep_delta:+d} min",
        delta=f"{abs(sleep_delta) // 60}h {abs(sleep_delta) % 60}m",
        delta_color="normal" if sleep_delta >= 0 else "inverse",
    )
    train_delta = params.get("training_load_delta", 0)
    c3.metric(
        "Training load",
        f"{train_delta:+d}",
        delta_color="off",
    )

    c4, c5, c6 = st.columns(3)
    c4.metric("Caffeine cutoff", f"{params.get('caffeine_cutoff_hour', 15):02d}:00")
    c5.metric("Alcohol", "Yes" if params.get("alcohol") else "No")
    c6.metric("Late meal", "Yes" if params.get("late_meal") else "No")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def app() -> None:
    init_session_state()

    st.title("Personal Energy Forecast and Planning")
    st.caption(
        "3-day energy forecast based on Oura daily data. "
        "Compare baseline vs scenario including uncertainty bands."
    )

    personal_df, load_info = load_personal_dataframe()
    model_bundle = training_block()
    if model_bundle is None:
        st.stop()

    metadata = load_metadata()

    min_date = personal_df["date"].min().date()
    max_date = personal_df["date"].max().date()

    # -----------------------------------------------------------------------
    # Sidebar controls
    # -----------------------------------------------------------------------
    st.sidebar.header("Controls")

    selected_range = st.sidebar.date_input(
        "Data range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_range, tuple):
        start_date, end_date = selected_range
    else:
        start_date, end_date = min_date, selected_range

    reference_date = st.sidebar.date_input(
        "Forecast starts from",
        value=end_date,
        min_value=start_date,
        max_value=end_date,
    )

    show_baseline = st.sidebar.toggle("Show baseline", value=True)
    show_scenario = st.sidebar.toggle("Show scenario", value=True)

    filtered = personal_df[
        (personal_df["date"].dt.date >= start_date)
        & (personal_df["date"].dt.date <= end_date)
    ].copy()
    if filtered.empty:
        filtered = personal_df.copy()

    state = get_latest_state(filtered, reference_date=pd.Timestamp(reference_date))
    baseline_bedtime_hour = float(state.get("bedtime_start_hour", 23.0))
    baseline_sleep_duration_min = float(state.get("total_sleep_duration_min", 420.0))
    baseline_wake_hour = baseline_bedtime_hour + baseline_sleep_duration_min / 60.0

    st.sidebar.markdown("---")
    st.sidebar.subheader("Scenario Builder")
    st.sidebar.caption(
        "Baseline: "
        f"bedtime {model_hour_to_label(baseline_bedtime_hour)}, "
        f"wake {model_hour_to_label(baseline_wake_hour)}, "
        f"sleep {duration_label(baseline_sleep_duration_min)}"
    )

    current_params = st.session_state.scenario_params
    default_target_hour = current_params.get("bedtime_target_hour")
    if default_target_hour is None:
        default_target_hour = baseline_bedtime_hour + float(
            current_params.get("bedtime_shift_hours", 0.0)
        )
    default_wake_hour = current_params.get("wake_target_hour")
    if default_wake_hour is None:
        default_wake_hour = baseline_wake_hour

    with st.sidebar.form("scenario_form"):
        scenario_name = st.text_input(
            "Scenario name", value=current_params.get("name", "Current Scenario")
        )
        target_bedtime = st.time_input(
            "Target bedtime",
            value=model_hour_to_time(float(default_target_hour)),
            step=900,
        )
        target_wakeup = st.time_input(
            "Target wake-up time",
            value=model_hour_to_time(float(default_wake_hour)),
            step=900,
        )
        target_sleep_duration_min = clock_duration_minutes(target_bedtime, target_wakeup)
        preview_sleep_delta = int(round(target_sleep_duration_min - baseline_sleep_duration_min))
        st.caption(
            f"Target sleep: {duration_label(target_sleep_duration_min)} "
            f"({preview_sleep_delta:+d} min)"
        )
        training_load_delta = st.slider(
            "Training load delta",
            min_value=-40,
            max_value=40,
            value=int(current_params.get("training_load_delta", 0)),
            step=1,
        )
        caffeine_cutoff_hour = st.slider(
            "Caffeine cutoff hour",
            min_value=12,
            max_value=22,
            value=int(current_params.get("caffeine_cutoff_hour", 15)),
            step=1,
        )
        col_a, col_b = st.columns(2)
        alcohol = col_a.checkbox("Alcohol", value=bool(current_params.get("alcohol", False)))
        late_meal = col_b.checkbox("Late meal", value=bool(current_params.get("late_meal", False)))
        apply_clicked = st.form_submit_button("Apply scenario")

    if apply_clicked:
        bedtime_target_hour = time_to_model_hour(target_bedtime)
        wake_target_hour = target_wakeup.hour + target_wakeup.minute / 60.0
        sleep_delta_min = int(
            min(max(int(round(target_sleep_duration_min - baseline_sleep_duration_min)), -180), 180)
        )
        bedtime_shift_hours = min(
            max(circular_hour_diff(bedtime_target_hour, baseline_bedtime_hour), -4.0), 4.0
        )
        st.session_state.scenario_params = {
            "name": scenario_name or "Current Scenario",
            "bedtime_target_hour": bedtime_target_hour,
            "wake_target_hour": wake_target_hour,
            "bedtime_shift_hours": bedtime_shift_hours,
            "sleep_delta_min": sleep_delta_min,
            "training_load_delta": training_load_delta,
            "caffeine_cutoff_hour": caffeine_cutoff_hour,
            "alcohol": alcohol,
            "late_meal": late_meal,
        }

    # -----------------------------------------------------------------------
    # Compute forecasts
    # -----------------------------------------------------------------------
    scenario_for_run = dict(st.session_state.scenario_params)
    target_hour_for_run = scenario_for_run.get("bedtime_target_hour")
    if target_hour_for_run is not None:
        dynamic_shift = circular_hour_diff(float(target_hour_for_run), baseline_bedtime_hour)
        scenario_for_run["bedtime_shift_hours"] = min(max(dynamic_shift, -4.0), 4.0)
    wake_hour_for_run = scenario_for_run.get("wake_target_hour")
    if target_hour_for_run is not None and wake_hour_for_run is not None:
        recomputed_sleep_duration = clock_duration_minutes(
            model_hour_to_time(float(target_hour_for_run)),
            model_hour_to_time(float(wake_hour_for_run)),
        )
        scenario_for_run["sleep_delta_min"] = min(
            max(int(round(recomputed_sleep_duration - baseline_sleep_duration_min)), -180), 180
        )

    baseline_forecast = simulate_forecast(
        model_bundle, base_state=state, scenario=None,
        horizon_days=3, n_samples=1200, random_state=42,
    )
    scenario_forecast = simulate_forecast(
        model_bundle, base_state=state, scenario=scenario_for_run,
        horizon_days=3, n_samples=1200, random_state=123,
    )
    comparison = compare_forecasts(baseline_forecast, scenario_forecast)

    # -----------------------------------------------------------------------
    # Tabs
    # -----------------------------------------------------------------------
    tab_forecast, tab_log, tab_smart, tab_optimizer, tab_drivers = st.tabs(
        ["Forecast", "Experiment Log", "Smart Scenario", "AI Optimizer", "Drivers"]
    )

    # ---- Forecast tab (unchanged from Assignment 1) ----
    with tab_forecast:
        fig = build_forecast_figure(
            baseline=baseline_forecast,
            scenario=scenario_forecast,
            show_baseline=show_baseline,
            show_scenario=show_scenario,
        )
        st.plotly_chart(fig, use_container_width=True)

        if not comparison.empty:
            day1 = comparison.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Day 1 Delta", f"{day1['delta_median']:+.1f}")
            c2.metric("Day 1 Baseline", f"{day1['median_baseline']:.1f}")
            c3.metric("Day 1 Scenario", f"{day1['median_scenario']:.1f}")
            c4.metric("Risk (Day 1)", f"{scenario_forecast.iloc[0]['risk']}")

        display_df = comparison[[
            "day", "date", "median_baseline", "median_scenario",
            "delta_median", "p10_baseline", "p10_scenario",
            "p90_baseline", "p90_scenario", "risk_scenario",
        ]].copy().rename(columns={
            "median_baseline": "baseline_median",
            "median_scenario": "scenario_median",
            "p10_baseline": "baseline_p10",
            "p10_scenario": "scenario_p10",
            "p90_baseline": "baseline_p90",
            "p90_scenario": "scenario_p90",
            "risk_scenario": "risk",
        })
        st.dataframe(display_df, use_container_width=True)

        with st.expander("Energy Score Formula and Model Assumptions"):
            st.markdown("""
**Energy Formula**
- `stress_component = clamp(100 * stress_high_min / (stress_high_min + recovery_min), 0, 100)`
- `energy_score = 0.5*readiness_score + 0.3*sleep_score + 0.2*clamp(100 - stress_component)`

**Forecast Logic**
- The model predicts `t+1` autoregressively over 3 days.
- Uncertainty uses Monte Carlo residual sampling with p10/p50/p90 bands.
            """)

    # ---- Drivers tab (unchanged from Assignment 1) ----
    with tab_drivers:
        st.subheader("Forecast Drivers")
        importance = model_bundle.get("feature_importance", {})
        imp_fig = build_importance_figure(importance, top_n=12)
        st.plotly_chart(imp_fig, use_container_width=True)

        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        if sorted_items:
            st.markdown("Top drivers in the current model:")
            for feature, score in sorted_items:
                st.write(f"- `{feature}`: {score:.4f}")

        if metadata.get("metrics"):
            metrics = metadata["metrics"]
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{metrics['mae']:.2f}")
            c2.metric("RMSE", f"{metrics['rmse']:.2f}")
            c3.metric("R2", f"{metrics['r2']:.3f}")

        with st.expander("Data and Mapping Assumptions"):
            if load_info.get("assumptions"):
                for line in load_info["assumptions"]:
                    st.write(f"- {line}")

    # ---- Experiment Log tab (unchanged from Assignment 1) ----
    with tab_log:
        st.subheader("Experiment Log")
        st.caption("Document your own experiments, maintain tags, and save the log as CSV.")

        with st.form("log_form"):
            log_date = st.date_input("Date", value=pd.Timestamp.today().date())
            log_tags = st.text_input(
                "Tags (comma-separated)",
                value=",".join(
                    t for t in [
                        "alcohol" if st.session_state.scenario_params.get("alcohol") else "",
                        "late_meal" if st.session_state.scenario_params.get("late_meal") else "",
                    ] if t
                ),
            )
            log_notes = st.text_input("Notes", value="")
            day1_delta = float(comparison.iloc[0]["delta_median"]) if not comparison.empty else 0.0
            add_log = st.form_submit_button("Add to log")

        if add_log:
            new_row = {
                "created_at": str(log_date),
                "scenario_name": st.session_state.scenario_params.get("name", "Scenario"),
                "tags": log_tags,
                "day1_delta": round(day1_delta, 3),
                "notes": log_notes,
            }
            st.session_state.experiment_log = pd.concat(
                [st.session_state.experiment_log, pd.DataFrame([new_row])],
                ignore_index=True,
            )
            st.success("Entry added.")

        edited_df = st.data_editor(
            st.session_state.experiment_log,
            use_container_width=True,
            num_rows="dynamic",
            key="experiment_editor",
        )
        st.session_state.experiment_log = edited_df

        if st.button("Save experiment log as CSV"):
            persist_experiment_log()
            st.success("Saved to data/experiment_log.csv")

        # ----------------------------------------------------------------
        # RAG: Ask questions about your experiment history (Feature C)
        # ----------------------------------------------------------------
        st.markdown("---")
        st.subheader("Ask About Your Experiments")
        st.caption(
            "Ask any natural language question about your experiment history. "
            "The AI finds the most relevant entries using semantic search and "
            "answers based strictly on your personal data."
        )


        n_entries = len(st.session_state.experiment_log)
        if n_entries < 2:
            st.warning(
                f"You have {n_entries} log {'entry' if n_entries == 1 else 'entries'}. "
                "Add at least 2 entries above before querying your history."
            )
        else:
            example_questions = [
                "When did going to bed earlier give me the biggest energy boost?",
                "What effect did alcohol have on my energy across all experiments?",
                "Which scenario produced the best Day 1 result?",
                "Are there any patterns between training load and my energy the next day?",
            ]
            st.markdown("**Example questions:**")
            for q in example_questions:
                st.markdown(f"- *{q}*")

            rag_question = st.text_input(
                "Your question",
                placeholder="e.g. What happened to my energy when I had alcohol?",
                key="rag_question_input",
            )

            if st.button(
                "Ask AI", type="primary",
                disabled=not rag_question.strip(),
                key="rag_ask_btn",
            ):
                from src.llm import answer_from_log
                try:
                    with st.spinner("Searching your experiment history..."):
                        rag_result = answer_from_log(
                            question=rag_question,
                            log_df=st.session_state.experiment_log.copy(),
                            top_k=5,
                        )
                    st.session_state["rag_result"] = rag_result
                except ImportError as e:
                    st.error(f"Missing dependency: {e}")
                except ValueError as e:
                    st.error(f"Configuration error: {e}")
                except Exception as e:
                    st.error(f"RAG query failed: {e}")

            if st.session_state.get("rag_result"):
                rag = st.session_state["rag_result"]
                st.markdown("**Answer:**")
                st.markdown(rag["answer"])

                sources = rag.get("sources", [])
                if sources:
                    with st.expander(f"Retrieved entries used as context ({len(sources)})"):
                        for i, src in enumerate(sources):
                            st.markdown(f"**Entry {i + 1}**")
                            st.code(src["text"], language=None)

    # -----------------------------------------------------------------------
    # Tab: Smart Scenario (Feature A)
    # -----------------------------------------------------------------------
    with tab_smart:
        st.subheader("Smart Scenario Builder")
        st.caption(
            "Describe your upcoming situation in plain English and the AI will automatically "
            "fill in the scenario parameters and show you how it affects your energy forecast."
        )


        example_prompts = [
            "I'm flying to New York tomorrow, will sleep about 5 hours, and have a work dinner with drinks.",
            "I have a triathlon on Saturday so I'll do a hard training session tomorrow, skipping my evening coffee.",
            "Quiet weekend - planning to sleep in, no alcohol, going to bed early at 10pm.",
        ]
        st.markdown("**Example descriptions:**")
        for ex in example_prompts:
            st.markdown(f"- *{ex}*")

        user_description = st.text_area(
            "Your situation",
            placeholder="e.g. I'm flying overnight tomorrow, sleeping 5 hours, and having drinks at the company dinner.",
            height=100,
        )

        col_parse, col_clear = st.columns([1, 4])
        parse_clicked = col_parse.button("Parse with AI", type="primary", disabled=not user_description.strip())
        if col_clear.button("Clear"):
            st.session_state.llm_parsed_params = None
            st.rerun()

        if parse_clicked and user_description.strip():
            from src.llm import parse_scenario_from_text
            try:
                with st.spinner("Calling LLM to extract parameters..."):
                    parsed = parse_scenario_from_text(
                        user_text=user_description,
                        baseline_state=state.to_dict(),
                    )
                st.session_state.llm_parsed_params = parsed
            except ImportError as e:
                st.error(f"Missing dependency: {e}")
            except ValueError as e:
                st.error(f"Configuration error: {e}")
            except Exception as e:
                st.error(f"LLM call failed: {e}")

        parsed_params = st.session_state.llm_parsed_params
        if parsed_params:
            st.success("Parameters extracted successfully.")

            st.markdown("**AI Reasoning:**")
            st.markdown(f"> {parsed_params.get('reasoning', '')}")

            st.markdown("**Extracted parameters:**")
            _render_param_grid(parsed_params)

            # Show what this does to the forecast
            llm_scenario = {
                "bedtime_shift_hours": parsed_params["bedtime_shift_hours"],
                "sleep_delta_min": parsed_params["sleep_delta_min"],
                "training_load_delta": parsed_params["training_load_delta"],
                "caffeine_cutoff_hour": parsed_params["caffeine_cutoff_hour"],
                "alcohol": parsed_params["alcohol"],
                "late_meal": parsed_params["late_meal"],
            }
            with st.spinner("Running forecast preview..."):
                llm_forecast = simulate_forecast(
                    model_bundle,
                    base_state=state,
                    scenario=llm_scenario,
                    horizon_days=3,
                    n_samples=800,
                    random_state=77,
                )
                llm_comparison = compare_forecasts(baseline_forecast, llm_forecast)

            st.markdown("**Forecast preview for this scenario:**")
            if not llm_comparison.empty:
                fc1, fc2, fc3 = st.columns(3)
                for i, col in enumerate([fc1, fc2, fc3]):
                    if i < len(llm_comparison):
                        row = llm_comparison.iloc[i]
                        col.metric(
                            f"Day {row['day']} ({row['date'].strftime('%a')})",
                            f"{row['median_scenario']:.1f}",
                            delta=f"{row['delta_median']:+.1f} vs baseline",
                            delta_color="normal" if row["delta_median"] >= 0 else "inverse",
                        )

            if st.button("Apply this scenario to main forecast", type="primary"):
                apply_llm_scenario_to_state(
                    parsed_params,
                    baseline_bedtime_hour,
                    baseline_sleep_duration_min,
                    name="AI: " + user_description[:40],
                )
                st.success("Scenario applied. Switch to the Forecast tab to see the results.")
                st.rerun()

    # -----------------------------------------------------------------------
    # Tab: AI Optimizer (Feature B)
    # -----------------------------------------------------------------------
    with tab_optimizer:
        st.subheader("AI Energy Optimizer")
        st.caption(
            "Tell the AI what you want to achieve and it will explore different combinations "
            "of sleep, training, and caffeine adjustments to find what works best for your goal."
        )


        example_goals = [
            "Maximize my average energy over the next 3 days.",
            "I have an important presentation on Day 2 - optimize for peak energy that day.",
            "I need to recover from a hard week of training while keeping Day 3 energy high.",
        ]
        st.markdown("**Example goals:**")
        for g in example_goals:
            st.markdown(f"- *{g}*")

        goal_input = st.text_input(
            "Your energy goal",
            placeholder="e.g. Maximize my energy on Day 2 - I have a big presentation.",
        )
        constraint_input = st.text_input(
            "Any fixed constraints? (optional)",
            placeholder="e.g. I must go to the gym tomorrow, I can't avoid the work dinner.",
        )

        full_goal = goal_input.strip()
        if constraint_input.strip():
            full_goal += f" Constraints: {constraint_input.strip()}"

        col_run, col_clear2 = st.columns([1, 4])
        run_clicked = col_run.button(
            "Run AI Optimizer", type="primary", disabled=not goal_input.strip()
        )
        if col_clear2.button("Clear results"):
            st.session_state.optimizer_result = None
            st.rerun()

        if run_clicked and full_goal:
            from src.llm import optimize_scenario_with_agent

            iteration_log: list[tuple[int, dict, dict]] = []

            def progress_callback(iteration: int, params: dict, result: dict) -> None:
                iteration_log.append((iteration, params, result))

            try:
                with st.status("AI Optimizer running...", expanded=True) as status:
                    status.write("Starting agentic optimization loop...")
                    opt_result = optimize_scenario_with_agent(
                        goal=full_goal,
                        model_bundle=model_bundle,
                        base_state=state,
                        max_iterations=6,
                        on_tool_call=progress_callback,
                    )
                    for it, params, res in iteration_log:
                        status.write(
                            f"Iteration {it}: avg energy = **{res['average_energy']:.1f}** "
                            f"(D1={res['day1_energy']:.1f}, D2={res.get('day2_energy', '?'):.1f}, "
                            f"D3={res.get('day3_energy', '?'):.1f})"
                        )
                    status.update(
                        label=f"Done - {opt_result['iterations']} scenarios explored.",
                        state="complete",
                    )
                st.session_state.optimizer_result = opt_result
            except ImportError as e:
                st.error(f"Missing dependency: {e}")
            except ValueError as e:
                st.error(f"Configuration error: {e}")
            except Exception as e:
                st.error(f"Optimizer failed: {e}")

        opt_result = st.session_state.optimizer_result
        if opt_result:
            st.markdown("---")
            best_avg = opt_result.get("best_avg_energy", 0.0)
            n_iter = opt_result.get("iterations", 0)
            st.success(f"Optimization complete - {n_iter} scenarios explored. Best average energy: **{best_avg:.1f}/100**")

            # Final recommendation text
            final_text = opt_result.get("final_recommendation", "")
            if final_text.strip():
                with st.expander("AI Recommendation (full text)", expanded=True):
                    st.markdown(final_text)

            # Best scenario parameters
            best_scenario = opt_result.get("best_scenario")
            if best_scenario:
                st.markdown("**Best scenario parameters found:**")
                _render_param_grid(best_scenario)

                # Forecast for best scenario
                with st.spinner("Running forecast for best scenario..."):
                    best_forecast = simulate_forecast(
                        model_bundle,
                        base_state=state,
                        scenario=best_scenario,
                        horizon_days=3,
                        n_samples=800,
                        random_state=99,
                    )
                    best_comparison = compare_forecasts(baseline_forecast, best_forecast)

                st.markdown("**Projected energy with best scenario:**")
                if not best_comparison.empty:
                    bc1, bc2, bc3 = st.columns(3)
                    for i, col in enumerate([bc1, bc2, bc3]):
                        if i < len(best_comparison):
                            row = best_comparison.iloc[i]
                            col.metric(
                                f"Day {row['day']} ({row['date'].strftime('%a')})",
                                f"{row['median_scenario']:.1f}",
                                delta=f"{row['delta_median']:+.1f} vs baseline",
                                delta_color="normal" if row["delta_median"] >= 0 else "inverse",
                            )

                if st.button("Apply best scenario to main forecast", type="primary"):
                    apply_llm_scenario_to_state(
                        best_scenario,
                        baseline_bedtime_hour,
                        baseline_sleep_duration_min,
                        name="AI Optimized",
                    )
                    st.success("Applied. Switch to the Forecast tab to see the full view.")
                    st.rerun()

            # Optimization journey
            call_history = opt_result.get("call_history", [])
            if call_history:
                with st.expander(f"Optimization Journey ({len(call_history)} tool calls)"):
                    for record in call_history:
                        it = record["iteration"]
                        p = record["params"]
                        r = record["result"]
                        avg = r.get("average_energy", 0.0)
                        is_best = best_scenario and all(
                            abs(float(p.get(k, 0)) - float(best_scenario.get(k, 0))) < 0.01
                            for k in ["bedtime_shift_hours", "sleep_delta_min", "training_load_delta"]
                        )
                        label = f"**Iteration {it}** - avg energy {avg:.1f}" + (" ✅ best" if is_best else "")
                        st.markdown(label)
                        col_p, col_r = st.columns(2)
                        with col_p:
                            st.markdown("*Parameters tried:*")
                            st.json(p)
                        with col_r:
                            st.markdown("*Forecast result:*")
                            days = r.get("forecast_by_day", [])
                            if days:
                                st.dataframe(pd.DataFrame(days), use_container_width=True)
                        st.markdown("---")

    # -----------------------------------------------------------------------
    # Footer metrics
    # -----------------------------------------------------------------------
    footer_left, footer_mid, footer_right = st.columns(3)
    footer_left.metric("Personal data days", len(personal_df))
    footer_mid.metric("Scenario", st.session_state.scenario_params.get("name", "Current Scenario"))
    footer_right.metric("Demo data", "Yes" if bool(personal_df["is_demo"].any()) else "No")


if __name__ == "__main__":
    app()
