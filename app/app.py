
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = RESULTS_DIR / "figures"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Solar Power Intelligence",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD MODEL + RESULTS
# =========================================================

@st.cache_resource
def load_model():
    theta = np.load(RESULTS_DIR / "theta_B_normal.npy")
    scaling = np.load(RESULTS_DIR / "set_B_scaling.npz")
    means = scaling["means"]
    stds = scaling["stds"]
    return theta, means, stds


@st.cache_data
def load_results():
    rmse = pd.read_csv(RESULTS_DIR / "rmse_results.csv")
    theta_results = pd.read_csv(RESULTS_DIR / "theta_results.csv")
    predictions = pd.read_csv(
        RESULTS_DIR / "test_predictions.csv",
        parse_dates=["datetime"]
    )
    return rmse, theta_results, predictions


theta, means, stds = load_model()
rmse_df, theta_df, predictions_df = load_results()


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(circle at 85% 5%, rgba(255,174,0,0.10), transparent 28%),
            radial-gradient(circle at 5% 35%, rgba(0,170,255,0.08), transparent 30%),
            #07111f;
        color: #f5f7fa;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background: #06101b;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .hero {
        padding: 34px 38px;
        border-radius: 26px;
        margin-bottom: 25px;
        background:
            linear-gradient(135deg,
                rgba(255,177,0,0.20),
                rgba(0,145,255,0.10),
                rgba(255,255,255,0.025));
        border: 1px solid rgba(255,255,255,0.11);
        box-shadow: 0 18px 60px rgba(0,0,0,0.28);
    }

    .hero-title {
        font-size: 44px;
        font-weight: 850;
        letter-spacing: -1.5px;
        margin: 0;
    }

    .hero-subtitle {
        margin-top: 9px;
        color: #aebdcc;
        font-size: 17px;
        line-height: 1.6;
        max-width: 850px;
    }

    .badge {
        display: inline-block;
        padding: 7px 13px;
        border-radius: 999px;
        background: rgba(255,181,0,0.14);
        border: 1px solid rgba(255,181,0,0.28);
        color: #ffd36a;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .card {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 19px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 10px 35px rgba(0,0,0,0.16);
    }

    .card-title {
        font-size: 20px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .card-text {
        color: #9eafbf;
        font-size: 14px;
        line-height: 1.6;
    }

    .kpi {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 17px;
        padding: 20px;
        min-height: 105px;
    }

    .kpi-label {
        color: #91a3b4;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        margin-top: 7px;
    }

    .prediction {
        text-align: center;
        padding: 38px 25px;
        border-radius: 24px;
        background:
            linear-gradient(135deg,
                rgba(255,169,0,0.22),
                rgba(255,211,82,0.07));
        border: 1px solid rgba(255,190,50,0.35);
        box-shadow: 0 15px 50px rgba(255,170,0,0.08);
    }

    .prediction-label {
        color: #d5dee7;
        font-size: 15px;
    }

    .prediction-value {
        font-size: 55px;
        font-weight: 850;
        line-height: 1.1;
        margin: 10px 0;
    }

    .prediction-unit {
        color: #aebdcc;
        font-size: 16px;
    }

    .feature {
        text-align: center;
        padding: 17px 10px;
        border-radius: 15px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
    }

    .feature-icon {
        font-size: 26px;
    }

    .feature-name {
        color: #b9c6d2;
        font-size: 13px;
        margin-top: 7px;
    }

    .footer {
        text-align: center;
        color: #718394;
        font-size: 12px;
        margin-top: 35px;
        padding-top: 20px;
        border-top: 1px solid rgba(255,255,255,0.07);
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
        padding: 12px;
        border-radius: 14px;
    }

    .stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 13px;
        font-weight: 750;
        font-size: 16px;
        border: none;
        background: linear-gradient(90deg, #ff9f1c, #ffc247);
        color: #101820;
    }

    .stButton > button:hover {
        box-shadow: 0 10px 30px rgba(255,170,0,0.22);
        transform: translateY(-1px);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center;padding:12px 5px 22px;">
            <div style="font-size:58px;">☀️</div>
            <div style="font-size:25px;font-weight:800;">Solar AI</div>
            <div style="color:#8093a6;font-size:13px;">
                Power Intelligence Dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Navigation")

    st.markdown(
        """
        <div style="
            padding:12px;
            border-radius:11px;
            background:rgba(255,255,255,0.05);
            margin-bottom:7px;">
            🏠 <b>Dashboard</b>
        </div>

        <div style="
            padding:12px;
            border-radius:11px;
            background:rgba(255,255,255,0.05);
            margin-bottom:7px;">
            ⚡ <b>Live Prediction</b>
        </div>

        <div style="
            padding:12px;
            border-radius:11px;
            background:rgba(255,255,255,0.05);
            margin-bottom:7px;">
            📊 <b>Results & Metrics</b>
        </div>

        <div style="
            padding:12px;
            border-radius:11px;
            background:rgba(255,255,255,0.05);">
            📈 <b>Figures</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Model")

    st.caption("Linear Regression")
    st.caption("Feature Set: Set B")
    st.caption("Solver: Normal Equation")
    st.caption("5 normalized features")

    st.divider()

    st.markdown(
        """
        <div style="color:#748799;font-size:12px;line-height:1.6;">
        Assignment 01<br>
        Machine Learning / AML<br>
        Solar AC Power Prediction
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="badge">MACHINE LEARNING • SET B • NORMAL EQUATION</div>
        <div class="hero-title">☀️ Solar Power Intelligence</div>
        <div class="hero-subtitle">
            A professional dashboard for exploring solar generation results
            and predicting hourly AC power from radiation, temperature,
            cloud cover and time features.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# KPI SUMMARY
# =========================================================

set_b_normal = rmse_df[
    (rmse_df["Solver"] == "Normal equation") &
    (rmse_df["Features"] == "Set B")
]

set_a_normal = rmse_df[
    (rmse_df["Solver"] == "Normal equation") &
    (rmse_df["Features"] == "Set A")
]

set_b_rmse = float(set_b_normal["All hours RMSE (kW)"].iloc[0])
set_b_day_rmse = float(set_b_normal["Daytime RMSE (kW)"].iloc[0])
set_a_rmse = float(set_a_normal["All hours RMSE (kW)"].iloc[0])

peak_actual = float(predictions_df["actual_ac_power"].max())
test_rows = len(predictions_df)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">Set B RMSE</div>
            <div class="kpi-value">{set_b_rmse:,.1f} kW</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">Daytime RMSE</div>
            <div class="kpi-value">{set_b_day_rmse:,.1f} kW</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k3:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">Observed Peak</div>
            <div class="kpi-value">{peak_actual:,.0f} kW</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k4:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">Test Samples</div>
            <div class="kpi-value">{test_rows:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# TABS
# =========================================================

tab_dashboard, tab_predict, tab_results, tab_figures = st.tabs(
    [
        "🏠 Dashboard",
        "⚡ Live Prediction",
        "📊 Results & Metrics",
        "📈 Figures"
    ]
)


# =========================================================
# DASHBOARD TAB
# =========================================================

with tab_dashboard:

    c1, c2 = st.columns([1.15, 0.85], gap="large")

    with c1:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">📈 Actual vs Set B Prediction</div>
                <div class="card-text">
                    Test-week comparison between measured AC power and
                    the saved Set B Normal Equation predictions.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        chart_df = predictions_df[
            ["datetime", "actual_ac_power", "set_B_normal"]
        ].copy()

        chart_df = chart_df.set_index("datetime")

        st.line_chart(
            chart_df,
            height=370
        )

    with c2:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">🧠 Model Snapshot</div>
                <div class="card-text">
                    The frontend uses the saved Set B Normal Equation
                    weights and the exact training means/stds stored
                    in the results folder.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric(
            "Set A Normal Equation RMSE",
            f"{set_a_rmse:,.2f} kW"
        )

        st.metric(
            "Set B Normal Equation RMSE",
            f"{set_b_rmse:,.2f} kW"
        )

        st.metric(
            "Daytime Set B RMSE",
            f"{set_b_day_rmse:,.2f} kW"
        )

        st.info(
            "Use the Live Prediction tab to enter weather conditions "
            "and generate a new AC power estimate."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <div class="card-title">🔬 Set B Features</div>
            <div class="card-text">
                The model uses five normalized features:
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    cols = st.columns(5)

    feature_items = [
        ("☀️", "Shortwave Radiation"),
        ("🌡️", "Temperature 2m"),
        ("☁️", "Cloud Cover"),
        ("〰️", "Sin(Hour)"),
        ("〰️", "Cos(Hour)")
    ]

    for col, (icon, name) in zip(cols, feature_items):
        with col:
            st.markdown(
                f"""
                <div class="feature">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-name">{name}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# LIVE PREDICTION TAB
# =========================================================

with tab_predict:

    left, right = st.columns([1.0, 0.9], gap="large")

    with left:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">🌤️ Enter Weather Conditions</div>
                <div class="card-text">
                    Provide the four required inputs. The interface
                    automatically creates Sin(Hour) and Cos(Hour),
                    then applies the saved Set B scaling statistics.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        hour = st.slider(
            "🕐 Hour of Day",
            min_value=0,
            max_value=23,
            value=12
        )

        shortwave_radiation = st.number_input(
            "☀️ Shortwave Radiation (W/m²)",
            min_value=0.0,
            max_value=1500.0,
            value=500.0,
            step=10.0
        )

        temperature_2m = st.number_input(
            "🌡️ Temperature at 2m (°C)",
            min_value=-20.0,
            max_value=60.0,
            value=30.0,
            step=0.5
        )

        cloud_cover = st.number_input(
            "☁️ Cloud Cover (%)",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=1.0
        )

        predict = st.button(
            "⚡ Predict AC Power",
            key="predict_button"
        )

    with right:

        if predict:

            sw_radiation_kw_m2 = shortwave_radiation / 1000.0

            sin_hour = np.sin(
                2 * np.pi * hour / 24
            )

            cos_hour = np.cos(
                2 * np.pi * hour / 24
            )

            raw_features = np.array([
                sw_radiation_kw_m2,
                temperature_2m,
                cloud_cover,
                sin_hour,
                cos_hour
            ], dtype=float)

            scaled_features = (
                raw_features - means
            ) / stds

            X = np.concatenate([
                [1.0],
                scaled_features
            ])

            prediction = float(X @ theta)

            prediction = max(
                0.0,
                prediction
            )

            st.markdown(
                f"""
                <div class="prediction">
                    <div class="prediction-label">
                        Predicted AC Power
                    </div>
                    <div class="prediction-value">
                        {prediction:,.2f}
                    </div>
                    <div class="prediction-unit">
                        kilowatts (kW)
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            p1, p2 = st.columns(2)

            with p1:
                st.metric(
                    "Selected Hour",
                    f"{hour:02d}:00"
                )

            with p2:
                st.metric(
                    "Radiation",
                    f"{shortwave_radiation:,.0f} W/m²"
                )

            p3, p4 = st.columns(2)

            with p3:
                st.metric(
                    "Temperature",
                    f"{temperature_2m:.1f} °C"
                )

            with p4:
                st.metric(
                    "Cloud Cover",
                    f"{cloud_cover:.0f}%"
                )

            if prediction == 0:
                st.warning(
                    "The predicted output is 0 kW after applying "
                    "the non-negative physical constraint."
                )
            elif prediction < 500:
                st.info("Low expected generation.")
            elif prediction < 5000:
                st.success("Moderate expected generation.")
            else:
                st.success("High expected generation.")

        else:

            st.markdown(
                """
                <div class="prediction">
                    <div style="font-size:55px;">⚡</div>
                    <div class="prediction-label">
                        Your prediction will appear here
                    </div>
                    <div style="
                        color:#7f91a2;
                        margin-top:10px;
                        font-size:14px;">
                        Enter the weather conditions and click
                        Predict AC Power.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# RESULTS TAB
# =========================================================

with tab_results:

    st.markdown(
        """
        <div class="card">
            <div class="card-title">📊 Assignment Results</div>
            <div class="card-text">
                Saved evaluation results from the Normal Equation,
                Batch Gradient Descent and Stochastic Gradient Descent.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.dataframe(
        rmse_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    r1, r2 = st.columns(2)

    with r1:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">🎯 Model Weights</div>
                <div class="card-text">
                    Saved Set A Normal Equation coefficients.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.dataframe(
            theta_df,
            use_container_width=True,
            hide_index=True
        )

    with r2:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">📅 Test Week Preview</div>
                <div class="card-text">
                    Actual AC power and predictions from the saved
                    test_predictions.csv file.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        preview = predictions_df[
            [
                "datetime",
                "actual_ac_power",
                "set_B_normal",
                "set_B_batch",
                "set_B_sgd"
            ]
        ].head(48)

        st.dataframe(
            preview,
            use_container_width=True,
            hide_index=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <div class="card-title">📈 Test Week Trend</div>
            <div class="card-text">
                Actual generation compared with the three Set B solvers.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    trend = predictions_df[
        [
            "datetime",
            "actual_ac_power",
            "set_B_normal",
            "set_B_batch",
            "set_B_sgd"
        ]
    ].set_index("datetime")

    st.line_chart(
        trend,
        height=420
    )


# =========================================================
# FIGURES TAB
# =========================================================

with tab_figures:

    st.markdown(
        """
        <div class="card">
            <div class="card-title">📈 Assignment Figures</div>
            <div class="card-text">
                All major figures generated during the analysis are
                available here directly from the results/figures folder.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    figure_files = [
        ("Actual vs Predicted — Test Week",
         "10_actual_vs_predicted_test_week.png"),

        ("Residuals vs Hour",
         "11_residuals_vs_hour.png"),

        ("Batch GD Learning Rates — Set A",
         "06_batch_learning_rates_set_A.png"),

        ("Batch GD Learning Rates — Set B",
         "07_batch_learning_rates_set_B.png"),

        ("SGD Learning Rates — Set A",
         "08_sgd_learning_rates_set_A.png"),

        ("SGD Learning Rates — Set B",
         "09_sgd_learning_rates_set_B.png"),
    ]

    for i in range(0, len(figure_files), 2):

        c1, c2 = st.columns(2, gap="large")

        for col, (title, filename) in zip(
            [c1, c2],
            figure_files[i:i + 2]
        ):

            with col:

                path = FIGURES_DIR / filename

                st.markdown(
                    f"""
                    <div class="card">
                        <div class="card-title">{title}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if path.exists():
                    st.image(
                        str(path),
                        use_container_width=True
                    )
                else:
                    st.warning(
                        f"Figure not found: {filename}"
                    )

    st.markdown("<br>", unsafe_allow_html=True)

    extra_figures = [
        ("AC Power vs Irradiation", "ac_power_vs_irradiation.png"),
        ("AC Power vs DC Power", "ac_vs_dc.png"),
        ("Average AC Power by Hour", "average_ac_by_hour.png"),
        ("Module Temperature vs Ambient Temperature",
         "module_vs_ambient.png"),
    ]

    st.markdown(
        """
        <div class="card">
            <div class="card-title">🔎 Exploratory Data Analysis</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    for i in range(0, len(extra_figures), 2):

        c1, c2 = st.columns(2, gap="large")

        for col, (title, filename) in zip(
            [c1, c2],
            extra_figures[i:i + 2]
        ):

            with col:

                path = RESULTS_DIR / filename

                st.markdown(
                    f"""
                    <div class="card">
                        <div class="card-title">{title}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if path.exists():
                    st.image(
                        str(path),
                        use_container_width=True
                    )
                else:
                    st.warning(
                        f"Figure not found: {filename}"
                    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        ☀️ Solar Power Intelligence Dashboard
        &nbsp;•&nbsp;
        Assignment 01
        &nbsp;•&nbsp;
        Set B Linear Regression
    </div>
    """,
    unsafe_allow_html=True
)
