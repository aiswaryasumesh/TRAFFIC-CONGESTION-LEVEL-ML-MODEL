import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Traffic Flow",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ================================
       MAIN APPLICATION
       ================================ */

    .stApp {
        background: linear-gradient(
            135deg,
            #f7f9fc 0%,
            #eef3f9 100%
        );
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }


    /* ================================
       HEADER
       ================================ */

    .top-header {
        background: linear-gradient(
            110deg,
            #061a3a,
            #102e68,
            #8d173e
        );

        padding: 32px 40px;

        border-radius: 22px;

        margin-bottom: 22px;

        box-shadow:
            0 12px 30px rgba(6, 26, 58, 0.18);
    }

    .brand {
        color: #ffffff !important;

        font-size: 14px;

        font-weight: 800;

        letter-spacing: 3px;

        margin-bottom: 10px;
    }

    .main-title {
        color: #ffffff !important;

        font-size: 42px;

        font-weight: 850;

        line-height: 1.1;

        margin: 0;
    }

    .main-subtitle {
        color: #dce7f7 !important;

        font-size: 16px;

        margin-top: 10px;
    }


    /* ================================
       STATUS BAR
       ================================ */

    .status-bar {
        background: #ffffff;

        border: 1px solid #dce4ef;

        border-radius: 15px;

        padding: 14px 20px;

        margin-bottom: 25px;

        box-shadow:
            0 5px 18px rgba(0, 0, 0, 0.05);
    }

    .status-left {
        color: #14213d !important;

        font-weight: 700;

        font-size: 14px;
    }

    .status-right {
        color: #16803c !important;

        font-weight: 700;

        font-size: 14px;
    }


    /* ================================
       SECTION
       ================================ */

    .section-card {
        background: #ffffff;

        padding: 25px 28px;

        border-radius: 20px;

        border: 1px solid #dfe6ef;

        box-shadow:
            0 7px 24px rgba(7, 26, 61, 0.06);

        margin-bottom: 5px;
    }

    .section-heading {
        color: #071a3d !important;

        font-size: 23px;

        font-weight: 800;

        margin-bottom: 7px;
    }

    .section-line {
        height: 4px;

        width: 55px;

        background: #9c1942;

        border-radius: 10px;
    }


    /* ================================
       INPUT LABELS
       ================================ */

    .main .block-container label {
        color: #172033 !important;

        font-weight: 650 !important;
    }


    /* ================================
       PREDICT BUTTON
       ================================ */

    div.stButton > button {

        width: 100%;

        height: 56px;

        border-radius: 13px;

        border: none;

        background: linear-gradient(
            90deg,
            #8b1538,
            #b3204d
        );

        color: #ffffff !important;

        font-size: 17px;

        font-weight: 800;

        box-shadow:
            0 7px 18px rgba(139, 21, 56, 0.25);

        transition: all 0.2s ease;
    }

    div.stButton > button:hover {

        background: linear-gradient(
            90deg,
            #74122f,
            #98193f
        );

        transform: translateY(-1px);

        color: #ffffff !important;
    }


    /* ================================
       PREDICTION RESULT
       ================================ */

    .prediction-card {

        padding: 32px;

        border-radius: 22px;

        text-align: center;

        margin-top: 25px;

        margin-bottom: 24px;

        box-shadow:
            0 12px 30px rgba(0, 0, 0, 0.08);
    }

    .prediction-small {

        font-size: 14px;

        font-weight: 800;

        letter-spacing: 2px;

        text-transform: uppercase;

        margin-bottom: 8px;
    }

    .prediction-big {

        font-size: 44px;

        font-weight: 900;

        letter-spacing: 1px;
    }


    /* ================================
       METRIC CARDS
       ================================ */

    .metric-card {

        background: #ffffff;

        border-radius: 17px;

        padding: 21px;

        text-align: center;

        border: 1px solid #dfe6ef;

        box-shadow:
            0 5px 18px rgba(0, 0, 0, 0.05);
    }

    .metric-number {

        color: #0d2b63 !important;

        font-size: 27px;

        font-weight: 900;
    }

    .metric-label {

        color: #64748b !important;

        font-size: 13px;

        margin-top: 4px;
    }


    /* ================================
       FOOTER
       ================================ */

    .footer {

        text-align: center;

        color: #718096 !important;

        font-size: 12px;

        margin-top: 35px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# DAY MAPPING
# ============================================================

DAY_MAPPING = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

DAYS = list(DAY_MAPPING.keys())


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("Traffic.csv")

    df["Hour"] = pd.to_datetime(
        df["Time"],
        format="%I:%M:%S %p"
    ).dt.hour

    df["Day_Number"] = df["Day of the week"].map(
        DAY_MAPPING
    )

    return df


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(df):

    features = [
        "Hour",
        "Day_Number",
        "CarCount",
        "BikeCount",
        "BusCount",
        "TruckCount"
    ]

    X = df[features]

    y = df["Traffic Situation"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = DecisionTreeClassifier(
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

df = load_data()

model = train_model(df)


# ============================================================
# HEADER
# ============================================================

st.html("""
<div class="top-header">

    <div class="brand">
        🚦 TRAFFIC FLOW
    </div>

    <div class="main-title">
        Congestion Level Predictor
    </div>

    <div class="main-subtitle">
        Enter traffic conditions to estimate the current congestion level.
    </div>

</div>
""")


# ============================================================
# STATUS BAR
# ============================================================

st.html("""
<div class="status-bar">

    <div class="status-left">
        📍 Traffic Monitoring
    </div>

    <div class="status-right">
        ● Prediction System Ready
    </div>

</div>
""")


# ============================================================
# TRAFFIC INPUT SECTION
# ============================================================

st.html("""
<div class="section-card">

    <div class="section-heading">
        🚘 Traffic Conditions
    </div>

    <div class="section-line"></div>

</div>
""")


col1, col2 = st.columns(2)


# ============================================================
# LEFT INPUTS
# ============================================================

with col1:

    day = st.selectbox(
        "Day of the Week",
        DAYS,
        index=1
    )

    time = st.time_input(
        "Time",
        value=pd.Timestamp(
            "08:00:00"
        ).time()
    )

    car_count = st.number_input(
        "🚗 Number of Cars",
        min_value=0,
        max_value=1000,
        value=100,
        step=1
    )


# ============================================================
# RIGHT INPUTS
# ============================================================

with col2:

    bike_count = st.number_input(
        "🏍️ Number of Bikes",
        min_value=0,
        max_value=1000,
        value=50,
        step=1
    )

    bus_count = st.number_input(
        "🚌 Number of Buses",
        min_value=0,
        max_value=500,
        value=20,
        step=1
    )

    truck_count = st.number_input(
        "🚛 Number of Trucks",
        min_value=0,
        max_value=500,
        value=20,
        step=1
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict = st.button(
    "🚦  CHECK TRAFFIC CONGESTION"
)


# ============================================================
# PREDICTION
# ============================================================

if predict:

    # Convert time to hour

    hour = time.hour

    # Convert day to number

    day_number = DAY_MAPPING[day]


    # Create input dataframe

    input_data = pd.DataFrame({

        "Hour": [hour],

        "Day_Number": [day_number],

        "CarCount": [car_count],

        "BikeCount": [bike_count],

        "BusCount": [bus_count],

        "TruckCount": [truck_count]

    })


    # Prediction

    prediction = model.predict(
        input_data
    )[0]


    # Prediction probabilities

    probabilities = model.predict_proba(
        input_data
    )[0]


    classes = model.classes_


    probability_df = pd.DataFrame({

        "Traffic Level": classes,

        "Probability": probabilities

    })


    confidence = max(
        probabilities
    ) * 100


    # ========================================================
    # RESULT COLORS
    # ========================================================

    if prediction == "low":

        bg = "#e3f7eb"

        text = "#157347"

        emoji = "🟢"


    elif prediction == "normal":

        bg = "#fff4d6"

        text = "#8a6500"

        emoji = "🟡"


    elif prediction == "heavy":

        bg = "#ffe4d1"

        text = "#a33a00"

        emoji = "🟠"


    else:

        bg = "#ffe0e5"

        text = "#9b1738"

        emoji = "🔴"


    # ========================================================
    # PREDICTION RESULT CARD
    # ========================================================

    st.html(f"""
    <div class="prediction-card"
         style="background:{bg};">

        <div class="prediction-small"
             style="color:{text};">

            Current Congestion Level

        </div>

        <div class="prediction-big"
             style="color:{text};">

            {emoji} {prediction.upper()}

        </div>

    </div>
    """)


    # ========================================================
    # METRICS
    # ========================================================

    total_vehicles = (
        car_count
        + bike_count
        + bus_count
        + truck_count
    )


    metric1, metric2, metric3 = st.columns(3)


    # Confidence

    with metric1:

        st.html(f"""
        <div class="metric-card">

            <div class="metric-number">
                {confidence:.1f}%
            </div>

            <div class="metric-label">
                Prediction Confidence
            </div>

        </div>
        """)


    # Total vehicles

    with metric2:

        st.html(f"""
        <div class="metric-card">

            <div class="metric-number">
                {total_vehicles}
            </div>

            <div class="metric-label">
                Total Vehicles
            </div>

        </div>
        """)


    # Selected time

    with metric3:

        st.html(f"""
        <div class="metric-card">

            <div class="metric-number">
                {time.strftime("%I:%M %p")}
            </div>

            <div class="metric-label">
                Selected Time
            </div>

        </div>
        """)


    # ========================================================
    # PROBABILITY GRAPH
    # ========================================================

    st.markdown(
        "### 📊 Traffic Level Probability"
    )


    fig, ax = plt.subplots(
        figsize=(9, 4)
    )


    bars = ax.bar(
        probability_df["Traffic Level"],
        probability_df["Probability"] * 100
    )


    ax.set_ylabel(
        "Probability (%)"
    )

    ax.set_xlabel(
        "Traffic Level"
    )

    ax.set_ylim(
        0,
        100
    )

    ax.set_title(
        "Prediction Probability"
    )


    # Display percentages

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            height + 2,

            f"{height:.1f}%",

            ha="center",

            fontsize=10
        )


    plt.tight_layout()

    st.pyplot(fig)


# ============================================================
# TRAFFIC LEVEL INDICATORS
# ============================================================

st.markdown(
    "### 🚦 Congestion Levels"
)


level1, level2, level3, level4 = st.columns(4)


with level1:

    st.success(
        "🟢 **LOW**"
    )


with level2:

    st.warning(
        "🟡 **NORMAL**"
    )


with level3:

    st.warning(
        "🟠 **HEAVY**"
    )


with level4:

    st.error(
        "🔴 **HIGH**"
    )


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">

    Traffic Flow • Congestion Prediction

</div>
""")