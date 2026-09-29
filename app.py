import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import os

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Airline Fare Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
    <style>
    /* Global Styles */
    .main {
        padding-top: 1rem;
    }
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Card Container */
    .metric-card {
        background: linear-gradient(135deg, #1e2640 0%, #111827 100%);
        border: 100px;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    .price-display {
        font-size: 2.5rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 10px 0;
    }

    .price-bounds {
        font-size: 1.1rem;
        color: #94a3b8;
    }
    
    .badge-recommendation {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 10px;
    }
    
    .badge-green {
        background-color: #064e3b;
        color: #34d399;
        border: 1px solid #059669;
    }
    
    .badge-yellow {
        background-color: #78350f;
        color: #fbbf24;
        border: 1px solid #d97706;
    }

    .badge-red {
        background-color: #7f1d1d;
        color: #f87171;
        border: 1px solid #dc2626;
    }

    /* Form Section styling */
    .stSelectbox label, .stSlider label {
        font-weight: 600;
        color: #cbd5e1;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("model.joblib")
        with open("meta.json", "r") as f:
            meta = json.load(f)
        with open("options.json", "r") as f:
            options = json.load(f)
        return model, meta, options
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None


model, meta, options = load_artifacts()

if model is None or meta is None or options is None:
    st.warning("⚠️ Model artifacts missing. Please ensure `model.joblib`, `meta.json`, and `options.json` are present in the directory.")
    st.stop()

# Header Section
st.title("✈️ Airline Fare Prediction Engine")
st.markdown("Predict flight ticket prices in India, explore price dynamics over time, and compare airline rates powered by Machine Learning.")

# Sidebar Controls
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Mode",
    ["✈️ Fare Predictor", "📈 Price Trend ('When to Book')", "📊 Route & Airline Comparison", "ℹ️ Model Performance & Analytics"]
)

# Helper function for stops label mapping
stop_label_to_num = {"Direct (0 stops)": 0, "1 Stop": 1, "2+ Stops": 2}
stop_num_to_label = {0: "Direct (0 stops)", 1: "1 Stop", 2: "2+ Stops"}

if page == "✈️ Fare Predictor":
    st.subheader("Flight Details Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        source_city = st.selectbox("Source City", options["cities"], index=options["cities"].index("Delhi") if "Delhi" in options["cities"] else 0)
        airline = st.selectbox("Airline", options["airlines"], index=0)
        flight_class = st.selectbox("Cabin Class", options["classes"], index=options["classes"].index("Economy") if "Economy" in options["classes"] else 0)

    with col2:
        dest_cities = [c for c in options["cities"] if c != source_city]
        destination_city = st.selectbox("Destination City", dest_cities, index=0)
        departure_time = st.selectbox("Departure Time", options["departure_times"], index=0)
        arrival_time = st.selectbox("Arrival Time", options["arrival_times"], index=0)

    with col3:
        stops_label = st.selectbox("Number of Stops", list(stop_label_to_num.keys()), index=0)
        duration = st.slider("Flight Duration (Hours)", min_value=0.5, max_value=40.0, value=2.5, step=0.5)
        days_left = st.slider("Days Remaining Before Departure", min_value=1, max_value=50, value=15, step=1)

    stops_num = stop_label_to_num[stops_label]
    route = f"{source_city}_{destination_city}"

    st.markdown("---")

    # Predict Button & Computation
    input_data = pd.DataFrame([{
        "duration": duration,
        "days_left": days_left,
        "stops_num": stops_num,
        "airline": airline,
        "source_city": source_city,
        "destination_city": destination_city,
        "departure_time": departure_time,
        "arrival_time": arrival_time,
        "class": flight_class,
        "route": route
    }])

    try:
        predicted_price = float(model.predict(input_data)[0])
        p10 = meta.get("residual_p10", -1900)
        p90 = meta.get("residual_p90", 1900)
        lower_bound = max(500, predicted_price + p10)
        upper_bound = predicted_price + p90

        # Result Display Card
        res_col1, res_col2 = st.columns([1.5, 1])

        with res_col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Estimated Flight Fare</h3>
                <div class="price-display">₹ {predicted_price:,.0f}</div>
                <div class="price-bounds">80% Expected Price Range: <b>₹ {lower_bound:,.0f}</b> — <b>₹ {upper_bound:,.0f}</b></div>
                <hr style="border-color: #374151; margin: 15px 0;">
                <p><b>Route:</b> {source_city} ➔ {destination_city} | <b>Airline:</b> {airline} | <b>Class:</b> {flight_class}</p>
                <p><b>Duration:</b> {duration} hrs | <b>Stops:</b> {stops_label} | <b>Days to Flight:</b> {days_left} days</p>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("### Booking Advice")
            if days_left <= 7:
                badge = '<span class="badge-recommendation badge-red">⚠️ High Fare Alert: Departure is within 7 days</span>'
                advice = "Prices are currently near peak levels. If travel is confirmed, booking immediately is recommended to prevent further surge pricing."
            elif 8 <= days_left <= 20:
                badge = '<span class="badge-recommendation badge-yellow">⚡ Moderately Priced: 8-20 days out</span>'
                advice = "Fares are starting to rise as seat availability shrinks. Booking now is advisable if your dates are fixed."
            else:
                badge = '<span class="badge-recommendation badge-green">✅ Optimal Booking Period (20+ days)</span>'
                advice = "You are booking well in advance. Fares are near baseline for this route."

            st.markdown(f"{badge}<br><br>{advice}", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error predicting fare: {e}")

elif page == "📈 Price Trend ('When to Book')":
    st.subheader("📈 Fare Trajectory vs Booking Lead Time")
    st.markdown("Analyze how the fare for a specific flight configuration changes as departure date approaches (from 50 days to 1 day remaining).")

    col1, col2, col3 = st.columns(3)
    with col1:
        source_city = st.selectbox("Source City", options["cities"], index=0, key="trend_source")
        airline = st.selectbox("Airline", options["airlines"], index=0, key="trend_airline")
    with col2:
        dest_cities = [c for c in options["cities"] if c != source_city]
        destination_city = st.selectbox("Destination City", dest_cities, index=0, key="trend_dest")
        flight_class = st.selectbox("Cabin Class", options["classes"], index=0, key="trend_class")
    with col3:
        stops_label = st.selectbox("Stops", list(stop_label_to_num.keys()), index=0, key="trend_stops")
        duration = st.slider("Duration (Hrs)", 0.5, 40.0, 2.5, 0.5, key="trend_dur")

    route = f"{source_city}_{destination_city}"
    stops_num = stop_label_to_num[stops_label]

    days_range = list(range(1, 51))
    trend_rows = []
    for d in days_range:
        trend_rows.append({
            "duration": duration,
            "days_left": d,
            "stops_num": stops_num,
            "airline": airline,
            "source_city": source_city,
            "destination_city": destination_city,
            "departure_time": "Morning",
            "arrival_time": "Afternoon",
            "class": flight_class,
            "route": route
        })

    trend_df = pd.DataFrame(trend_rows)
    trend_df["predicted_fare"] = model.predict(trend_df)

    cheapest_row = trend_df.loc[trend_df["predicted_fare"].idxmin()]
    cheapest_day = int(cheapest_row["days_left"])
    cheapest_price = float(cheapest_row["predicted_fare"])

    st.success(f"💡 **Recommendation:** For this route & class, lowest predicted fare occurs approximately **{cheapest_day} days** prior to departure at **₹ {cheapest_price:,.0f}**.")

    # Plotly Line Chart
    fig = px.line(
        trend_df,
        x="days_left",
        y="predicted_fare",
        labels={"days_left": "Days Left Before Departure", "predicted_fare": "Predicted Fare (INR)"},
        title=f"Predicted Fare vs Days Left ({source_city} ➔ {destination_city} | {airline} {flight_class})",
        markers=True
    )
    fig.update_xaxes(autorange="reversed")  # 50 days on left, 1 day on right
    fig.add_hline(y=cheapest_price, line_dash="dash", line_color="green", annotation_text=f"Min Fare: ₹{cheapest_price:,.0f}")
    fig.update_traces(line_color="#38bdf8", line_width=3)
    fig.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Route & Airline Comparison":
    st.subheader("📊 Airline Fare Comparison Across Route")
    st.markdown("Compare predicted ticket prices across all available airlines for a given route and travel day.")

    col1, col2, col3 = st.columns(3)
    with col1:
        source_city = st.selectbox("Source City", options["cities"], index=0, key="comp_src")
    with col2:
        dest_cities = [c for c in options["cities"] if c != source_city]
        destination_city = st.selectbox("Destination City", dest_cities, index=0, key="comp_dest")
    with col3:
        flight_class = st.selectbox("Class", options["classes"], index=0, key="comp_class")

    days_left = st.slider("Days Remaining", 1, 50, 15, key="comp_days")

    comp_rows = []
    for a in options["airlines"]:
        comp_rows.append({
            "duration": 2.5,
            "days_left": days_left,
            "stops_num": 0,
            "airline": a,
            "source_city": source_city,
            "destination_city": destination_city,
            "departure_time": "Morning",
            "arrival_time": "Afternoon",
            "class": flight_class,
            "route": f"{source_city}_{destination_city}"
        })

    comp_df = pd.DataFrame(comp_rows)
    comp_df["Predicted Fare (INR)"] = model.predict(comp_df)
    comp_df = comp_df.sort_values("Predicted Fare (INR)")

    fig = px.bar(
        comp_df,
        x="airline",
        y="Predicted Fare (INR)",
        color="airline",
        title=f"Fare Comparison for {source_city} ➔ {destination_city} ({flight_class}, {days_left} days left)",
        text_auto=".0f"
    )
    fig.update_layout(template="plotly_dark", height=450, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(comp_df[["airline", "class", "days_left", "Predicted Fare (INR)"]], use_container_width=True)

elif page == "ℹ️ Model Performance & Analytics":
    st.subheader("ℹ️ Machine Learning Model Overview & Evaluation")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Architecture", meta.get("model_name", "RandomForest"))
    col2.metric("R² Score", f"{meta.get('r2', 0):.4f}")
    col3.metric("RMSE (Test)", f"₹ {meta.get('rmse', 0):,.2f}")
    col4.metric("MAE (Test)", f"₹ {meta.get('mae', 0):,.2f}")

    st.markdown("---")

    st.markdown("""
    ### 🔬 Methodology & Feature Engineering
    - **Dataset Source:** Scraped from EaseMyTrip containing ~300,000 flight entries covering 6 major Indian metro cities (Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, Chennai).
    - **Key Features Used:**
      - `days_left`: Booking lead time before flight departure.
      - `duration`: Total journey duration in hours.
      - `stops_num`: Number of flight layovers (0 for direct, 1, 2+).
      - `class`: Economy vs Business cabin class (primary price determinant).
      - `airline`, `source_city`, `destination_city`, `departure_time`, `arrival_time`, `route`.
    - **Preprocessing:** `StandardScaler` for numeric columns and `OneHotEncoder` for categorical parameters structured within an end-to-end `sklearn` Pipeline.
    """)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #6b7280; font-size: 0.85rem;'>Airline Fare Prediction Project • Built with Streamlit, Scikit-Learn & Plotly</div>", unsafe_allow_html=True)
