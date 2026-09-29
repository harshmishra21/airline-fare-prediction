import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SkyFare AI | Airline Fare Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Glassmorphism & Modern CSS Styling
# ---------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #0f172a, #090d16, #030712);
        color: #f1f5f9;
    }
    
    /* Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.12) 0%, rgba(99, 102, 241, 0.12) 50%, rgba(168, 85, 247, 0.12) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 30px;
        backdrop-filter: blur(16px);
        box-shadow: 0 10px 30px -10px rgba(14, 165, 233, 0.2);
    }
    
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
    }

    /* Glassmorphism Card Container */
    .glass-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 24px;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 24px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
    }

    /* Boarding Pass Ticket Styling */
    .ticket-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 20px;
        padding: 26px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 12px 40px -10px rgba(0, 0, 0, 0.5);
    }
    
    .ticket-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px dashed rgba(255, 255, 255, 0.15);
        padding-bottom: 14px;
        margin-bottom: 18px;
        color: #38bdf8;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 1px;
    }
    
    .route-visualizer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 20px 0;
    }
    
    .city-code {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f8fafc;
    }
    
    .flight-line {
        flex-grow: 1;
        height: 2px;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        margin: 0 20px;
        position: relative;
    }
    
    .flight-line::after {
        content: "✈";
        position: absolute;
        top: -12px;
        left: 50%;
        transform: translateX(-50%);
        color: #38bdf8;
        font-size: 1.2rem;
    }

    .hero-price {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
    }

    .price-range-text {
        font-size: 1rem;
        color: #94a3b8;
    }
    
    .price-range-text b {
        color: #e2e8f0;
    }

    /* Badges */
    .badge-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 6px;
    }

    .badge-blue { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-purple { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
    .badge-green { background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }
    .badge-yellow { background: rgba(251, 191, 36, 0.15); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
    .badge-red { background: rgba(248, 113, 113, 0.15); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.3); }

    /* Metric Cards */
    .metric-box {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #38bdf8;
    }
    
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Load Cached Artifacts
# ---------------------------------------------------------
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
        st.error(f"⚠️ Error loading model artifacts: {e}")
        return None, None, None


model, meta, options = load_artifacts()

if model is None or meta is None or options is None:
    st.warning("⚠️ Model artifacts missing. Ensure `model.joblib`, `meta.json`, and `options.json` are present in the directory.")
    st.stop()

# ---------------------------------------------------------
# Sidebar Navigation & Branding
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <h2 style="color: #38bdf8; margin: 0; font-weight: 800;">✈️ SkyFare AI</h2>
            <p style="color: #64748b; font-size: 0.85rem; margin-top: 4px;">Smart Airline Price Intelligence</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["✈️ Fare Predictor", "📈 'When to Book' Trend", "📊 Route & Airline Comparator", "📉 Route Price Matrix", "ℹ️ Model Intelligence"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.5); border-radius: 12px; padding: 14px; border: 1px solid rgba(255, 255, 255, 0.05);">
            <div style="font-size: 0.8rem; color: #94a3b8;"><b>Model Accuracy (R²):</b> 97.95%</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;"><b>Dataset:</b> ~300K Flights</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;"><b>Scope:</b> Top 6 Indian Hubs</div>
        </div>
    """, unsafe_allow_html=True)

# Mappings
stop_label_to_num = {"Direct (0 stops)": 0, "1 Stop": 1, "2+ Stops": 2}

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">✈️ SkyFare Intelligence Engine</div>
        <div class="hero-subtitle">Real-time domestic flight fare estimations, dynamic booking lead-time trajectories, and cross-airline rate intelligence.</div>
    </div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# PAGE 1: FARE PREDICTOR
# ---------------------------------------------------------
if page == "✈️ Fare Predictor":
    col_input, col_result = st.columns([1.1, 1])

    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🛫 Configure Flight Search")

        c1, c2 = st.columns(2)
        with c1:
            source_city = st.selectbox("Source City", options["cities"], index=options["cities"].index("Delhi") if "Delhi" in options["cities"] else 0)
            airline = st.selectbox("Airline Carrier", options["airlines"], index=0)
            flight_class = st.selectbox("Cabin Class", options["classes"], index=options["classes"].index("Economy") if "Economy" in options["classes"] else 0)
            departure_time = st.selectbox("Departure Time", options["departure_times"], index=0)

        with c2:
            dest_cities = [c for c in options["cities"] if c != source_city]
            destination_city = st.selectbox("Destination City", dest_cities, index=0)
            stops_label = st.selectbox("Number of Stops", list(stop_label_to_num.keys()), index=0)
            arrival_time = st.selectbox("Arrival Time", options["arrival_times"], index=0)

        st.markdown("---")
        c3, c4 = st.columns(2)
        with c3:
            duration = st.slider("Flight Duration (Hours)", min_value=0.5, max_value=35.0, value=2.5, step=0.5)
        with c4:
            days_left = st.slider("Days Remaining to Flight", min_value=1, max_value=50, value=15, step=1)

        st.markdown('</div>', unsafe_allow_html=True)

    stops_num = stop_label_to_num[stops_label]
    route = f"{source_city}_{destination_city}"

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

    with col_result:
        try:
            predicted_price = float(model.predict(input_data)[0])
            p10 = meta.get("residual_p10", -1900)
            p90 = meta.get("residual_p90", 1900)
            lower_bound = max(500, predicted_price + p10)
            upper_bound = predicted_price + p90

            # Boarding Pass Render
            st.markdown(f"""
            <div class="ticket-card">
                <div class="ticket-header">
                    <span>BOARDING PASS SUMMARY</span>
                    <span>{airline.upper()} • {flight_class.upper()}</span>
                </div>
                <div class="route-visualizer">
                    <div>
                        <div class="city-code">{source_city[:3].upper()}</div>
                        <div style="font-size: 0.85rem; color: #94a3b8;">{source_city}</div>
                    </div>
                    <div class="flight-line"></div>
                    <div style="text-align: right;">
                        <div class="city-code">{destination_city[:3].upper()}</div>
                        <div style="font-size: 0.85rem; color: #94a3b8;">{destination_city}</div>
                    </div>
                </div>
                <div>
                    <span class="badge-pill badge-blue">⏱ {duration} hrs</span>
                    <span class="badge-pill badge-purple">📍 {stops_label}</span>
                    <span class="badge-pill badge-green">🛫 {departure_time}</span>
                    <span class="badge-pill badge-blue">🛬 {arrival_time}</span>
                </div>
                <hr style="border-color: rgba(255, 255, 255, 0.1); margin: 18px 0;">
                <div style="font-size: 0.9rem; color: #94a3b8;">Estimated Ticket Fare</div>
                <div class="hero-price">₹ {predicted_price:,.0f}</div>
                <div class="price-range-text">80% Confidence Interval: <b>₹ {lower_bound:,.0f}</b> — <b>₹ {upper_bound:,.0f}</b></div>
            </div>
            """, unsafe_allow_html=True)

            # Advice Section
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 💡 Smart Booking Recommendation")
            if days_left <= 7:
                badge = '<span class="badge-pill badge-red">⚠️ High Fare Alert (Departure <= 7 days)</span>'
                advice = "Fares are near peak dynamic surge levels. If your travel date is fixed, booking immediately is recommended to avoid further price escalation."
            elif 8 <= days_left <= 20:
                badge = '<span class="badge-pill badge-yellow">⚡ Moderately Priced (8-20 days out)</span>'
                advice = "Seat availability is shrinking and price curves are beginning to rise. Booking now offers a reasonable trade-off between price and flexibility."
            else:
                badge = '<span class="badge-pill badge-green">✅ Optimal Booking Window (20+ days)</span>'
                advice = "You are booking well in advance. Fares are at or near baseline prices for this route."

            st.markdown(f"{badge}<br><p style='margin-top: 10px; color: #cbd5e1;'>{advice}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error computing price: {e}")


# ---------------------------------------------------------
# PAGE 2: WHEN TO BOOK TREND
# ---------------------------------------------------------
elif page == "📈 'When to Book' Trend":
    st.markdown("### 📈 50-Day Booking Lead Time Trajectory")
    st.markdown("Explore how expected fare changes as departure date approaches for your selected flight configuration.")

    col1, col2, col3 = st.columns(3)
    with col1:
        source_city = st.selectbox("Source City", options["cities"], index=0, key="tr_src")
        airline = st.selectbox("Airline Carrier", options["airlines"], index=0, key="tr_air")
    with col2:
        dest_cities = [c for c in options["cities"] if c != source_city]
        destination_city = st.selectbox("Destination City", dest_cities, index=0, key="tr_dst")
        flight_class = st.selectbox("Cabin Class", options["classes"], index=0, key="tr_cls")
    with col3:
        stops_label = st.selectbox("Stops", list(stop_label_to_num.keys()), index=0, key="tr_stp")
        duration = st.slider("Duration (Hrs)", 0.5, 35.0, 2.5, 0.5, key="tr_dur")

    route = f"{source_city}_{destination_city}"
    stops_num = stop_label_to_num[stops_label]

    days_range = list(range(1, 51))
    trend_rows = [{
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
    } for d in days_range]

    trend_df = pd.DataFrame(trend_rows)
    trend_df["predicted_fare"] = model.predict(trend_df)

    cheapest_row = trend_df.loc[trend_df["predicted_fare"].idxmin()]
    cheapest_day = int(cheapest_row["days_left"])
    cheapest_price = float(cheapest_row["predicted_fare"])
    max_price = float(trend_df["predicted_fare"].max())
    potential_savings = max_price - cheapest_price

    st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid #34d399;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #34d399;">💡 Optimal Booking Window: ~{cheapest_day} Days Out</div>
            <div style="color: #cbd5e1; margin-top: 4px;">
                Lowest predicted fare is <b>₹ {cheapest_price:,.0f}</b> when booked around {cheapest_day} days prior. 
                Booking during peak surge (1-3 days before departure) could increase fares up to <b>₹ {max_price:,.0f}</b> (a difference of <b>₹ {potential_savings:,.0f}</b>).
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Plotly Line Chart
    fig = px.area(
        trend_df,
        x="days_left",
        y="predicted_fare",
        labels={"days_left": "Days Left Before Departure (Departure is on Right)", "predicted_fare": "Predicted Fare (INR ₹)"},
        title=f"Fare Trajectory: {source_city} ➔ {destination_city} ({airline}, {flight_class})",
        markers=True
    )
    fig.update_xaxes(autorange="reversed")  # 50 on left, 1 on right
    fig.update_traces(line_color="#38bdf8", fillcolor="rgba(56, 189, 248, 0.15)", marker=dict(size=6))
    fig.add_hline(y=cheapest_price, line_dash="dash", line_color="#34d399", annotation_text=f"Minimum Fare: ₹{cheapest_price:,.0f}", annotation_position="bottom right")
    fig.update_layout(template="plotly_dark", height=480, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# PAGE 3: ROUTE & AIRLINE COMPARATOR
# ---------------------------------------------------------
elif page == "📊 Route & Airline Comparator":
    st.markdown("### 📊 Cross-Airline Fare Comparison")
    st.markdown("Compare price estimations across all operating carriers for your specific route and travel lead time.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        source_city = st.selectbox("Source City", options["cities"], index=0, key="cmp_src")
    with col2:
        dest_cities = [c for c in options["cities"] if c != source_city]
        destination_city = st.selectbox("Destination City", dest_cities, index=0, key="cmp_dst")
    with col3:
        flight_class = st.selectbox("Cabin Class", options["classes"], index=0, key="cmp_cls")
    with col4:
        days_left = st.slider("Days Remaining", 1, 50, 15, key="cmp_days")

    comp_rows = [{
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
    } for a in options["airlines"]]

    comp_df = pd.DataFrame(comp_rows)
    comp_df["Predicted Fare (INR)"] = model.predict(comp_df)
    comp_df = comp_df.sort_values("Predicted Fare (INR)")

    cheapest_carrier = comp_df.iloc[0]["airline"]
    cheapest_fare = comp_df.iloc[0]["Predicted Fare (INR)"]
    priciest_carrier = comp_df.iloc[-1]["airline"]
    priciest_fare = comp_df.iloc[-1]["Predicted Fare (INR)"]

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="metric-val">₹ {cheapest_fare:,.0f}</div><div class="metric-lbl">Lowest Carrier ({cheapest_carrier})</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="metric-val">₹ {priciest_fare:,.0f}</div><div class="metric-lbl">Highest Carrier ({priciest_carrier})</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-box"><div class="metric-val">₹ {priciest_fare - cheapest_fare:,.0f}</div><div class="metric-lbl">Carrier Fare Variance</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Custom Airline Palette
    palette = {"Vistara": "#a855f7", "Air_India": "#ef4444", "Indigo": "#3b82f6", "AirAsia": "#f97316", "SpiceJet": "#eab308", "GO_FIRST": "#14b8a6"}

    fig = px.bar(
        comp_df,
        x="airline",
        y="Predicted Fare (INR)",
        color="airline",
        color_discrete_map=palette,
        title=f"Predicted Fares for {source_city} ➔ {destination_city} ({flight_class}, {days_left} Days Out)",
        text_auto=".0f"
    )
    fig.update_layout(template="plotly_dark", height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# PAGE 4: ROUTE PRICE MATRIX
# ---------------------------------------------------------
elif page == "📉 Route Price Matrix":
    st.markdown("### 📉 Inter-City Route Fare Matrix & Analytics")
    st.markdown("Analyze estimated average fares across all 30 metro route combinations.")

    c1, c2 = st.columns(2)
    with c1:
        flight_class = st.selectbox("Select Cabin Class", options["classes"], index=0, key="mx_cls")
    with c2:
        days_left = st.slider("Days Remaining Before Flight", 1, 50, 20, key="mx_days")

    matrix_rows = []
    cities_list = options["cities"]
    for src in cities_list:
        for dst in cities_list:
            if src != dst:
                matrix_rows.append({
                    "duration": 2.5,
                    "days_left": days_left,
                    "stops_num": 0,
                    "airline": "Vistara",
                    "source_city": src,
                    "destination_city": dst,
                    "departure_time": "Morning",
                    "arrival_time": "Afternoon",
                    "class": flight_class,
                    "route": f"{src}_{dst}"
                })

    matrix_df = pd.DataFrame(matrix_rows)
    matrix_df["Predicted Fare"] = model.predict(matrix_df)
    
    # Pivot for Heatmap
    pivot_df = matrix_df.pivot(index="source_city", columns="destination_city", values="Predicted Fare")

    # Build explicit text matrix with formatted numbers inside EVERY cell
    text_matrix = []
    for s in cities_list:
        row_text = []
        for d in cities_list:
            if s == d:
                row_text.append("—")
            else:
                val = pivot_df.loc[s, d]
                row_text.append(f"₹{int(val):,}")
        text_matrix.append(row_text)

    # Plotly Heatmap with formatted numbers in each box
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=cities_list,
        y=cities_list,
        text=text_matrix,
        texttemplate="<b>%{text}</b>",
        textfont=dict(size=12, color="#ffffff"),
        colorscale="Viridis",
        colorbar=dict(title="Estimated Fare (₹)"),
        hoverongaps=False
    ))
    fig.update_layout(
        title=f"Route Price Matrix ({flight_class}, {days_left} Days Out)",
        xaxis_title="Destination City",
        yaxis_title="Source City",
        template="plotly_dark",
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    # Top 5 Most Expensive vs Top 5 Cheapest Routes
    st.markdown("#### 📊 Route Fare Rankings")
    matrix_sorted = matrix_df.sort_values("Predicted Fare", ascending=False).copy()
    matrix_sorted["Route Name"] = matrix_sorted["source_city"] + " ➔ " + matrix_sorted["destination_city"]

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.markdown("##### 🔴 Top 5 Most Expensive Routes")
        top_exp = matrix_sorted.head(5)
        fig_exp = px.bar(
            top_exp,
            x="Predicted Fare",
            y="Route Name",
            orientation="h",
            color="Predicted Fare",
            color_continuous_scale="Reds",
            text_auto=",.0f"
        )
        fig_exp.update_layout(template="plotly_dark", height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_exp, use_container_width=True)

    with r_col2:
        st.markdown("##### 🟢 Top 5 Most Affordable Routes")
        top_cheap = matrix_sorted.tail(5).sort_values("Predicted Fare")
        fig_cheap = px.bar(
            top_cheap,
            x="Predicted Fare",
            y="Route Name",
            orientation="h",
            color="Predicted Fare",
            color_continuous_scale="Greens_r",
            text_auto=",.0f"
        )
        fig_cheap.update_layout(template="plotly_dark", height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_cheap, use_container_width=True)


# ---------------------------------------------------------
# PAGE 5: MODEL INTELLIGENCE
# ---------------------------------------------------------
elif page == "ℹ️ Model Intelligence":
    st.markdown("### ℹ️ Machine Learning Pipeline Specifications")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{meta.get("r2", 0):.4f}</div><div class="metric-lbl">R² Score (Accuracy)</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box"><div class="metric-val">₹ {meta.get("rmse", 0):,.0f}</div><div class="metric-lbl">Root Mean Sq Error</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-box"><div class="metric-val">₹ {meta.get("mae", 0):,.0f}</div><div class="metric-lbl">Mean Absolute Error</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-box"><div class="metric-val">300,153</div><div class="metric-lbl">Training Samples</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <h4>🔬 Pipeline Architecture & Transformation Steps</h4>
        <ul>
            <li><b>Preprocessing:</b> Numerical features (<code>duration</code>, <code>days_left</code>, <code>stops_num</code>) scaled via <code>StandardScaler</code>. Categorical features (<code>airline</code>, <code>source_city</code>, <code>destination_city</code>, <code>departure_time</code>, <code>arrival_time</code>, <code>class</code>, <code>route</code>) encoded via <code>OneHotEncoder(handle_unknown='ignore')</code>.</li>
            <li><b>Regressor:</b> <code>RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=2, n_jobs=-1)</code>.</li>
            <li><b>Residual Bounds:</b> Empirical 10th and 90th percentile offsets calculated from out-of-sample predictions to render 80% confidence interval ranges.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>SkyFare AI • Built with Streamlit, Scikit-Learn & Plotly</div>", unsafe_allow_html=True)
