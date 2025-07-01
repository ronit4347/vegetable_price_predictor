# app.py
import streamlit as st
import datetime
import pandas as pd
import altair as alt # Import altair for custom charts
from predictor import predict_price, get_available_vegetables

st.set_page_config(
    page_title="VEGETABLE PRICE PREDICTOR",
    page_icon="🥦",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🥦 VEGETABLE PRICE PREDICTOR")
st.markdown("""
Welcome to the Vegetable Price Predictor!
Get estimated prices for vegetables and analyze their trends and distributions over time.
""")

# --- Sidebar ---
st.sidebar.header("About This App")
st.sidebar.markdown("""
This application predicts synthetic vegetable prices using a simple Linear Regression model.
It demonstrates how to build a basic ML model with a Streamlit frontend.
""")
st.sidebar.markdown("---")

# Initialize session state for history if it doesn't exist
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# --- Helper function to get available vegetables ---
@st.cache_data
def get_veg_list():
    """Caches the list of available vegetables to avoid re-loading on every rerun."""
    return get_available_vegetables()

available_vegetables = get_veg_list()

if not available_vegetables:
    st.error("Error: Could not load available vegetables. Please ensure the model is trained by running `python model_trainer.py`.")
    st.stop() # Stop the app if vegetables cannot be loaded

# --- Single Vegetable Prediction & Trend Analysis ---
st.header("Individual Vegetable Price Analysis")

col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input("Select a Date", value=datetime.date.today())

with col2:
    selected_vegetable = st.selectbox("Select a Vegetable", available_vegetables, key='single_veg_select')

forecast_horizon = st.slider(
    "Select Forecast Horizon (Days around selected date)",
    min_value=7, max_value=90, value=30, step=7,
    help="Number of days before and after the selected date to show in the trend graph."
)

# Chart type selection for individual analysis
chart_type_individual = st.radio(
    "Select Chart Type for Individual Analysis",
    ('Bar Chart', 'Line Chart'),
    key='chart_type_individual'
)

if st.button("Predict Price & Show Trend", help="Click to get the predicted price and view its trend over time for the selected vegetable."):
    if selected_date and selected_vegetable:
        date_str = selected_date.strftime('%Y-%m-%d')

        with st.spinner(f"Predicting price for {selected_vegetable} on {date_str}..."):
            predicted_price = predict_price(date_str, selected_vegetable)

            st.markdown("---")
            st.subheader("Prediction Result")
            if predicted_price != -1.0:
                st.success(f"The predicted price for **{selected_vegetable}** on **{selected_date.strftime('%B %d, %Y')}** is approximately: **₹{predicted_price:.2f}**")
                st.info("Please note: This is a synthetic model for demonstration purposes and may not reflect real-world prices accurately.")

                # Add to history
                st.session_state.prediction_history.append({
                    'Date': selected_date.strftime('%Y-%m-%d'),
                    'Vegetable': selected_vegetable,
                    'Predicted Price': f"₹{predicted_price:.2f}"
                })

                st.subheader("Price Trend Analysis")
                st.write(f"Predicted price trend for **{selected_vegetable}**:")

                # Generate dates for trend analysis
                start_date = selected_date - datetime.timedelta(days=forecast_horizon)
                end_date = selected_date + datetime.timedelta(days=forecast_horizon)
                date_range = pd.date_range(start=start_date, end=end_date, freq='D')

                trend_data = []
                for dt in date_range:
                    daily_price = predict_price(dt.strftime('%Y-%m-%d'), selected_vegetable)
                    if daily_price != -1.0:
                        trend_data.append({'Date': dt, 'Predicted Price': daily_price})

                if trend_data:
                    trend_df = pd.DataFrame(trend_data)
                    trend_df = trend_df.set_index('Date')

                    if chart_type_individual == 'Bar Chart':
                        st.bar_chart(trend_df['Predicted Price'])
                    else: # Line Chart
                        st.line_chart(trend_df['Predicted Price'])

                    st.markdown(f"The graph above shows the predicted price trend for {selected_vegetable} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")

                    min_price = trend_df['Predicted Price'].min()
                    max_price = trend_df['Predicted Price'].max()
                    st.markdown(f"**Min Predicted Price:** ₹{min_price:.2f} | **Max Predicted Price:** ₹{max_price:.2f}")
                else:
                    st.warning("Could not generate price trend data. Please ensure the model files are correctly loaded.")

            else:
                st.error("Could not predict the price. Please check the inputs and ensure the model files are correctly loaded.")
    else:
        st.warning("Please select both a date and a vegetable to predict the price.")

st.markdown("---")

# --- Seasonal Price Overview (Moved Section to be after Individual Price Analysis) ---
st.header("🌦️ SEASONAL PRICE OVERVIEW")
st.markdown("See how the predicted price of a vegetable varies across different seasons.")

selected_vegetable_season = st.selectbox(
    "Select a Vegetable for Seasonal Analysis",
    available_vegetables,
    key='season_veg_select'
)

# Define representative dates for seasons (adjust as per typical regional seasons)
seasonal_dates_map = {
    "Winter (Dec-Feb)": datetime.date(2024, 1, 15),  # Mid-January
    "Spring (Mar-May)": datetime.date(2024, 4, 15),  # Mid-April
    "Monsoon (Jun-Sep)": datetime.date(2024, 7, 15), # Mid-July
    "Post-Monsoon (Oct-Nov)": datetime.date(2024, 10, 15) # Mid-October
}

# New: Select a specific season for prediction
selected_season_for_prediction = st.selectbox(
    "Select a Specific Season to Predict For",
    ["Select All Seasons"] + list(seasonal_dates_map.keys()),
    key='specific_season_select'
)

col_season_btn1, col_season_btn2 = st.columns(2)

with col_season_btn1:
    if st.button("Show Overall Seasonal Trend", help="Click to see the predicted price for the selected vegetable across all defined seasons."):
        if selected_vegetable_season:
            with st.spinner(f"Calculating overall seasonal prices for {selected_vegetable_season}..."):
                seasonal_price_data = []
                for season, date_obj in seasonal_dates_map.items():
                    price = predict_price(date_obj.strftime('%Y-%m-%d'), selected_vegetable_season)
                    if price != -1.0:
                        seasonal_price_data.append({'Season': season, 'Predicted Price': price})

                if seasonal_price_data:
                    seasonal_df = pd.DataFrame(seasonal_price_data)
                    st.subheader(f"Predicted Seasonal Prices for {selected_vegetable_season}")
                    st.bar_chart(seasonal_df.set_index('Season'))
                    st.markdown("The bar chart above shows the predicted average price for the selected vegetable in different seasons.")
                else:
                    st.warning("Could not generate seasonal price data. Please ensure the model is loaded and the vegetable is valid.")
        else:
            st.warning("Please select a vegetable for seasonal analysis.")

with col_season_btn2:
    if st.button("Predict Price for Selected Season", help="Click to get the predicted price for the selected vegetable in the chosen season."):
        if selected_vegetable_season and selected_season_for_prediction != "Select All Seasons":
            with st.spinner(f"Predicting price for {selected_vegetable_season} in {selected_season_for_prediction}..."):
                representative_date = seasonal_dates_map[selected_season_for_prediction]
                predicted_price_season = predict_price(representative_date.strftime('%Y-%m-%d'), selected_vegetable_season)

                if predicted_price_season != -1.0:
                    st.success(f"The predicted price for **{selected_vegetable_season}** in **{selected_season_for_prediction}** (around {representative_date.strftime('%B %d')}) is approximately: **₹{predicted_price_season:.2f}**")
                else:
                    st.error(f"Could not predict price for {selected_vegetable_season} in {selected_season_for_prediction}. Please check inputs.")
        elif selected_season_for_prediction == "Select All Seasons":
            st.info("Please use 'Show Overall Seasonal Trend' button to view all seasons, or select a specific season above.")
        else:
            st.warning("Please select a vegetable and a specific season.")

st.markdown("---")

# --- Comparative Price Analysis ---
st.header("🆚 Comparative Price Analysis")
st.markdown("Compare the predicted price trends of multiple vegetables over a selected period.")

col3, col4 = st.columns(2)

with col3:
    comp_start_date = st.date_input("Trend Start Date", value=datetime.date.today() - datetime.timedelta(days=30), key='comp_start_date')
with col4:
    comp_end_date = st.date_input("Trend End Date", value=datetime.date.today() + datetime.timedelta(days=30), key='comp_end_date')

selected_vegetables_comp = st.multiselect(
    "Select Vegetables for Comparison",
    available_vegetables,
    default=available_vegetables[:2] if len(available_vegetables) >= 2 else available_vegetables,
    key='multi_veg_select'
)

# Chart type selection for comparative analysis
chart_type_comparative = st.radio(
    "Select Chart Type for Comparative Analysis",
    ('Bar Chart', 'Line Chart'),
    key='chart_type_comparative'
)

if st.button("Generate Comparative Trend", help="Click to compare trends of selected vegetables."):
    if comp_start_date and comp_end_date and selected_vegetables_comp:
        if comp_start_date >= comp_end_date:
            st.error("Trend Start Date must be before Trend End Date.")
        else:
            with st.spinner("Generating comparative trends..."):
                comp_date_range = pd.date_range(start=comp_start_date, end=comp_end_date, freq='D')
                comparative_df = pd.DataFrame({'Date': comp_date_range})
                comparative_df = comparative_df.set_index('Date')

                for veg in selected_vegetables_comp:
                    veg_prices = []
                    for dt in comp_date_range:
                        daily_price = predict_price(dt.strftime('%Y-%m-%d'), veg)
                        veg_prices.append(daily_price if daily_price != -1.0 else None) # Use None for failed predictions

                    comparative_df[veg] = veg_prices

                if not comparative_df.empty:
                    st.subheader("Comparative Price Trends")
                    if chart_type_comparative == 'Bar Chart':
                        st.bar_chart(comparative_df)
                    else: # Line Chart
                        st.line_chart(comparative_df)
                    st.markdown(f"The graph above shows the predicted price trends for {', '.join(selected_vegetables_comp)} from {comp_start_date.strftime('%Y-%m-%d')} to {comp_end_date.strftime('%Y-%m-%d')}.")
                else:
                    st.warning("Could not generate comparative trend data. Please ensure the model files are correctly loaded and selected vegetables are valid.")
    else:
        st.warning("Please select start/end dates and at least one vegetable for comparison.")

st.markdown("---")

# --- Price Distribution Analysis (New Section for Pie Chart) ---
st.header("💵 Price Distribution Analysis (Pie Chart)")
st.markdown("Visualize the proportional predicted prices of selected vegetables on a specific date.")

pie_date = st.date_input("Select Date for Distribution", value=datetime.date.today(), key='pie_date')
selected_vegetables_pie = st.multiselect(
    "Select Vegetables for Pie Chart",
    available_vegetables,
    default=available_vegetables[:3] if len(available_vegetables) >= 3 else available_vegetables,
    key='pie_veg_select'
)

if st.button("Generate Price Distribution Chart", help="Click to see the price distribution of selected vegetables on the chosen date."):
    if pie_date and selected_vegetables_pie:
        with st.spinner(f"Generating price distribution for {pie_date.strftime('%Y-%m-%d')}..."):
            distribution_data = []
            for veg in selected_vegetables_pie:
                price = predict_price(pie_date.strftime('%Y-%m-%d'), veg)
                if price != -1.0:
                    distribution_data.append({'Vegetable': veg, 'Predicted Price': price})

            if distribution_data:
                distribution_df = pd.DataFrame(distribution_data)

                # Create the pie chart using Altair
                chart = alt.Chart(distribution_df).mark_arc().encode(
                    theta=alt.Theta(field="Predicted Price", type="quantitative"),
                    color=alt.Color(field="Vegetable", type="nominal", title="Vegetable"),
                    order=alt.Order("Predicted Price", sort="descending"),
                    tooltip=['Vegetable', 'Predicted Price']
                ).properties(
                    title=f"Price Distribution on {pie_date.strftime('%B %d, %Y')}"
                )
                st.altair_chart(chart, use_container_width=True)
                st.markdown("The pie chart above shows the predicted price distribution of the selected vegetables on the chosen date.")
            else:
                st.warning("Could not generate price distribution data. Please ensure valid vegetables are selected and the model is loaded.")
    else:
        st.warning("Please select a date and at least one vegetable for the pie chart.")

st.markdown("---")

# --- Prediction History in Sidebar ---
st.sidebar.header("Prediction History")
if st.session_state.prediction_history:
    # Display history in reverse chronological order
    for entry in reversed(st.session_state.prediction_history):
        st.sidebar.markdown(f"- **{entry['Vegetable']}** on {entry['Date']}: {entry['Predicted Price']}")
    if st.sidebar.button("Clear History"):
        st.session_state.prediction_history = []
        st.experimental_rerun() # Rerun to clear the display
else:
    st.sidebar.info("No predictions made yet.")

st.markdown("STAY HEALTHY AND HAPPY")
