import streamlit as st
import joblib
import pandas as pd

model = joblib.load("epl_winner_model.pkl")

st.set_page_config(page_title="EPL Match Win Predictor ⚽", layout="centered")

st.title("🏆 English Premier League Win Predictor")
st.markdown("Predict whether a team will **win** their next match based on basic stats!")
st.header("📋 Enter Match Details")

teams = [
    "Manchester City", "Chelsea", "Arsenal", "Tottenham Hotspur", "Manchester United",
    "West Ham United", "Wolverhampton Wanderers", "Newcastle United", "Leicester City",
    "Brighton and Hove Albion", "Brentford", "Southampton", "Crystal Palace",
    "Aston Villa", "Leeds United", "Burnley", "Everton", "Watford",
    "Norwich City", "Liverpool", "Fulham", "West Bromwich Albion", "Sheffield United"
]

venue = st.selectbox("Venue", ["Home", "Away"])
opponent_name = st.selectbox("Opponent Team", teams)
hour = st.slider("Match Hour (24h format)", 0, 23, 15)
day_code = st.slider("Day of Week (0 = Mon ... 6 = Sun)", 0, 6, 2)

opponent_code = teams.index(opponent_name)

input_data = pd.DataFrame({
    "venue": [venue],
    "opponent_code": [opponent_code],
    "hour": [hour],
    "day_code": [day_code]
})

if st.button("Predict Result"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"🎉 Prediction: **WIN vs {opponent_name}** (Confidence: {prob*100:.2f}%)")
    else:
        st.error(f"❌ Prediction: **NOT WIN vs {opponent_name}** (Confidence: {(1-prob)*100:.2f}%)")

st.markdown("---")
st.caption("⚽ Built with XGBoost + Streamlit by Krish Sharma")