
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ============================================================
# SPORTS PERFORMANCE ANALYTICS & ATHLETE SEGMENTATION
# Standard Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Sports Performance Analytics",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom styling
# -----------------------------
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #12355b, #1f6f8b);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
    }
    .hero p {
        margin: 0.35rem 0 0 0;
        opacity: 0.9;
    }
    .metric-card {
        background: white;
        padding: 1rem 1.1rem;
        border-radius: 14px;
        border: 1px solid #e6eaf0;
        box-shadow: 0 2px 10px rgba(20, 40, 70, 0.05);
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #18324a;
        margin-top: 0.5rem;
        margin-bottom: 0.7rem;
    }
    .result-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 3px 12px rgba(20, 40, 70, 0.06);
    }
    .small-note {
        color: #687789;
        font-size: 0.88rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e6eaf0;
        padding: 0.8rem;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

DATA_PATH = "athletes_analytics.csv"

FEATURES = [
    "Age",
    "Weight_kg",
    "Training_Experience_Years",
    "Training_Hours_Per_Week",
    "VO2_Max_ml_kg_min",
    "Sprint_Speed_m_s",
    "Reaction_Time_ms",
    "Strength_Test_Score",
    "Resting_Heart_Rate_bpm",
    "Sleep_Hours_Per_Day",
    "Training_Days_Per_Week",
    "Recovery_Heart_Rate_bpm"
]

RF_FEATURES = FEATURES + ["Cluster"]

LR_FEATURES = [
    "Age",
    "Weight_kg",
    "Training_Days_Per_Week",
    "Sleep_Hours_Per_Day",
    "Cluster",
    "Predicted_Overall_Performance_%"
]

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    for col in FEATURES + ["Medal_Chance%", "Overall_Performance_Score_%"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=FEATURES + [
        "Medal_Chance%", "Overall_Performance_Score_%"
    ]).reset_index(drop=True)

@st.cache_resource
def train_models(df):
    data = df.copy()

    # ---------- K-Means ----------
    cluster_scaler = MinMaxScaler()
    X_cluster = cluster_scaler.fit_transform(data[FEATURES])

    k_values = range(2, 7)
    sil_scores = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X_cluster)
        sil_scores.append(silhouette_score(X_cluster, labels))

    best_k = list(k_values)[int(np.argmax(sil_scores))]

    kmeans_model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=20
    )
    data["Cluster"] = kmeans_model.fit_predict(X_cluster)

    cluster_summary = data.groupby("Cluster")[
        FEATURES + ["Overall_Performance_Score_%", "Medal_Chance%"]
    ].mean()

    cluster_order = (
        cluster_summary["Overall_Performance_Score_%"]
        .sort_values(ascending=False)
        .index
    )

    if best_k == 3:
        names = ["High Performer", "Intermediate", "Developing"]
        cluster_labels = {
            cluster: names[i] for i, cluster in enumerate(cluster_order)
        }
    else:
        cluster_labels = {
            cluster: f"Performance Group {i + 1}"
            for i, cluster in enumerate(cluster_order)
        }

    data["Performance_Level"] = data["Cluster"].map(cluster_labels)

    # ---------- Random Forest ----------
    X_rf = data[RF_FEATURES]
    y_rf = data["Overall_Performance_Score_%"]

    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
        X_rf, y_rf, test_size=0.20, random_state=42
    )

    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_rf, y_train_rf)

    rf_pred = np.clip(rf_model.predict(X_test_rf), 0, 100)

    rf_metrics = {
        "MAE": mean_absolute_error(y_test_rf, rf_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test_rf, rf_pred)),
        "R2": r2_score(y_test_rf, rf_pred)
    }

    data["Predicted_Overall_Performance_%"] = np.clip(
        rf_model.predict(data[RF_FEATURES]), 0, 100
    ).round(2)

    # ---------- Linear Regression ----------
    X_lr = data[LR_FEATURES]
    y_lr = data["Medal_Chance%"]

    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
        X_lr, y_lr, test_size=0.20, random_state=42
    )

    lr_model = LinearRegression()
    lr_model.fit(X_train_lr, y_train_lr)

    lr_pred = np.clip(lr_model.predict(X_test_lr), 0, 100)

    lr_metrics = {
        "MAE": mean_absolute_error(y_test_lr, lr_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test_lr, lr_pred)),
        "R2": r2_score(y_test_lr, lr_pred)
    }

    data["Predicted_Medal_Chance%"] = np.clip(
        lr_model.predict(data[LR_FEATURES]), 0, 100
    ).round(2)

    medians = data[FEATURES].median()

    def recommendation(row):
        if row["Predicted_Overall_Performance_%"] >= 75:
            return "Good performance - maintain current training."

        rec = []

        if row["Training_Hours_Per_Week"] < medians["Training_Hours_Per_Week"]:
            rec.append("Improve training consistency.")
        if row["VO2_Max_ml_kg_min"] < medians["VO2_Max_ml_kg_min"]:
            rec.append("Improve aerobic fitness.")
        if row["Strength_Test_Score"] < medians["Strength_Test_Score"]:
            rec.append("Improve strength training.")
        if row["Sprint_Speed_m_s"] < medians["Sprint_Speed_m_s"]:
            rec.append("Improve speed training.")
        if row["Reaction_Time_ms"] > medians["Reaction_Time_ms"]:
            rec.append("Practice reaction-time drills.")
        if row["Sleep_Hours_Per_Day"] < medians["Sleep_Hours_Per_Day"]:
            rec.append("Improve sleep and recovery.")
        if row["Training_Days_Per_Week"] < medians["Training_Days_Per_Week"]:
            rec.append("Increase training consistency.")
        if row["Resting_Heart_Rate_bpm"] > medians["Resting_Heart_Rate_bpm"]:
            rec.append("Focus on cardiovascular fitness.")
        if row["Recovery_Heart_Rate_bpm"] > medians["Recovery_Heart_Rate_bpm"]:
            rec.append("Improve conditioning and recovery.")

        if not rec:
            return "Performance is below 75%. Maintain balanced training and monitor progress."

        return " ".join(rec)

    data["Recommendation"] = data.apply(recommendation, axis=1)

    importance = pd.DataFrame({
        "Feature": RF_FEATURES,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return {
        "data": data,
        "cluster_scaler": cluster_scaler,
        "kmeans": kmeans_model,
        "cluster_labels": cluster_labels,
        "best_k": best_k,
        "k_values": list(k_values),
        "sil_scores": sil_scores,
        "rf": rf_model,
        "rf_metrics": rf_metrics,
        "lr": lr_model,
        "lr_metrics": lr_metrics,
        "importance": importance,
        "medians": medians
    }

# -----------------------------
# Load + train
# -----------------------------
try:
    raw_df = load_data()
    models = train_models(raw_df)
except Exception as e:
    st.error(f"Unable to load the project data: {e}")
    st.stop()

data = models["data"]

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>🏅 Sports Performance Analytics & Athlete Segmentation</h1>
    <p>Analyze athlete characteristics, segment similar athletes, predict performance, estimate medal chance, and generate recommendations.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio(
        "Go to",
        [
            "Dashboard",
            "Athlete Prediction",
            "Athlete Explorer",
            "Model Insights"
        ]
    )

    st.divider()
    st.markdown("### Project Pipeline")
    st.write("1. K-Means → Athlete Segmentation")
    st.write("2. Random Forest → Performance Prediction")
    st.write("3. Linear Regression → Medal Chance")
    st.write("4. Recommendation → Training Suggestions")

    st.divider()
    st.caption(f"Dataset: {len(data):,} athletes")
    st.caption(f"Selected clusters: K = {models['best_k']}")

# -----------------------------
# Dashboard
# -----------------------------
if page == "Dashboard":
    st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Athletes", f"{len(data):,}")
    c2.metric("Clusters", models["best_k"])
    c3.metric("RF R²", f"{models['rf_metrics']['R2']:.3f}")
    c4.metric("LR R²", f"{models['lr_metrics']['R2']:.3f}")

    st.write("")

    left, right = st.columns(2)

    with left:
        st.markdown("### Athlete Segmentation")
        counts = data["Performance_Level"].value_counts()

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(counts.index, counts.values)
        ax.set_ylabel("Number of Athletes")
        ax.set_xlabel("Performance Group")
        ax.set_title("Athlete Distribution by Segment")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown("### Average Predicted Performance")
        avg_perf = data.groupby("Performance_Level")[
            "Predicted_Overall_Performance_%"
        ].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(avg_perf.index, avg_perf.values)
        ax.set_ylabel("Predicted Performance (%)")
        ax.set_xlabel("Performance Group")
        ax.set_title("Predicted Performance by Segment")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("### Key Athlete Indicators")
    indicators = pd.DataFrame({
        "Indicator": [
            "Average VO₂ Max",
            "Average Training Hours / Week",
            "Average Sleep / Day",
            "Average Strength Score",
            "Average Sprint Speed"
        ],
        "Value": [
            f"{data['VO2_Max_ml_kg_min'].mean():.2f}",
            f"{data['Training_Hours_Per_Week'].mean():.2f}",
            f"{data['Sleep_Hours_Per_Day'].mean():.2f}",
            f"{data['Strength_Test_Score'].mean():.2f}",
            f"{data['Sprint_Speed_m_s'].mean():.2f}"
        ]
    })
    st.dataframe(indicators, use_container_width=True, hide_index=True)

# -----------------------------
# Athlete Prediction
# -----------------------------
elif page == "Athlete Prediction":
    st.markdown('<div class="section-title">New Athlete Prediction</div>', unsafe_allow_html=True)
    st.caption("Enter athlete information to receive segmentation, performance prediction, medal chance, and recommendations.")

    with st.form("athlete_form"):
        st.markdown("#### Athlete Profile")
        p1, p2, p3 = st.columns(3)

        with p1:
            athlete_id = st.text_input("Athlete ID", "NEW001")
            age = st.number_input("Age", 18.0, 30.0, 24.0, 1.0)
            weight = st.number_input("Weight (kg)", 50.0, 85.3, 68.0, 0.1)
            experience = st.number_input("Training Experience (years)", 1.0, 8.0, 4.0, 1.0)

        with p2:
            training_hours = st.number_input("Training Hours / Week", 4.1, 13.8, 8.5, 0.1)
            vo2 = st.number_input("VO₂ Max (ml/kg/min)", 30.3, 65.0, 48.0, 0.1)
            sprint = st.number_input("Sprint Speed (m/s)", 6.91, 10.0, 8.5, 0.01)
            reaction = st.number_input("Reaction Time (ms)", 178.0, 275.0, 220.0, 1.0)

        with p3:
            strength = st.number_input("Strength Test Score", 51.9, 110.0, 83.0, 0.1)
            resting_hr = st.number_input("Resting Heart Rate (bpm)", 53.0, 82.0, 67.0, 1.0)
            sleep = st.number_input("Sleep Hours / Day", 5.5, 9.0, 7.3, 0.1)
            training_days = st.number_input("Training Days / Week", 3.0, 6.0, 4.0, 1.0)
            recovery_hr = st.number_input("Recovery Heart Rate (bpm)", 68.0, 117.0, 93.0, 1.0)

        submitted = st.form_submit_button("🔍 Analyze Athlete", use_container_width=True)

    if submitted:
        new = pd.DataFrame([{
            "Age": age,
            "Weight_kg": weight,
            "Training_Experience_Years": experience,
            "Training_Hours_Per_Week": training_hours,
            "VO2_Max_ml_kg_min": vo2,
            "Sprint_Speed_m_s": sprint,
            "Reaction_Time_ms": reaction,
            "Strength_Test_Score": strength,
            "Resting_Heart_Rate_bpm": resting_hr,
            "Sleep_Hours_Per_Day": sleep,
            "Training_Days_Per_Week": training_days,
            "Recovery_Heart_Rate_bpm": recovery_hr
        }])

        # K-Means uses scaling
        new_scaled = models["cluster_scaler"].transform(new[FEATURES])
        cluster = int(models["kmeans"].predict(new_scaled)[0])
        level = models["cluster_labels"][cluster]

        # Random Forest does NOT use scaling
        new_rf = new.copy()
        new_rf["Cluster"] = cluster
        performance = float(
            np.clip(models["rf"].predict(new_rf[RF_FEATURES])[0], 0, 100)
        )

        new_lr = new.copy()
        new_lr["Cluster"] = cluster
        new_lr["Predicted_Overall_Performance_%"] = performance

        medal = float(
            np.clip(models["lr"].predict(new_lr[LR_FEATURES])[0], 0, 100)
        )

        result_row = new.iloc[0].copy()
        result_row["Predicted_Overall_Performance_%"] = performance

        if performance >= 75:
            rec = "Good performance - maintain current training."
        else:
            recs = []
            med = models["medians"]
            if training_hours < med["Training_Hours_Per_Week"]:
                recs.append("Improve training consistency.")
            if vo2 < med["VO2_Max_ml_kg_min"]:
                recs.append("Improve aerobic fitness.")
            if strength < med["Strength_Test_Score"]:
                recs.append("Improve strength training.")
            if sprint < med["Sprint_Speed_m_s"]:
                recs.append("Improve speed training.")
            if reaction > med["Reaction_Time_ms"]:
                recs.append("Practice reaction-time drills.")
            if sleep < med["Sleep_Hours_Per_Day"]:
                recs.append("Improve sleep and recovery.")
            if training_days < med["Training_Days_Per_Week"]:
                recs.append("Increase training consistency.")
            if resting_hr > med["Resting_Heart_Rate_bpm"]:
                recs.append("Focus on cardiovascular fitness.")
            if recovery_hr > med["Recovery_Heart_Rate_bpm"]:
                recs.append("Improve conditioning and recovery.")
            rec = " ".join(recs) if recs else "Maintain balanced training and monitor progress."

        st.markdown("### Prediction Result")
        r1, r2, r3 = st.columns(3)
        r1.metric("Performance Group", level)
        r2.metric("Overall Performance", f"{performance:.2f}%")
        r3.metric("Medal Chance", f"{medal:.2f}%")

        st.progress(min(performance / 100, 1.0), text=f"Overall Performance: {performance:.2f}%")

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"**Athlete:** {athlete_id}")
        st.markdown(f"**Cluster:** {cluster}")
        st.markdown(f"**Recommendation:** {rec}")
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Athlete Explorer
# -----------------------------
elif page == "Athlete Explorer":
    st.markdown('<div class="section-title">Athlete Explorer</div>', unsafe_allow_html=True)

    groups = ["All"] + sorted(data["Performance_Level"].unique().tolist())
    selected_group = st.selectbox("Filter by Performance Group", groups)

    view = data.copy()
    if selected_group != "All":
        view = view[view["Performance_Level"] == selected_group]

    search_id = st.text_input("Search Athlete ID", "")
    if search_id:
        view = view[
            view["Athlete_ID"].astype(str).str.contains(
                search_id, case=False, na=False
            )
        ]

    display_cols = [
        "Athlete_ID",
        "Performance_Level",
        "Predicted_Overall_Performance_%",
        "Predicted_Medal_Chance%",
        "Training_Hours_Per_Week",
        "VO2_Max_ml_kg_min",
        "Strength_Test_Score",
        "Sleep_Hours_Per_Day",
        "Recommendation"
    ]

    st.dataframe(
        view[display_cols].sort_values(
            "Predicted_Overall_Performance_%",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

# -----------------------------
# Model Insights
# -----------------------------
else:
    st.markdown('<div class="section-title">Model Insights</div>', unsafe_allow_html=True)

    a, b = st.columns(2)

    with a:
        st.markdown("### K-Means: Cluster Selection")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(models["k_values"], models["sil_scores"], marker="o")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Score for K Selection")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.info(f"Selected K = {models['best_k']} based on the highest silhouette score.")

    with b:
        st.markdown("### Random Forest: Feature Importance")
        imp = models["importance"].head(10).sort_values("Importance")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh(imp["Feature"], imp["Importance"])
        ax.set_xlabel("Importance")
        ax.set_title("Top Features for Performance Prediction")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("### Model Evaluation")
    eval_df = pd.DataFrame({
        "Model": ["Random Forest Regression", "Linear Regression"],
        "Target": [
            "Overall Performance",
            "Medal Chance"
        ],
        "MAE": [
            models["rf_metrics"]["MAE"],
            models["lr_metrics"]["MAE"]
        ],
        "RMSE": [
            models["rf_metrics"]["RMSE"],
            models["lr_metrics"]["RMSE"]
        ],
        "R²": [
            models["rf_metrics"]["R2"],
            models["lr_metrics"]["R2"]
        ]
    })
    st.dataframe(
        eval_df.style.format({
            "MAE": "{:.3f}",
            "RMSE": "{:.3f}",
            "R²": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Methodology Used")
    m1, m2, m3, m4 = st.columns(4)
    m1.info("K-Means\n\nAthlete Segmentation")
    m2.info("Random Forest\n\nPerformance Prediction")
    m3.info("Linear Regression\n\nMedal Chance Prediction")
    m4.info("Recommendation System\n\nTraining Suggestions")

st.divider()
st.caption("Sports Performance Analytics & Athlete Segmentation • Academic Project • Dataset marked Synthetic/Demo")
