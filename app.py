import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="AI Mental Health Lite Dashboard", layout="wide")
st.title("AI Mental Health Dashboard (Lite Version)")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("mental_health_lifestyle.csv")
        return df
    except:
        st.error("CSV file not found. تأكد من وجود 'mental_health_lifestyle.csv'")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

st.sidebar.title("Dashboard Control")
if st.sidebar.checkbox("Show first 10 rows"):
    st.dataframe(df.head(10))
st.sidebar.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Target Column Distribution")
    target_col = st.selectbox("Select Target Column", df.columns, index=0)
    fig = px.histogram(df, x=target_col, color=target_col, title=f"Distribution of {target_col}")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, cmap="Purples", annot=False, cbar=True, square=True, linewidths=0.5)
    st.pyplot(fig, use_container_width=True)

df = df.dropna()
label_cols = df.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for col in label_cols:
    df[col] = encoder.fit_transform(df[col])

target = target_col
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.header("Model Training (Lite)")

models = {
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500)
}

selected_models = st.multiselect(
    "Select models to train", list(models.keys()), default=["Random Forest"]
)

results = {}
if st.button("Train Selected Models"):
    for name in selected_models:
        model = models[name]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc

    result_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False)
    st.subheader("Model Accuracy Comparison")
    fig = px.bar(result_df, x="Model", y="Accuracy", color="Accuracy", text_auto=".2f")
    st.plotly_chart(fig, use_container_width=True)

    best_model_name = max(results, key=results.get)
    st.success(f"Best Model: {best_model_name} | Accuracy: {results[best_model_name]:.2f}")

if st.checkbox("Show Confusion Matrix for Best Model"):
    best_model = models[best_model_name]
    preds = best_model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False)
    st.pyplot(fig)

st.markdown("---")
st.markdown("Lite Version Developed for AI Mental Health Research")




