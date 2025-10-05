import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from textblob import TextBlob
import nltk
nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="Lite AI Mental Health Dashboard", layout="wide")
st.title("Lite AI Mental Health and Lifestyle Dashboard")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("mental_health_lifestyle.csv").sample(n=1000, random_state=42)
        return df
    except:
        st.error("CSV file not found. Make sure 'mental_health_lifestyle.csv' is in your folder.")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

st.sidebar.title("Dashboard Control")
if st.sidebar.checkbox("Show first 5 rows"):
    st.dataframe(df.head(5))
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
    fig, ax = plt.subplots(figsize=(10,7))
    sns.heatmap(corr, cmap="Purples", annot=True, fmt=".2f", square=False, linewidths=0.5, cbar_kws={"shrink":0.8})
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
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

st.header("Model Training and Evaluation")

models = {
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),  # أخف
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500)
}

selected_models = st.multiselect(
    "Select models to train", list(models.keys()), default=["Random Forest", "Logistic Regression"]
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
    st.table(result_df)

    best_model_name = max(results, key=results.get)
    st.success(f"Best Model: {best_model_name} | Accuracy: {results[best_model_name]:.2f}")

if st.checkbox("Show Confusion Matrix for Best Model"):
        best_model = models[best_model_name]
        preds = best_model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax)
        st.pyplot(fig)

st.header("Sentiment Analysis")
text_input = st.text_area("Enter text to analyze sentiment:")
if st.button("Analyze Sentiment"):
    if text_input.strip():
        sentiment = TextBlob(text_input).sentiment.polarity
        if sentiment > 0:
            st.success(f"Positive Sentiment ({sentiment:.2f})")
        elif sentiment < 0:
            st.error(f"Negative Sentiment ({sentiment:.2f})")
        else:
            st.warning("Neutral Sentiment (0.00)")
    else:
        st.warning("Please enter valid text.")

st.markdown("---")
st.markdown("Lite Version for Free Users")
