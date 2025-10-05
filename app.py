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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Mental Health Dashboard", layout="wide")
st.title("Mental Health Analysis Dashboard")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("mental_health_lifestyle.csv")
        return df
    except:
        # إنشاء بيانات نموذجية بسيطة إذا لم يوجد الملف
        st.info("Using sample data - upload 'mental_health_lifestyle.csv' for your own data")
        np.random.seed(42)
        n_samples = 200
        data = {
            'age': np.random.randint(18, 65, n_samples),
            'stress_level': np.random.randint(1, 10, n_samples),
            'sleep_hours': np.random.randint(3, 9, n_samples),
            'physical_activity': np.random.randint(0, 7, n_samples),
            'mental_health_score': np.random.randint(1, 10, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples)
        }
        return pd.DataFrame(data)

df = load_data()

st.sidebar.title("Controls")
if st.sidebar.checkbox("Show data sample"):
    st.dataframe(df.head(5))
st.sidebar.write(f"Data: {df.shape[0]} rows, {df.shape[1]} columns")

st.subheader("Target Distribution")
target_col = st.selectbox("Select target column", df.columns, index=len(df.columns)-1)
fig = px.histogram(df, x=target_col, title=f"Distribution of {target_col}")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Correlation Analysis")
numeric_df = df.select_dtypes(include=['float64', 'int64'])
if not numeric_df.empty:
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(8,6))
    
    cmap = sns.light_palette("#8B93AF", as_cmap=True, reverse=False)
    
    sns.heatmap(corr, cmap=cmap, center=0, square=True, linewidths=0.5,
                annot=True, fmt=".2f", cbar_kws={"shrink": 0.8})
    
    plt.title("Feature Correlations", pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    st.pyplot(fig, use_container_width=True)

df_clean = df.dropna()
label_cols = df_clean.select_dtypes(include=['object']).columns
if len(label_cols) > 0:
    encoder = LabelEncoder()
    for col in label_cols:
        df_clean[col] = encoder.fit_transform(df_clean[col])

if not df_clean.empty and target_col in df_clean.columns:
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    if y.nunique() > 2:
        y = (y > y.median()).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.header("Model Training")
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    selected_models = st.multiselect(
        "Select models to train", 
        list(models.keys()), 
        default=["Random Forest", "Logistic Regression"]
    )

    if st.button("Train Models") and selected_models:
        results = {}
        for name in selected_models:
            model = models[name]
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results[name] = acc

        result_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
        result_df = result_df.sort_values("Accuracy", ascending=False)
        
        st.subheader("Model Performance")
        fig = px.bar(result_df, x="Model", y="Accuracy", 
                    color="Accuracy", text_auto=".2f",
                    color_continuous_scale="viridis")
        st.plotly_chart(fig, use_container_width=True)

        best_model_name = max(results, key=results.get)
        st.success(f"Best Model: {best_model_name} (Accuracy: {results[best_model_name]:.2f})")


st.markdown("---")
st.markdown("Simplified Mental Health Analysis Dashboard")






