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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from textblob import TextBlob
import nltk
nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="AI Mental Health & Lifestyle Dashboard", layout="wide")
st.title("AI-Powered Mental Health and Lifestyle Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("mental_health_lifestyle.csv")
    return df

df = load_data()
df = df.dropna()

st.sidebar.title("Dashboard Settings")
if st.sidebar.checkbox("Show Sample Data"):
    st.dataframe(df.head(10))

st.sidebar.write("Rows:", df.shape[0])
st.sidebar.write("Columns:", df.shape[1])

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution of Selected Column")
    target_col = st.selectbox("Select Target Column", df.columns)
    fig = px.histogram(df, x=target_col, color=target_col, template="plotly_dark", title=f"Distribution of {target_col}")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_theme(style="white")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.8, "label": "Correlation Strength"}
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, weight='bold', pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    st.pyplot(fig, use_container_width=True)

label_cols = df.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for col in label_cols:
    df[col] = encoder.fit_transform(df[col])

target = target_col
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

st.header("Machine Learning Model Evaluation")

models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

selected_models = st.multiselect("Select Models to Train", list(models.keys()), default=["Random Forest", "Logistic Regression"])
results = {}

if st.button("Train Models"):
    progress = st.progress(0)
    for i, name in enumerate(selected_models):
        model = models[name]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        progress.progress((i + 1) / len(selected_models))
    result_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False)
    st.subheader("Model Accuracy Comparison")
    fig = px.bar(result_df, x="Model", y="Accuracy", color="Accuracy", text_auto=".2f", template="plotly_dark", title="Model Performance Comparison")
    st.plotly_chart(fig, use_container_width=True)
    best_model_name = max(results, key=results.get)
    st.success(f"Best Model: {best_model_name} with Accuracy {results[best_model_name]:.3f}")

if st.checkbox("Show Confusion Matrix of Best Model"):
    best_model = models[best_model_name]
    preds = best_model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
    st.pyplot(fig)

st.header("Neural Network Model")

if st.button("Train Neural Network"):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    nn_model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.25),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = nn_model.fit(X_train_scaled, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=0)
    test_loss, test_acc = nn_model.evaluate(X_test_scaled, y_test, verbose=0)
    st.success(f"Neural Network Test Accuracy: {test_acc:.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plt.style.use('dark_background')
    ax[0].plot(history.history['accuracy'], label='Train', color='cyan')
    ax[0].plot(history.history['val_accuracy'], label='Validation', color='orange')
    ax[0].set_title("Accuracy Over Epochs")
    ax[0].legend()
    ax[1].plot(history.history['loss'], label='Train', color='cyan')
    ax[1].plot(history.history['val_loss'], label='Validation', color='orange')
    ax[1].set_title("Loss Over Epochs")
    ax[1].legend()
    st.pyplot(fig)

st.header("Sentiment Analysis")

text_input = st.text_area("Enter a sentence to analyze its sentiment:")
if st.button("Analyze Sentiment"):
    if text_input.strip():
        sentiment = TextBlob(text_input).sentiment.polarity
        if sentiment > 0:
            st.success(f"Positive Sentiment ({sentiment:.2f})")
        elif sentiment < 0:
            st.error(f"Negative Sentiment ({sentiment:.2f})")
        else:
            st.warning(f"Neutral Sentiment ({sentiment:.2f})")
    else:
        st.warning("Please enter text.")

st.header("Data Clustering and Pattern Visualization")

num_clusters = st.slider("Number of Clusters", 2, 10, 3)
scaled = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(scaled)
df['Cluster'] = labels
pca = PCA(2)
components = pca.fit_transform(scaled)
pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
pca_df['Cluster'] = labels
fig = px.scatter(pca_df, x="PC1", y="PC2", color=pca_df["Cluster"].astype(str), template="plotly_dark", title="K-Means Clustering Visualization (PCA Reduced)")
st.plotly_chart(fig, use_container_width=True)

st.header("Make Predictions")

user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(f"Enter value for {col}", value=float(df[col].mean()))
user_df = pd.DataFrame([user_input])

if st.button("Predict Using Best Model"):
    best_model = models[max(results, key=results.get)]
    prediction = best_model.predict(user_df)[0]
    st.success(f"Predicted Class: {prediction}")

st.header("Insights and Feature Importance")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top Correlations")
    corr = df.corr()[target].sort_values(ascending=False).head(10)
    st.dataframe(corr)

with col2:
    st.subheader("Feature Importance (Random Forest)")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_}).sort_values(by='Importance', ascending=False)
    fig = px.bar(importances.head(10), x='Feature', y='Importance', template="plotly_dark", title='Top 10 Most Important Features')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("Developed for AI-Based Mental Health and Lifestyle Research Dashboard")










