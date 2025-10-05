import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from textblob import TextBlob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

st.set_page_config(page_title="AI Mental Health System", layout="wide")
st.title("AI-Powered Mental Health & Lifestyle Intelligence System")

@st.cache_data
def load_data():
    df = pd.read_csv("mental_health_lifestyle.csv")
    return df

df = load_data()

# Data Overview
st.subheader("Data Overview")
rows = st.slider("Number of rows to display:", 5, 100, 10)
st.dataframe(df.head(rows))
st.metric("Number of rows", df.shape[0])
st.metric("Number of columns", df.shape[1])

if st.checkbox("Show descriptive statistics"):
    st.write(df.describe(include='all'))

if st.checkbox("Show missing values"):
    st.write(df.isnull().sum())

# Column Distribution
st.subheader("Column Distribution")
col = st.selectbox("Select a column to visualize:", df.columns)
fig = px.histogram(df, x=col, color_discrete_sequence=["#0083B8"], title=f"Distribution of {col}")
st.plotly_chart(fig, use_container_width=True)

# Data Preparation
df = df.dropna()
label_cols = df.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for c in label_cols:
    df[c] = encoder.fit_transform(df[c])

target_col = st.selectbox("Select target column:", df.columns)
X = df.drop(columns=[target_col])
y = df[target_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Correlation Matrix (Black/Grey)
if st.checkbox("Show Correlation Matrix"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap="Greys", annot=True, fmt=".2f", annot_kws={"size":12}, ax=ax, cbar_kws={"shrink":0.8})
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    st.pyplot(fig)

# Train Models Individually (Optional for free version)
st.subheader("Train Models Individually (Optional for Free Version)")
st.info("اختيار أي نموذج للتدريب يزيد من استخدام الموارد. اختر بعناية في النسخة المجانية.")

if st.checkbox("Train Random Forest"):
    model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)  # أقل عدد لتخفيف الحمل
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.success(f"Random Forest Accuracy: {acc:.2f}")

if st.checkbox("Train Gradient Boosting"):
    model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.success(f"Gradient Boosting Accuracy: {acc:.2f}")

if st.checkbox("Train AdaBoost"):
    model = AdaBoostClassifier(n_estimators=50)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.success(f"AdaBoost Accuracy: {acc:.2f}")

if st.checkbox("Train Logistic Regression"):
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.success(f"Logistic Regression Accuracy: {acc:.2f}")

if st.checkbox("Train SVM"):
    model = SVC(kernel='rbf', C=1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.success(f"SVM Accuracy: {acc:.2f}")

if st.checkbox("Train KNN"):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.success(f"KNN Accuracy: {acc:.2f}")

if st.checkbox("Train Naive Bayes"):
    model = GaussianNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.success(f"Naive Bayes Accuracy: {acc:.2f}")

# Optional Neural Network
st.subheader("Optional Neural Network")
if st.checkbox("Train Neural Network"):
    epochs = st.slider("Select epochs", 1, 5, 3)  # أقل epochs لتقليل الحمل
    scaler_nn = StandardScaler()
    X_train_scaled = scaler_nn.fit_transform(X_train)
    X_test_scaled = scaler_nn.transform(X_test)

    nn_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = nn_model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)
    test_loss, test_acc = nn_model.evaluate(X_test_scaled, y_test, verbose=0)
    st.success(f"Neural Network Test Accuracy: {test_acc:.2f}")

    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].plot(history.history['accuracy'], label='Train Accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    ax[0].legend()
    ax[0].set_title("Accuracy Over Epochs")

    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Val Loss')
    ax[1].legend()
    ax[1].set_title("Loss Over Epochs")
    st.pyplot(fig)

# Optional KMeans + PCA
st.subheader("Optional KMeans + PCA Clustering")
if st.checkbox("Run KMeans Clustering"):
    num_clusters = st.slider("Select number of clusters", 2, 10, 3)
    scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled)
    df['Cluster'] = labels
    pca = PCA(2)
    components = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(data=components, columns=["PC1","PC2"])
    pca_df['Cluster'] = labels
    fig = px.scatter(pca_df, x="PC1", y="PC2", color=pca_df["Cluster"].astype(str),
                     color_discrete_sequence=px.colors.sequential.Black,
                     title="KMeans Clustering with PCA")
    st.plotly_chart(fig, use_container_width=True)

# Sentiment Analysis
st.subheader("Sentiment Analysis")
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
st.markdown("Developed for AI Mental Health Research Dashboard")
