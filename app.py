import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from textblob import TextBlob
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

st.set_page_config(page_title="AI Mental Health System", layout="wide")
st.title("AI-Powered Mental Health & Lifestyle Intelligence System")

@st.cache_data
def load_data():
    df = pd.read_csv("mental_health_lifestyle.csv")
    return df

df = load_data()

st.subheader("Data Overview")
rows = st.slider("Number of rows to display:", 5, 100, 10)
st.dataframe(df.head(rows))

st.metric("Number of rows", df.shape[0])
st.metric("Number of columns", df.shape[1])

if st.checkbox("Show descriptive statistics"):
    st.write(df.describe(include='all'))

if st.checkbox("Show missing values"):
    st.write(df.isnull().sum())

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

models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1),
    "AdaBoost": AdaBoostClassifier(n_estimators=200),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM": SVC(kernel='rbf', C=2),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB()
}

selected_models = st.multiselect("Select models to run:", list(models.keys()), default=["Random Forest", "Gradient Boosting", "AdaBoost"])
results = {}

if st.button("Train and Run Models"):
    progress = st.progress(0)
    for i, name in enumerate(selected_models):
        model = models[name]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cv = cross_val_score(model, X_scaled, y, cv=5).mean()
        results[name] = {"Accuracy": acc, "CV": cv}
        progress.progress((i+1)/len(selected_models))
        st.success(f"{name}: Accuracy = {acc*100:.2f}% | CV = {cv*100:.2f}%")

    results_df = pd.DataFrame(results).T
    st.write(results_df)
    fig = px.bar(results_df, x=results_df.index, y="Accuracy", title="Model Accuracy Comparison", color="Accuracy")
    st.plotly_chart(fig, use_container_width=True)

if st.checkbox("Show Correlation Matrix"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap="Greys", annot=True, fmt=".2f", annot_kws={"size":12}, ax=ax)
    st.pyplot(fig)

# Optional Neural Network
st.subheader("Optional Neural Network Training")
if st.checkbox("Train Neural Network"):
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
    history = nn_model.fit(X_train_scaled, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=0)
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
st.header("Optional KMeans + PCA Clustering")
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
    fig = px.scatter(pca_df, x="PC1", y="PC2", color=pca_df["Cluster"].astype(str), title="KMeans Clustering with PCA")
    st.plotly_chart(fig, use_container_width=True)

# Sentiment Analysis
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
st.markdown("Developed for AI Mental Health Research Dashboard")



