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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from textblob import TextBlob
import nltk
nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="AI Mental Health Dashboard", layout="wide")
st.title("AI Mental Health Dashboard - Light Version")

# Load Data (sample if large)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("mental_health_lifestyle.csv")
        if df.shape[0] > 5000:
            df = df.sample(5000, random_state=42).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"CSV file not found or could not be loaded: {e}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

st.subheader("Dataset Overview")
st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
st.dataframe(df.head(10))

# Preprocessing
df = df.dropna()
label_cols = df.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for col in label_cols:
    df[col] = encoder.fit_transform(df[col])

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
target_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]

X = df.drop(columns=[target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
numeric_df = df.select_dtypes(include=['float64', 'int64'])
if numeric_df.shape[1] > 0:
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(7,5))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    st.pyplot(fig)

# Model Training
st.header("Model Training and Evaluation")
models = {
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = accuracy_score(y_test, preds)

result_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False)
st.subheader("Model Accuracy Comparison")
fig = px.bar(result_df, x="Model", y="Accuracy", color="Accuracy", text_auto=".2f")
st.plotly_chart(fig, use_container_width=True)

best_model_name = max(results, key=results.get)
st.success(f"Best Model: {best_model_name} | Accuracy: {results[best_model_name]:.2f}")

# Confusion Matrix
st.subheader("Confusion Matrix")
best_model = models[best_model_name]
preds = best_model.predict(X_test)
cm = confusion_matrix(y_test, preds)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# Neural Network
st.header("Neural Network Training")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
nn_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = nn_model.fit(X_train_scaled, y_train, epochs=3, batch_size=32, validation_split=0.2, verbose=0)
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

# KMeans + PCA
st.header("KMeans + PCA Clustering")
scaled = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(scaled)
pca = PCA(2)
components = pca.fit_transform(scaled)
pca_df = pd.DataFrame(data=components, columns=["PC1","PC2"])
pca_df['Cluster'] = labels
fig = px.scatter(pca_df, x="PC1", y="PC2", color=pca_df["Cluster"].astype(str),
                 title="KMeans Clustering with PCA")
st.plotly_chart(fig, use_container_width=True)

# Sentiment Analysis
st.header("Sentiment Analysis")
text_input = st.text_area("Enter text to analyze sentiment:")
if text_input.strip():
    sentiment = TextBlob(text_input).sentiment.polarity
    if sentiment > 0:
        st.success(f"Positive Sentiment ({sentiment:.2f})")
    elif sentiment < 0:
        st.error(f"Negative Sentiment ({sentiment:.2f})")
    else:
        st.warning("Neutral Sentiment (0.00)")

st.markdown("---")
st.markdown("Developed for AI Mental Health Research Dashboard")


