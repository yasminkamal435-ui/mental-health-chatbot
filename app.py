import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
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

# Reduce data size for free tier
df = df.sample(min(500, len(df)), random_state=42)  # take max 500 rows

# Data preparation
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

# ===== Confusion Matrix =====
with st.expander("Show Confusion Matrix for Random Forest"):
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
    rf_model.fit(X_train, y_train)
    preds = rf_model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size":12}, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Random Forest")
    st.pyplot(fig)

# ===== Neural Network =====
with st.expander("Optional Neural Network Training"):
    epochs = st.slider("Select epochs (reduce for free tier)", 1, 5, 3)
    scaler_nn = StandardScaler()
    X_train_scaled = scaler_nn.fit_transform(X_train)
    X_test_scaled = scaler_nn.transform(X_test)

    nn_model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = nn_model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)
    test_loss, test_acc = nn_model.evaluate(X_test_scaled, y_test, verbose=0)
    st.success(f"Neural Network Test Accuracy: {test_acc:.2f}")

    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    ax[0].plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
    ax[0].set_title("Accuracy Over Epochs")
    ax[0].legend()
    ax[0].grid(True)
    ax[1].plot(history.history['loss'], label='Train Loss', marker='o')
    ax[1].plot(history.history['val_loss'], label='Val Loss', marker='o')
    ax[1].set_title("Loss Over Epochs")
    ax[1].legend()
    ax[1].grid(True)
    st.pyplot(fig)

# ===== KMeans + PCA =====
with st.expander("Optional KMeans + PCA Clustering"):
    num_clusters = st.slider("Select number of clusters", 2, 5, 3)
    scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled)
    df['Cluster'] = labels
    pca = PCA(2)
    components = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(data=components, columns=["PC1","PC2"])
    pca_df['Cluster'] = labels
    fig = px.scatter(
        pca_df, x="PC1", y="PC2", color=pca_df["Cluster"].astype(str),
        title="KMeans Clustering with PCA",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig, use_container_width=True)

# ===== Sentiment Analysis =====
with st.expander("Sentiment Analysis"):
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





