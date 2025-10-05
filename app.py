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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from textblob import TextBlob
import nltk
nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="AI Mental Health Dashboard", layout="wide")
st.title("AI Mental Health Dashboard - Compact Version")

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("mental_health_lifestyle.csv")
        if df.shape[0] > 5000:
            df = df.sample(5000, random_state=42).reset_index(drop=True)
        return df
    except:
        st.error("CSV file not found.")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()
st.dataframe(df.head(10))

# Preprocessing
df = df.dropna()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
target_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]

X = df.drop(columns=[target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
st.header("Model Training & Evaluation")
models = {
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "SVM": SVC(kernel="rbf", probability=True),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = accuracy_score(y_test, preds)

result_df = pd.DataFrame(list(results.items()), columns=["Model","Accuracy"]).sort_values(by="Accuracy", ascending=False)
fig = px.bar(result_df, x="Model", y="Accuracy", color="Accuracy", text_auto=".2f")
st.plotly_chart(fig, use_container_width=True)

best_model_name = max(results, key=results.get)
st.success(f"Best Model: {best_model_name} | Accuracy: {results[best_model_name]:.2f}")

cm = confusion_matrix(y_test, models[best_model_name].predict(X_test))
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Neural Network
st.header("Neural Network")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = nn.fit(X_train_scaled, y_train, epochs=3, batch_size=32, validation_split=0.2, verbose=0)
test_loss, test_acc = nn.evaluate(X_test_scaled, y_test, verbose=0)
st.success(f"NN Test Accuracy: {test_acc:.2f}")

fig, ax = plt.subplots(1,2, figsize=(10,4))
ax[0].plot(history.history['accuracy'], label='Train Accuracy')
ax[0].plot(history.history['val_accuracy'], label='Val Accuracy')
ax[0].legend(); ax[0].set_title("Accuracy")
ax[1].plot(history.history['loss'], label='Train Loss')
ax[1].plot(history.history['val_loss'], label='Val Loss')
ax[1].legend(); ax[1].set_title("Loss")
st.pyplot(fig)

# KMeans + PCA
st.header("KMeans + PCA")
scaled = StandardScaler().fit_transform(X)
labels = KMeans(n_clusters=3, random_state=42).fit_predict(scaled)
pca_df = pd.DataFrame(PCA(2).fit_transform(scaled), columns=["PC1","PC2"])
pca_df['Cluster'] = labels
fig = px.scatter(pca_df, x="PC1", y="PC2", color=pca_df["Cluster"].astype(str))
st.plotly_chart(fig, use_container_width=True)

# Sentiment Analysis
st.header("Sentiment Analysis")
text_input = st.text_area("Enter text for sentiment:")
sentiment = TextBlob(text_input).sentiment.polarity if text_input.strip() else 0
if sentiment > 0: st.success(f"Positive ({sentiment:.2f})")
elif sentiment < 0: st.error(f"Negative ({sentiment:.2f})")
else: st.warning("Neutral (0.00)")

st.markdown("---")
st.markdown("Developed for AI Mental Health Research Dashboard")

