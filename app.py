
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from textblob import TextBlob
import nltk
nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="AI Mental Health Dashboard", layout="wide")
st.title("AI Mental Health and Lifestyle Dashboard")

# MARK 1 - Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("mental_health_lifestyle.csv")
        return df
    except Exception as e:
        st.error(f"CSV file not found or could not be loaded: {e}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.error("Dataset is empty. Please check your CSV file or path.")
    st.stop()

# MARK 2 - Dataset Overview
st.subheader("Dataset Overview")
st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
st.dataframe(df.head(10))

# MARK 3 - Numeric Columns Distribution
numeric_cols = df.select_dtypes(include=['float64','int64']).columns
for col in numeric_cols:
    fig = px.histogram(df, x=col, title=f"Distribution of {col}")
    st.plotly_chart(fig, use_container_width=True)

# MARK 4 - Correlation Heatmap
numeric_df = df[numeric_cols].dropna()
if numeric_df.shape[1] > 0:
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(7,5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    st.pyplot(fig, use_container_width=True)

# MARK 5 - Label Encoding
df = df.dropna()
label_cols = df.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for col in label_cols:
    df[col] = encoder.fit_transform(df[col])

# MARK 6 - Target & Features
if len(numeric_cols) > 0:
    target_col = numeric_cols[0]
else:
    target_col = df.columns[0]

X = df.drop(columns=[target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write("Target column:", target_col)
st.write("Features shape:", X.shape)
st.write("Target shape:", y.shape)

# MARK 7 - Initialize Models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(random_state=42)
ada = AdaBoostClassifier(random_state=42)
svm = SVC(kernel='rbf', probability=True)
lr = LogisticRegression(max_iter=1000)
knn = KNeighborsClassifier()
nb = GaussianNB()
dt = DecisionTreeClassifier(random_state=42)

models = {
    "Random Forest": rf,
    "Gradient Boosting": gb,
    "AdaBoost": ada,
    "SVM": svm,
    "Logistic Regression": lr,
    "KNN": knn,
    "Naive Bayes": nb,
    "Decision Tree": dt
}

# MARK 8 - Train & Evaluate Models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = accuracy_score(y_test, preds)

result_df = pd.DataFrame(list(results.items()), columns=["Model","Accuracy"]).sort_values(by="Accuracy", ascending=False)
fig = px.bar(result_df, x="Model", y="Accuracy", color="Accuracy", text_auto=".2f", title="Model Accuracy Comparison")
st.plotly_chart(fig, use_container_width=True)

best_model_name = max(results, key=results.get)
st.success(f"Best Model: {best_model_name} | Accuracy: {results[best_model_name]:.2f}")

# MARK 9 - Confusion Matrix
best_model = models[best_model_name]
preds = best_model.predict(X_test)
cm = confusion_matrix(y_test, preds)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# MARK 10 - Neural Network Training
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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

# MARK 11 - PCA + KMeans Clustering
num_clusters = 3
scaled = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(scaled)
df['Cluster'] = labels

pca = PCA(n_components=2)
components = pca.fit_transform(scaled)
pca_df = pd.DataFrame(data=components, columns=["PC1","PC2"])
pca_df['Cluster'] = labels
fig = px.scatter(pca_df, x="PC1", y="PC2", color=pca_df["Cluster"].astype(str), title="KMeans Clustering with PCA")
st.plotly_chart(fig, use_container_width=True)

# MARK 12 - Sentiment Analysis
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








