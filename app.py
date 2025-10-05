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

st.set_page_config(page_title="AI Mental Health Dashboard", layout="wide")
st.title("AI Mental Health and Lifestyle Dashboard")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("mental_health_lifestyle.csv")
        return df
    except:
        st.error("CSV file not found. Make sure 'mental_health_lifestyle.csv' is in your project folder.")
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
    fig, ax = plt.subplots(figsize=(7,5))
    sns.set_theme(style="white")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=0.5,
                annot=True, fmt=".2f", cbar_kws={"shrink":0.8, "label":"Correlation"})
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
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
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "SVM": SVC(kernel="rbf", probability=True),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
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
    fig = px.bar(result_df, x="Model", y="Accuracy", color="Accuracy", text_auto=".2f", title="Model Accuracy Comparison")
    st.plotly_chart(fig, use_container_width=True)

    best_model_name = max(results, key=results.get)
    st.success(f"Best Model: {best_model_name} | Accuracy: {results[best_model_name]:.2f}")

if st.checkbox("Show Confusion Matrix for Best Model"):
        best_model = models[best_model_name]
        preds = best_model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

st.subheader("Optional Neural Network Training")
if st.checkbox("Train Neural Network"):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
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
    fig = px.scatter(pca_df, x="PC1", y="PC2", color=pca_df["Cluster"].astype(str),
                     title="KMeans Clustering with PCA")
    st.plotly_chart(fig, use_container_width=True)

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


