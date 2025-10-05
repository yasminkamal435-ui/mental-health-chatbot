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
from textblob import TextBlob
import nltk
import random

nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="Mental Health & Lifestyle Lite", layout="wide")
st.title("Mental Health & Lifestyle Lite")

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

if "points" not in st.session_state:
    st.session_state.points = 0
if "streak" not in st.session_state:
    st.session_state.streak = 0

st.sidebar.title("Dashboard Control")
if st.sidebar.checkbox("Show first 5 rows"):
    st.dataframe(df.head(5))
st.sidebar.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

col1, col2 = st.columns(2)

color_palettes = {
    "Social_Media_Usage": ["#a8dadc","#f1faee","#d3d3d3"],
    "Diet_Quality": ["#457b9d","#adb5bd","#6c757d"],
    "Smoking_Habit": ["#1d3557","#a8dadc","#ced4da"],
    "Alcohol_Consumption": ["#6c757d","#f1faee","#495057"],
    "Medication_Usage": ["#a8dadc","#457b9d","#adb5bd"]
}

with col1:
    st.subheader("Target Column Distribution")
    target_col = st.selectbox("Select Target Column", df.columns)
    colors = color_palettes.get(target_col, px.colors.qualitative.Plotly)
    fig = px.histogram(df, x=target_col, color=target_col, color_discrete_sequence=colors, title=f"Distribution of {target_col}")
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

st.header("Column Insights (Numeric Data)")
numeric_cols = df.select_dtypes(include=['float64','int64']).columns
for col in numeric_cols:
    mean_val = df[col].mean()
    min_val = df[col].min()
    max_val = df[col].max()
    st.write(f"**{col}**: Mean = {mean_val:.2f}, Min = {min_val:.2f}, Max = {max_val:.2f}")

df = df.dropna()
label_cols = df.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for col in label_cols:
    df[col] = encoder.fit_transform(df[col])

target = target_col
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

st.header("Model Training and Evaluation")
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM": SVC(kernel="rbf", probability=True),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

selected_models = st.multiselect("Select models to train", list(models.keys()), default=["Random Forest", "Logistic Regression"])
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
    st.session_state.points += 5
    st.session_state.streak += 1

if st.checkbox("Show Confusion Matrix for Best Model"):
    best_model = models[best_model_name]
    preds = best_model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax)
    st.pyplot(fig, use_container_width=True)

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
        st.session_state.points += 1
    else:
        st.warning("Please enter valid text.")

st.header("Daily Mini Quiz")
quiz_score = 0
questions = {
    "How often do you exercise per week?": ["0 times","1-2 times","3-5 times","Everyday"],
    "How many hours of sleep do you get per night?": ["<5","5-6","6-8","8+"],
    "How many servings of fruits/vegetables do you eat daily?": ["0-1","2-3","4+"]
}
answers = {}
for q, opts in questions.items():
    answers[q] = st.radio(q, opts)
if st.button("Submit Quiz"):
    if answers["How often do you exercise per week?"] in ["3-5 times","Everyday"]:
        quiz_score += 1
    if answers["How many hours of sleep do you get per night?"] in ["6-8","8+"]:
        quiz_score += 1
    if answers["How many servings of fruits/vegetables do you eat daily?"] == "4+":
        quiz_score += 1
    st.info(f"Your Mini Wellness Score: {quiz_score}/3")
    if quiz_score == 3:
        st.success("Excellent! Keep up the healthy habits.")
    elif quiz_score == 2:
        st.warning("Good! Try to improve a bit more.")
    else:
        st.error("Consider improving your lifestyle habits.")
    st.session_state.points += quiz_score
    st.session_state.streak += 1

st.header("Random Wellness Tip")
tips = [
    "Drink at least 8 glasses of water today.",
    "Take a 10-minute walk to refresh your mind.",
    "Practice deep breathing for 5 minutes.",
    "Include more fruits and vegetables in your meals.",
    "Limit social media usage to improve focus.",
    "Take short breaks every hour if working/studying.",
    "Try to get at least 7 hours of sleep tonight.",
    "Write down 3 things you are grateful for today."
]
st.info(random.choice(tips))

st.header("Mini Lifestyle Tracker")
exercise = st.checkbox("Did you exercise today?")
sleep = st.checkbox("Did you sleep at least 7 hours?")
water = st.checkbox("Did you drink 8 glasses of water?")
fruits = st.checkbox("Did you eat enough fruits/vegetables?")
habits_score = sum([exercise,sleep,water,fruits])
if st.button("Submit Daily Habits"):
    st.info(f"Your Daily Habits Score: {habits_score}/4")
    if habits_score == 4:
        st.success("Excellent! Full healthy habits achieved today.")
    elif habits_score >= 2:
        st.warning("Good! Try to improve the remaining habits.")
    else:
        st.error("Consider improving your daily lifestyle habits.")
    st.session_state.points += habits_score
    st.session_state.streak += 1

st.header("Daily Challenge / Reminder")
daily_challenges = [
    "Drink 8 glasses of water today.",
    "Walk 10 minutes outside.",
    "Do 5 minutes of stretching.",
    "Take a 5-minute meditation break."
]
challenge = random.choice(daily_challenges)
if st.button("Mark Challenge Done"):
    st.success(f"You completed: {challenge}")
    st.session_state.points += 2
    st.session_state.streak += 1
else:
    st.info(f"Today's Challenge: {challenge}")

st.header("Gamification Summary")
st.write(f"Total Points: {st.session_state.points}")
st.write(f"Current Streak: {st.session_state.streak}")
if st.session_state.points >= 20:
    st.success("Achievement Unlocked: Wellness Champion!")
elif st.session_state.points >= 10:
    st.info("Achievement Unlocked: Wellness Enthusiast!")

st.markdown("---")
st.markdown("Lite Version for Free Users")







