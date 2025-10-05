import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="🧠 Mental Health & Lifestyle Dashboard", layout="wide")

st.title("🧠 Mental Health & Lifestyle Data Analysis Dashboard")
st.markdown("تحليل العلاقة بين نمط الحياة والحالة النفسية باستخدام بيانات حقيقية (50K row)")

# ====== تحميل البيانات ======
@st.cache_data
def load_data():
    df = pd.read_csv("mental_health_lifestyle.csv")
    return df

df = load_data()

# ====== عرض أولي ======
st.subheader("📊 عرض أول البيانات")
st.dataframe(df.head(10))

col1, col2 = st.columns(2)
col1.metric("📈 عدد الصفوف", df.shape[0])
col2.metric("📊 عدد الأعمدة", df.shape[1])

# ====== نظرة عامة ======
if st.checkbox("🔍 عرض الإحصائيات الوصفية"):
    st.write(df.describe(include='all'))

# ====== عرض توزيع الأعمدة ======
col = st.selectbox("اختار عمود لعرض توزيعه:", df.columns)
fig = px.histogram(df, x=col, color_discrete_sequence=["#5A9"])
st.plotly_chart(fig, use_container_width=True)

# ====== تنظيف ومعالجة البيانات ======
st.subheader("⚙️ تجهيز البيانات للنمذجة")
df = df.dropna()
label_cols = df.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for c in label_cols:
    df[c] = encoder.fit_transform(df[c])

# ====== اختيار الهدف ======
target_col = st.selectbox("🎯 اختار العمود المستهدف (Target):", df.columns)
X = df.drop(columns=[target_col])
y = df[target_col]

# ====== تحجيم البيانات ======
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ====== تقسيم البيانات ======
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ====== اختيار الخوارزمية ======
st.subheader("🤖 اختيار الخوارزمية")
model_name = st.selectbox(
    "اختار النموذج الذي ترغب في تجربته:",
    ("Random Forest", "Gradient Boosting", "Logistic Regression", "Support Vector Machine", "K-Nearest Neighbors")
)

if st.button("🚀 تدريب النموذج وتشغيله"):
    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=250, learning_rate=0.1, random_state=42)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=500)
    elif model_name == "Support Vector Machine":
        model = SVC(kernel='rbf', C=2)
    else:
        model = KNeighborsClassifier(n_neighbors=7)
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.success(f"✅ دقة النموذج ({model_name}): {acc*100:.2f}%")

    st.write("📋 تقرير التصنيف:")
    st.text(classification_report(y_test, preds))

    # ====== مصفوفة الالتباس ======
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

# ====== تحليل الارتباط ======
if st.checkbox("📈 عرض مصفوفة الارتباط"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ====== تحليل إضافي ======
if st.checkbox("📊 تحليل العلاقة بين عمودين"):
    c1 = st.selectbox("المتغير الأول:", df.columns)
    c2 = st.selectbox("المتغير الثاني:", df.columns)
    fig = px.scatter(df, x=c1, y=c2, color=target_col, trendline="ols", title=f"العلاقة بين {c1} و {c2}")
    st.plotly_chart(fig, use_container_width=True)


