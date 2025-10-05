import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Mental Health & Lifestyle Analysis", layout="wide")

st.title("🧠 Mental Health & Lifestyle Data Analysis Dashboard")
st.markdown("تحليل العلاقة بين نمط الحياة والحالة النفسية باستخدام بيانات حقيقية (50K row)")

# ====== تحميل البيانات ======
@st.cache_data
def load_data():
    df = pd.read_csv("mental_health_lifestyle.csv")
    return df

df = load_data()

# ====== عرض أولي ======
st.subheader("📊 عرض أول 10 صفوف من البيانات")
st.dataframe(df.head(10))

st.write("**حجم البيانات:**", df.shape)
st.write("**الأعمدة:**", list(df.columns))

# ====== نظرة عامة ======
if st.checkbox("عرض إحصائيات وصفية"):
    st.write(df.describe(include='all'))

# ====== عرض توزيع الأعمدة ======
col = st.selectbox("اختار عمود لعرض توزيعه", df.columns)
fig, ax = plt.subplots()
sns.histplot(df[col], kde=True, ax=ax)
st.pyplot(fig)

# ====== معالجة أولية ======
st.subheader("⚙️ تجهيز البيانات للنموذج")

df = df.dropna()
label_cols = df.select_dtypes(include=['object']).columns

encoder = LabelEncoder()
for col in label_cols:
    df[col] = encoder.fit_transform(df[col])

# ====== اختيار الهدف ======
target_col = st.selectbox("اختار العمود المستهدف (target)", df.columns)
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====== تدريب نموذج بسيط ======
if st.button(" تشغيل النموذج"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    st.success(f" دقة النموذج: {acc:.2f}")
    st.text("تقرير التصنيف:")
    st.text(classification_report(y_test, preds))

# ====== رسم ارتباطات ======
if st.checkbox("عرض مصفوفة الارتباط"):
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

