# ====================== استيراد المكتبات ======================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from textblob import TextBlob
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ====================== إعداد الصفحة ======================
st.set_page_config(page_title="🧠 AI Mental Health System", layout="wide")
st.title("🧠 AI-Powered Mental Health & Lifestyle Intelligence System")
st.markdown("""
هذا النظام يستخدم الذكاء الاصطناعي لتحليل أنماط الحياة والحالة النفسية للأفراد.  
يمكنك تحليل البيانات، مقارنة النماذج، التنبؤ، وحتى التفاعل مع مساعد ذكي.
""")

# ====================== تحميل البيانات ======================
@st.cache_data
def load_data():
    df = pd.read_csv("mental_health_lifestyle.csv")
    return df

df = load_data()

st.sidebar.header("🔧 إعدادات العرض")
st.sidebar.info("تحكم في واجهة التحليل واختيارات النماذج من هنا.")

# ====================== عرض البيانات ======================
st.subheader("📊 عرض البيانات الأساسية")
rows = st.slider("عدد الصفوف للعرض:", 5, 100, 10)
st.dataframe(df.head(rows))

st.metric("📈 عدد الصفوف", df.shape[0])
st.metric("📊 عدد الأعمدة", df.shape[1])

# ====================== التحليل الوصفي ======================
if st.checkbox("🔍 عرض الإحصائيات الوصفية"):
    st.write(df.describe(include='all'))

if st.checkbox("📈 عرض القيم المفقودة"):
    st.write(df.isnull().sum())

# ====================== تحليل التوزيعات ======================
st.subheader("📉 تحليل التوزيعات")
col = st.selectbox("اختار عمود لعرض توزيعه:", df.columns)
fig = px.histogram(df, x=col, color_discrete_sequence=["#0083B8"], title=f"توزيع العمود: {col}")
st.plotly_chart(fig, use_container_width=True)

# ====================== تنظيف البيانات ======================
st.subheader("⚙️ تجهيز البيانات للنمذجة")
df = df.dropna()
label_cols = df.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for c in label_cols:
    df[c] = encoder.fit_transform(df[c])

target_col = st.selectbox("🎯 اختار العمود المستهدف (Target):", df.columns)
X = df.drop(columns=[target_col])
y = df[target_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ====================== النماذج ======================
st.subheader("🤖 مقارنة خوارزميات الذكاء الاصطناعي")

models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1),
    "AdaBoost": AdaBoostClassifier(n_estimators=200),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM": SVC(kernel='rbf', C=2),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB()
}

selected_models = st.multiselect("اختر النماذج التي ترغب بتجربتها:", list(models.keys()), default=["Random Forest", "Gradient Boosting", "AdaBoost"])
results = {}

if st.button("🚀 تدريب وتشغيل النماذج"):
    progress = st.progress(0)
    for i, name in enumerate(selected_models):
        model = models[name]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cv = cross_val_score(model, X_scaled, y, cv=5).mean()
        results[name] = {"Accuracy": acc, "CV": cv}
        progress.progress((i+1)/len(selected_models))
        st.success(f"{name}: دقة = {acc*100:.2f}% | CrossVal = {cv*100:.2f}%")

    results_df = pd.DataFrame(results).T
    st.write(results_df)
    fig = px.bar(results_df, x=results_df.index, y="Accuracy", title="📊 مقارنة دقة النماذج", color="Accuracy")
    st.plotly_chart(fig, use_container_width=True)

# ====================== مصفوفة الارتباط ======================
if st.checkbox("🧩 عرض مصفوفة الارتباط"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax, annot=True)
    st.pyplot(fig)

# ====================== تحليل العلاقة ======================
if st.checkbox("🔗 تحليل العلاقة بين متغيرين"):
    c1 = st.selectbox("المتغير الأول:", df.columns)
    c2 = st.selectbox("المتغير الثاني:", df.columns)
    fig = px.scatter(df, x=c1, y=c2, color=target_col, trendline="ols", title=f"العلاقة بين {c1} و {c2}")
    st.plotly_chart(fig, use_container_width=True)

# ====================== تنبؤ تفاعلي ======================
st.subheader("🧮 تنبؤ جديد بالحالة النفسية")
user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(f"{col}:", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

if st.button("🔮 تنفيذ التنبؤ"):
    model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict([list(user_input.values())])[0]
    st.success(f"✅ الحالة المتوقعة: {target_col} = {pred}")

# ====================== تحليل النصوص النفسية ======================
st.subheader("🧠 تحليل المشاعر النصية (Sentiment & Emotion)")
text_input = st.text_area("اكتب جملة تصف حالتك النفسية الآن:")
if st.button("🩺 تحليل النص"):
    if text_input:
        blob = TextBlob(text_input)
        polarity = blob.sentiment.polarity
        sentiment = "إيجابي 😊" if polarity > 0 else "سلبي 😞" if polarity < 0 else "محايد 😐"
        st.info(f"تحليل المشاعر: {sentiment} | درجة الإيجابية: {polarity:.2f}")
    else:
        st.warning("رجاءً اكتب نصًا للتحليل.")

# ====================== توليد تقرير PDF ======================
st.subheader("📑 توليد تقرير PDF بالنتائج")
def generate_pdf():
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, "Mental Health & Lifestyle AI Report")
    c.drawString(100, 730, f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    c.drawString(100, 710, f"Target Variable: {target_col}")
    for i, (name, vals) in enumerate(results.items()):
        c.drawString(100, 690 - i*20, f"{name}: Accuracy={vals['Accuracy']:.2f}, CV={vals['CV']:.2f}")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

if st.button("📥 تحميل التقرير PDF"):
    pdf = generate_pdf()
    st.download_button("تحميل التقرير", data=pdf, file_name="AI_Mental_Health_Report.pdf", mime="application/pdf")

# ====================== توصيات ذكية ======================
st.subheader("💡 توصيات الذكاء الاصطناعي لتحسين صحتك النفسية")
tips = [
    "🧘 مارس التأمل والتنفس العميق 10 دقائق يوميًا.",
    "🚶‍♀️ تحرك كل ساعة لتجنب الخمول الذهني.",
    "📵 قلل من وقت الشاشات والنوم متأخرًا.",
    "🧑‍🤝‍🧑 تواصل مع أصدقاء أو أقارب تثق بهم.",
    "🍎 تناول أطعمة غنية بالأوميغا 3 والمغنيسيوم.",
]
st.write("بناءً على حالتك، إليك بعض التوصيات:")
st.markdown("\n".join(f"- {tip}" for tip in tips))

# ====================== مساعد ذكي نفسي ======================
st.subheader("💬 المساعد النفسي الذكي")
user_feeling = st.text_input("كيف تشعر اليوم؟ (اكتب جملة مثل: أشعر بالتوتر أو أنا سعيد)")
if st.button("🎧 تحدث مع المساعد"):
    if not user_feeling:
        st.warning("رجاءً اكتب شعورك.")
    else:
        blob = TextBlob(user_feeling)
        p = blob.sentiment.polarity
        if p > 0.3:
            st.write("🌞 يبدو أنك في مزاج جيد! حافظ على هذا الروتين واستمر في النشاطات التي تحبها ❤️")
        elif p < -0.3:
            st.write("😔 لاحظت أنك تشعر بالضيق. حاول التحدث مع صديق أو ممارسة التأمل اليوم.")
        else:
            st.write("😐 تبدو في حالة متوسطة. خذ قسط راحة واشرب ماء بانتظام.")


