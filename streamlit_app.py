import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Load dataset
df = pd.read_csv("train.csv")
df["Context"] = df["Context"].fillna("")
df["Response"] = df["Response"].fillna("Sorry, I don’t have an answer for that.")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")
context_embeddings = model.encode(df["Context"].tolist(), convert_to_tensor=True)

SIMILARITY_THRESHOLD = 0.55
TOP_K = 3

def chatbot(user_input):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    cos_scores = util.cos_sim(user_embedding, context_embeddings)[0]
    top_k_scores, top_k_indices = torch.topk(cos_scores, k=TOP_K)

    responses_to_return = []
    for score, idx in zip(top_k_scores, top_k_indices):
        if score.item() >= SIMILARITY_THRESHOLD:
            responses_to_return.append(df["Response"].iloc[idx.item()])

    if not responses_to_return:
        return "Sorry, I didn’t understand that. Could you rephrase?"

    return "\n\n".join(responses_to_return)

# Streamlit UI
st.set_page_config(page_title="Mental Health Chatbot", layout="wide")
st.title("🧠 Mental Health Chatbot")
st.write("Type your message below:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:")

if st.button("Send") and user_input:
    reply = chatbot(user_input)
    st.session_state.chat_history.append((user_input, reply))

for user_msg, bot_msg in st.session_state.chat_history:
    st.markdown(f"**You:** {user_msg}")
    st.markdown(f"**Bot:** {bot_msg}")
