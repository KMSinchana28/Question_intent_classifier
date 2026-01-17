import streamlit as st
import joblib

# Load trained ML model
model = joblib.load("intent_model.pkl")

st.set_page_config(page_title="Question Intent Classifier")

st.title("🧠 Question Intent Classification")
st.write("Enter a question and the system will predict its intent.")

question = st.text_input("Enter your question:")

if st.button("Predict Intent"):
    if question.strip() == "":
        st.warning("Please enter a question")
    else:
        q = question.lower().strip()

        # ---------- RULE-BASED INTENT (FIRST PRIORITY) ----------
        if q.startswith((
            "is", "are", "can", "does", "do", "did",
            "was", "were", "will", "should", "could", "would"
        )):
            prediction = "YesNo"

        elif "compare" in q or "difference" in q:
            prediction = "Comparison"

        # ---------- MACHINE LEARNING INTENT ----------
        else:
            prediction = model.predict([question])[0]

        st.success(f"Predicted Intent: **{prediction}**")
