import streamlit as st
from customer_agent.run_support_agent import run_customer_support
from workflow_builder.workflow import app
from typing import Dict, TypedDict




# --- Streamlit UI ---
st.set_page_config(page_title="24/7 Customer Support Bot", page_icon="🤖")

st.title("📞 24/7 AI Customer Support")
st.markdown("---")

st.write(
    "Welcome! I'm your AI customer support assistant. "
    "Ask me anything related to technical issues, billing, or general inquiries."
)

user_query = st.text_area("Enter your query here:", height=100)

if st.button("Get Support"):
    if user_query:
        with st.spinner("Processing your request..."):
            result = run_customer_support(user_query)
        
        st.markdown("---")
        st.subheader("Analysis Results:")
        st.write(f"**Query:** {user_query}")
        st.write(f"**Category:** {result['category']}")
        st.write(f"**Sentiment:** {result['sentiment']}")
        
        st.subheader("AI Response:")
        st.info(result['response']) # Use st.info for a visually distinct output
    else:
        st.warning("Please enter a query to get support.")

st.markdown("---")
st.caption("Powered by LangGraph and Streamlit")