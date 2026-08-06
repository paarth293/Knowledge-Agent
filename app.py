"""
Streamlit Frontend — Personal Knowledge Agent
===============================================
Run with: streamlit run app.py
"""

import streamlit as st
import requests

# ─── Page Config ───
st.set_page_config(
    page_title="Personal Knowledge Agent",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Personal Knowledge Agent")
st.caption("Upload your documents and have a conversation with them.")

API_BASE = "http://localhost:8000/api"

# ─── Sidebar: Document Upload ───
with st.sidebar:
    st.header("📁 Your Documents")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file and st.button("Index Document"):
        with st.spinner("Uploading and indexing..."):
            response = requests.post(
                f"{API_BASE}/documents",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            )
            if response.status_code == 200:
                data = response.json()
                st.success(f"Document uploaded! ID: {data['document_id'][:8]}...")
                st.info("Processing in background. Refresh the list in a moment.")
            else:
                st.error("Upload failed. Is the API running?")

    if st.button("Refresh Document List"):
        response = requests.get(f"{API_BASE}/documents")
        if response.status_code == 200:
            docs = response.json()
            if docs:
                for doc in docs:
                    status_icon = "✅" if doc["status"] == "ready" else "⏳"
                    st.write(f"{status_icon} {doc['filename']} ({doc['chunk_count']} chunks)")
            else:
                st.write("No documents uploaded yet.")

# ─── Main: Chat Interface ───
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    import uuid
    st.session_state.conversation_id = str(uuid.uuid4())

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for src in msg["sources"]:
                    st.write(f"• {src['filename']} — Page {src['page_number']}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_BASE}/chat",
                json={"message": prompt, "conversation_id": st.session_state.conversation_id},
                stream=True
            )

            full_answer = ""
            sources = []
            placeholder = st.empty()

            if response.status_code == 200:
                import json
                for line in response.iter_lines():
                    if line and line.startswith(b"data: "):
                        data = json.loads(line[6:])
                        if data["type"] == "token":
                            full_answer += data["content"]
                            placeholder.write(full_answer)
                        elif data["type"] == "done":
                            sources = data.get("sources", [])
                            break

                if sources:
                    with st.expander("📎 Sources"):
                        for src in sources:
                            st.write(f"• {src['filename']} — Page {src['page_number']}")
            else:
                full_answer = "Error: Could not reach the API. Is it running?"
                placeholder.error(full_answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": sources
    })