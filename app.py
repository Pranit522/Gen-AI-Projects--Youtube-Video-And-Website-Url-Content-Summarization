import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.llms import Ollama
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from bs4 import BeautifulSoup
import re

st.set_page_config(page_title="🦙 Gen AI Summarizer")
st.title("📽️ YouTube + 🌐 Website Content Summarizer using LLaMA3")

option = st.radio("Select content type:", ["YouTube Video", "Website Article"])
url_input = st.text_input("Enter YouTube or Website URL:")

if url_input:
    if option == "YouTube Video":
        # 🔧 Fixing the regex for video ID extraction
        video_id_match = re.search(r"(?:v=|be/)([0-9A-Za-z_-]{11})", url_input)
        if video_id_match:
            video_id = video_id_match.group(1)
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                full_text = " ".join([item["text"] for item in transcript])
            except:
                st.error("❌ Failed to fetch transcript. Try another video.")
                full_text = ""
        else:
            st.error("❌ Invalid YouTube URL")
            full_text = ""

    elif option == "Website Article":
        try:
            response = requests.get(url_input)
            soup = BeautifulSoup(response.text, "html.parser")
            full_text = soup.get_text()
        except:
            st.error("❌ Failed to fetch website content.")
            full_text = ""

    # Proceed only if text is available
    if full_text:
        st.subheader("🧾 Extracted Text Preview:")
        st.write(full_text[:1000] + "...")  # Show first 1000 chars

        # Text Splitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = splitter.create_documents([full_text])

        # Load LLama3 via Ollama
        llm = Ollama(model="llama3")

        # Load Refine Chain for better summary
        chain = load_summarize_chain(llm, chain_type="refine", verbose=False)

        # Run the chain
        st.subheader("🦙 Generating Summary...")
        with st.spinner("Summarizing using LLaMA3..."):
            summary = chain.run(docs)
            st.success("✅ Summary Ready")
            st.markdown(f"### 📝 Summary:\n\n{summary}")
