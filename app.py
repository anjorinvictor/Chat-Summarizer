import streamlit as st
import re
import html
import json
from fpdf import FPDF
from collections import Counter
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import os
import os

# ------------------- NLTK FIX -------------------
import nltk
# Download required NLTK tokenizers at runtime
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
# -------------------------------------------------

HISTORY_FILE = "summary_history.json"

# ---------- TEXT RANK SUMMARIZER ----------
def textrank_summary(text, sentences_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join(str(sentence) for sentence in summary)

def chunk_text(text, chunk_size=2500):
    for i in range(0, len(text), chunk_size):
        yield text[i:i+chunk_size]

def summarize_large_text(text, sentences_per_chunk=3, final_sentences=3):
    chunks = list(chunk_text(text))
    partial_summaries = [textrank_summary(chunk, sentences_per_chunk) for chunk in chunks]
    combined_summary = " ".join(partial_summaries)
    return textrank_summary(combined_summary, final_sentences)

# ---------- SIMPLE SENTIMENT ----------
def simple_sentiment(text):
    positive_words = ["good","great","excellent","amazing","love","happy","nice","positive"]
    negative_words = ["bad","terrible","sad","hate","angry","poor","negative"]
    score = 0
    for w in re.findall(r'\w+', text.lower()):
        if w in positive_words: score += 1
        if w in negative_words: score -= 1
    if score > 0: return f"Positive (score {score})"
    elif score < 0: return f"Negative (score {score})"
    else: return "Neutral"

# ---------- TEXT STATISTICS ----------
def text_stats(text):
    words = re.findall(r'\w+', text)
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_words = round(len(words)/len(sentences), 2) if sentences else 0
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_words_per_sentence": avg_words
    }

# ---------- TOP WORDS ----------
def top_words(text, n=10):
    stopwords = set([
        "the","and","to","of","a","in","is","it","you","that","i","for","on","with","was","as",
        "at","be","this","have","or","from","by","an"
    ])
    words = [w.lower() for w in re.findall(r'\w+', text) if w.lower() not in stopwords]
    freq = Counter(words)
    return freq.most_common(n)

# ---------- HELPER ----------
def strip_non_latin1(text):
    return text.encode('latin-1', 'ignore').decode('latin-1')

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="ChatSummarizer", layout="centered")

# Top UI section with instructions
st.markdown("""
    <div style='background-color:#f0f4f8;padding:20px;border-radius:10px'>
        <h2 style='color:#4A90E2'>💬 ChatSummarizer</h2><br>
        <div style='text-align:right; font-weight:bold; color:#333; margin-top:-30px;'>
            Developed by Olamide
        </div>
        <p>Quickly summarize your WhatsApp chats or any long text offline.</p>
        <b>How to use:</b>
        <ol>
            <li>Paste your text in the box below.</li>
            <li>Adjust the number of sentences for the summary.</li>
            <li>Click <b>Summarize</b> to get your summary.</li>
            <li>View, copy, or download your summary and report.</li>
        </ol>
        <p><b>Purpose:</b> Summarizes long text and provides word stats, top words, and sentiment analysis.</p>
    </div>
""", unsafe_allow_html=True)

# Input area
text_input = st.text_area("Paste your long message here", height=300)
if not text_input:
    st.info("Paste text above to summarize.")
    st.stop()

# Show total words before summarization
total_words = len(re.findall(r'\w+', text_input))
st.write(f"**Total words in original text:** {total_words}")

# Number of sentences slider and show original checkbox
col1, col2 = st.columns([1,1])
with col1:
    n_sent = st.slider("Number of sentences in summary", 1, 8, 3)
with col2:
    show_original = st.checkbox("Show original text after summary", value=False)

# ---------- HISTORY ----------
history = load_history()

# ---------- SUMMARIZATION BUTTON ----------
if st.button("Summarize"):
    summary = summarize_large_text(text_input, sentences_per_chunk=n_sent, final_sentences=n_sent)
    
    # Append to history and save
    history.append({"input": text_input, "summary": summary})
    save_history(history)
    
    # ---------- DISPLAY SUMMARY ----------
    st.subheader("✅ Summary")
    st.markdown(f"<div style='background-color:#E6E6E6;padding:10px;border-radius:5px'>{summary}</div>", unsafe_allow_html=True)
    
    # Summary stats
    summary_words = len(re.findall(r'\w+', summary))
    st.write(f"**Summary word count:** {summary_words}")
    
    # Sentiment
    sentiment = simple_sentiment(text_input)
    st.write("**Sentiment:**", sentiment)
    
    # Copy button
    escaped_summary = html.escape(summary).replace("\n", "\\n")
    copy_button = f"""
    <button onclick="navigator.clipboard.writeText('{escaped_summary}')"
    style="background-color:#4CAF50;color:white;padding:8px 12px;border:none;border-radius:5px;cursor:pointer;">
    📋 Copy summary
    </button>
    """
    st.markdown(copy_button, unsafe_allow_html=True)
    
    # Download TXT
    st.download_button("💾 Download summary (txt)", summary, file_name="summary.txt")
    
    # ---------- PDF REPORT (Upgraded & Styled) ----------
    stats = text_stats(text_input)
    top = top_words(text_input)
    clean_summary = strip_non_latin1(summary)

    pdf = FPDF()
    pdf.add_page()

    # ---------- Header ----------
    pdf.set_fill_color(74, 144, 226)  # blue header
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 15, "ChatSummarizer", ln=True, align="C", fill=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, "Developed by Olamide", ln=True, align="C", fill=False)
    pdf.ln(5)

    pdf.set_text_color(0, 0, 0)

    # ---------- Summary Section ----------
    pdf.set_fill_color(240, 240, 240)  # light gray
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Summary:", ln=True, fill=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, clean_summary)
    pdf.ln(3)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    # ---------- Text Statistics ----------
    pdf.set_fill_color(230, 250, 230)  # light green
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Text Statistics:", ln=True, fill=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Total words: {stats['word_count']}", ln=True)
    pdf.cell(0, 8, f"Sentence count: {stats['sentence_count']}", ln=True)
    pdf.cell(0, 8, f"Avg words per sentence: {stats['avg_words_per_sentence']}", ln=True)
    pdf.ln(3)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    # ---------- Top Words ----------
    pdf.set_fill_color(255, 250, 205)  # light yellow
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Top Words:", ln=True, fill=True)
    pdf.set_font("Arial", '', 12)
    for word, count in top:
        pdf.cell(0, 8, f"{word}: {count}", ln=True)
    pdf.ln(3)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    # ---------- Sentiment ----------
    pdf.set_fill_color(250, 230, 230)  # light pink
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Sentiment:", ln=True, fill=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, sentiment, ln=True)
    pdf.ln(10)

    # ---------- Footer ----------
    pdf.set_y(-20)
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Generated by ChatSummarizer", 0, 0, 'C')

    # Export PDF
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    st.download_button("📄 Download upgraded report (PDF)", data=pdf_bytes,
                       file_name="summary_report_upgraded.pdf", mime="application/pdf")

# ---------- DISPLAY HISTORY ----------
st.subheader("🕒 History")
delete_history = st.button("🗑️ Clear History")
if delete_history:
    history = []
    save_history(history)
    st.success("History cleared!")

if history:
    for i, item in enumerate(reversed(history), 1):
        with st.expander(f"Summary #{i}"):
            st.write("**Original Text:**")
            st.write(item["input"])
            st.write("**Summary:**")
            st.markdown(f"<div style='background-color:#E6E6E6;padding:5px;border-radius:5px'>{item['summary']}</div>", unsafe_allow_html=True)

# ---------- SHOW ORIGINAL TEXT ----------
if show_original:
    st.subheader("Original text")
    st.write(text_input)



