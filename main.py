from flask import Flask, render_template, request
from transformers import pipeline
import torch

app = Flask(__name__)

device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

def chunk_text(text, max_length=1000):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    text = request.form["text"]
    chunks = chunk_text(text)
    partial_summaries = [summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]["summary_text"] for chunk in chunks]
    combined = " ".join(partial_summaries)

    if len(chunks) > 1:
        final = summarizer(combined, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
    else:
        final = combined

    return render_template("index.html", summary=final)

if __name__ == "__main__":
    app.run(debug=True)
