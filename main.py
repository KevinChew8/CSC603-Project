from flask import Flask, render_template, request, jsonify
import requests
import re
from pathlib import Path

# -----------------------------
# MODEL SETUP (AUTO DOWNLOAD)
# -----------------------------

MODEL_PATH = Path("./Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")

MODEL_URL = "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"

def download_model():
    print("Model not found. Downloading...")

    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print("Download complete!")

if not MODEL_PATH.exists():
    download_model()

# -----------------------------
# LOAD LLM
# -----------------------------

from llama_cpp import Llama

llama3 = Llama(
    model_path=str(MODEL_PATH),
    verbose=False,
    n_gpu_layers=-1,
    n_ctx=8192,
)

def generate_response(_model, _messages):
    return _model.create_chat_completion(
        _messages,
        max_tokens=200,
        temperature=0.5,
    )["choices"][0]["message"]["content"]

# -----------------------------
# RECOMMENDATION LOGIC
# -----------------------------

def extract_items(text):
    lines = text.split("\n")
    items = []

    for line in lines:
        cleaned = re.sub(r"^\s*[\d\-\)\.]+\s*", "", line).strip()
        if cleaned:
            items.append(cleaned)

    return items


def parse_item(item):
    """
    Extract media type and title
    Example: [Movie] Get Out
    """
    match = re.match(r"\[(.*?)\]\s*(.*)", item)

    if match:
        media_type = match.group(1).lower()
        title = match.group(2)
    else:
        media_type = "unknown"
        title = item

    return media_type, title


def get_recommendations(model, user_input):
    system_prompt = (
        "You are a multimedia recommendation engine.\n\n"
        "You recommend movies, music, books, and video games.\n\n"
        "STRICT RULES:\n"
        "- Output EXACTLY 5 items\n"
        "- Each item must include a media tag:\n"
        "  [Movie], [Music], [Book], or [Game]\n"
        "- NO explanations\n"
        "- NO extra text\n"
        "- ONLY numbered list\n"
        "- Format exactly like:\n"
        "  1. [Movie] Get Out\n"
        "  2. [Music] Thriller - Michael Jackson\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    output = generate_response(model, messages)
    items = extract_items(output)

    parsed = []
    for item in items[:5]:
        media_type, title = parse_item(item)
        parsed.append({
            "type": media_type,
            "title": title
        })

    return parsed

# -----------------------------
# TMDb POSTER FETCH (MOVIES ONLY)
# -----------------------------

API_KEY = ""

def get_movie_poster(title):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": API_KEY,
        "query": title,
    }

    response = requests.get(url, params=params).json()

    if response.get("results"):
        for r in response["results"]:
            if r.get("poster_path"):
                return f"https://image.tmdb.org/t/p/w500{r['poster_path']}"

    return "https://via.placeholder.com/300x450?text=No+Image"

# -----------------------------
# FLASK APP
# -----------------------------

app = Flask(__name__)

@app.route('/')
def main_page():
    return render_template('main_page.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_input = data.get("query")

    items = get_recommendations(llama3, user_input)

    results = []
    for item in items:
        if item["type"] == "movie":
            poster = get_movie_poster(item["title"])
        else:
            poster = None

        results.append({
            "title": item["title"],
            "type": item["type"],
            "poster": poster
        })

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)