from flask import Flask, render_template, request, jsonify
import requests
import re
from pathlib import Path
import os
from dotenv import load_dotenv

# -----------------------------
# LOAD ENV
# -----------------------------

load_dotenv()

TMDB_API_KEY = os.getenv("API_KEY")      
RAWG_API_KEY = os.getenv("RAWG_API_KEY") 
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

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
        temperature=0.2,
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
        "Recommend movies, books, and video games.\n\n"
        "STRICT RULES:\n"
        "- Output EXACTLY 5 items\n"
        "- Use tags: [Movie], [Book], [Game]\n"
        "- NO explanations\n"
        "- ONLY numbered list\n"
        "- Example:\n"
        "  1. [Movie] Get Out\n"
        "  2. [Game] The Last of Us\n"
        "  3. [Book] It - Stephen King\n"
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
# 🎬 MOVIES (TMDb)
# -----------------------------

def get_movie_data(title):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
    }

    data = requests.get(url, params=params).json()

    if data.get("results"):
        for r in data["results"]:
            if r.get("poster_path"):
                return {
                    "poster": f"https://image.tmdb.org/t/p/w500{r['poster_path']}",
                    "rating": r.get("vote_average")
                }

    return {"poster": None, "rating": None}

# -----------------------------
# 🎮 GAMES (RAWG)
# -----------------------------

game_cache = {}

def get_game_data(title):
    url = "https://api.rawg.io/api/games"
    params = {
        "key": RAWG_API_KEY,
        "search": title
    }

    data = requests.get(url, params=params).json()

    if data.get("results"):
        game = data["results"][0]

        return {
            "poster": game.get("background_image"),
            "rating": game.get("rating")
        }

    return {"poster": None, "rating": None}
# -----------------------------
# 📚 BOOKS (Google Books)
# -----------------------------

book_cache = {}

def get_book_data(title):
    clean_title = re.sub(r"[^\w\s]", "", title)

    url = "https://www.googleapis.com/books/v1/volumes"
    params = {"q": clean_title, "maxResults": 1}

    data = requests.get(url, params=params).json()

    if "items" in data:
        volume = data["items"][0]["volumeInfo"]

        return {
            "poster": volume.get("imageLinks", {}).get("thumbnail"),
            "rating": volume.get("averageRating")
        }

    return {"poster": None, "rating": None}

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
        title = item["title"]
        media_type = item["type"]

        if media_type == "movie":
            data = get_movie_data(title)

        elif media_type == "game":
            data = get_game_data(title)

        elif media_type == "book":
            data = get_book_data(title)

        else:
            data = {"poster": None, "rating": None}

        results.append({
            "title": title,
            "type": media_type,
            "poster": data["poster"],
            "rating": data["rating"]
        })
        
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)