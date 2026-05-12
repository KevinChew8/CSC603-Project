from flask import Flask, render_template, request, jsonify
import requests
import re
from pathlib import Path
import os
from dotenv import load_dotenv
from llama_cpp import Llama

# -----------------------------
# LOAD ENV
# -----------------------------

load_dotenv()

TMDB_API_KEY = os.getenv("API_KEY")
RAWG_API_KEY = os.getenv("RAWG_API_KEY")

if not TMDB_API_KEY:
    raise ValueError("TMDB API key missing!")

if not RAWG_API_KEY:
    raise ValueError("RAWG API key missing!")

# -----------------------------
# MODEL SETUP
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

# Uncomment if needed
if not MODEL_PATH.exists():
    download_model()

# -----------------------------
# LOAD LLM
# -----------------------------

print("Loading model...")

llama3 = Llama(
    model_path=str(MODEL_PATH),
    verbose=False,
    n_gpu_layers=-1,
    n_ctx=8192,
)

print("Model loaded!")

# -----------------------------
# REAL GENERATION
# -----------------------------

def generate_response(_model, _messages):

    response = _model.create_chat_completion(
        messages=_messages,
        max_tokens=250,
        temperature=0.7,
        top_p=0.9,
    )

    return response["choices"][0]["message"]["content"]

# -----------------------------
# PARSING
# -----------------------------

def extract_items(text):

    lines = text.split("\n")
    items = []

    for line in lines:

        cleaned = re.sub(
            r"^\s*[\d\-\)\.]+\s*",
            "",
            line
        ).strip()

        if cleaned:
            items.append(cleaned)

    return items

def parse_item(item):

    match = re.match(r"\[(.*?)\]\s*(.*)", item)

    if match:
        media_type = match.group(1).lower().strip()
        title = match.group(2).strip()
    else:
        media_type = "unknown"
        title = item.strip()

    return media_type, title

# -----------------------------
# RECOMMENDATION ENGINE
# -----------------------------

def detect_media_type(user_input):

    text = user_input.lower()

    game_words = [
        "game", "games", "gaming"
    ]

    movie_words = [
        "movie", "movies", "film", "films",
        "cinema"
    ]

    book_words = [
        "book", "books", "novel", "novels",
        "manga", "reading"
    ]

    music_words = [
        "music", "song", "songs",
        "album", "albums", "artist"
    ]

    if any(word in text for word in game_words):
        return "game"

    if any(word in text for word in movie_words):
        return "movie"

    if any(word in text for word in book_words):
        return "book"

    if any(word in text for word in music_words):
        return "music"

    return None

def get_recommendations(model, user_input, favs, dis):

    requested_type = detect_media_type(user_input)

    system_prompt = (
        "You are an advanced multimedia recommendation engine.\n\n"

        "IMPORTANT RULES:\n"
        "- Output EXACTLY 5 items\n"
        "- NO explanations\n"
        "- NO markdown\n"
        "- ONLY numbered list\n"
        "- Use ONLY these tags:\n"
        "  [Movie]\n"
        "  [Game]\n"
        "  [Book]\n"
        "  [Music]\n\n"
    )

    # STRICT SINGLE-TYPE MODE
    if requested_type == "game":

        system_prompt += (
            "The user ONLY wants VIDEO GAMES.\n"
            "ALL 5 recommendations MUST be tagged [Game].\n"
            "DO NOT recommend movies, books, or music.\n\n"
        )

    elif requested_type == "movie":

        system_prompt += (
            "The user ONLY wants MOVIES.\n"
            "ALL 5 recommendations MUST be tagged [Movie].\n"
            "DO NOT recommend games, books, or music.\n\n"
        )

    elif requested_type == "book":

        system_prompt += (
            "The user ONLY wants BOOKS.\n"
            "ALL 5 recommendations MUST be tagged [Book].\n"
            "DO NOT recommend movies, games, or music.\n\n"
        )

    elif requested_type == "music":

        system_prompt += (
            "The user ONLY wants MUSIC.\n"
            "ALL 5 recommendations MUST be tagged [Music].\n"
            "DO NOT recommend movies, games, or books.\n\n"
        )

    # MIXED MODE
    else:

        system_prompt += (
            "Mix media types when appropriate.\n"
            "You may recommend movies, games, books, and music.\n\n"
        )

    system_prompt += (
        "Example:\n"
        "1. [Movie] Blade Runner 2049\n"
        "2. [Game] Control\n"
        "3. [Book] Dune\n"
        "4. [Music] Kid A - Radiohead\n"
        "5. [Movie] Arrival"
    )

    if favs:
        system_prompt += (
            "\n\nUSER FAVORITES:\n" +
            "\n".join(favs) +
            "\n\nRecommend things similar to these favorites but not identical."
        )

    if dis:
        system_prompt += (
            "\n\nUSER DISLIKES:\n" +
            "\n".join(dis) +
            "\n\nAvoid recommending anything similar to these dislikes."
        )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    output = generate_response(model, messages)

    print("\n========== RAW MODEL OUTPUT ==========")
    print(output)
    print("======================================\n")

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
# CACHES
# -----------------------------

movie_cache = {}
game_cache = {}
book_cache = {}
music_cache = {}

# -----------------------------
# MOVIES
# -----------------------------

def get_movie_data(title):

    if title in movie_cache:
        return movie_cache[title]

    url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
    }

    try:

        print(f"[TMDB] Searching for: {title}")

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        print("[TMDB] STATUS:", response.status_code)

        if response.status_code != 200:

            print("[TMDB] ERROR RESPONSE:")
            print(response.text)

            return {
                "poster": None,
                "rating": None
            }

        data = response.json()

        if data.get("results"):

            for r in data["results"]:

                if r.get("poster_path"):

                    result = {
                        "poster": f"https://image.tmdb.org/t/p/w500{r['poster_path']}",
                        "rating": r.get("vote_average")
                    }


                    movie_cache[title] = result
                    return result

    except Exception as e:
        print("[TMDB] EXCEPTION:", e)

    return {
        "poster": None,
        "rating": None
    }

# -----------------------------
# GAMES
# -----------------------------

def get_game_data(title):

    if title in game_cache:
        return game_cache[title]

    url = "https://api.rawg.io/api/games"

    params = {
        "key": RAWG_API_KEY,
        "search": title
    }

    try:

        print(f"[RAWG] Searching for: {title}")

        response = requests.get(
            url,
            params=params,
            timeout=10
        )


        if response.status_code != 200:

            print("[RAWG] ERROR RESPONSE:")
            print(response.text)

            return {
                "poster": None,
                "rating": None
            }

        data = response.json()

        if data.get("results"):

            game = data["results"][0]

            result = {
                "poster": game.get("background_image"),
                "rating": game.get("rating")
            }


            game_cache[title] = result
            return result

    except Exception as e:
        print("[RAWG] EXCEPTION:", e)

    return {
        "poster": None,
        "rating": None
    }

# -----------------------------
# BOOKS
# -----------------------------

def get_book_data(title):

    if title in book_cache:
        return book_cache[title]

    clean_title = re.sub(r"[^\w\s]", "", title)

    url = "https://openlibrary.org/search.json"

    params = {
        "title": clean_title,
        "limit": 1
    }

    try:

        print(f"[BOOK] Searching for: {title}")

        response = requests.get(
            url,
            params=params,
            timeout=10
        )


        if response.status_code != 200:

            print("[BOOK] ERROR RESPONSE:")
            print(response.text)

            return {
                "poster": None,
                "rating": None
            }

        data = response.json()

        docs = data.get("docs")

        if docs:

            book = docs[0]

            cover_id = book.get("cover_i")

            poster = None

            if cover_id:
                poster = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"

            result = {
                "poster": poster,
                "rating": None
            }


            book_cache[title] = result
            return result

    except Exception as e:
        print("[BOOK] EXCEPTION:", e)

    return {
        "poster": None,
        "rating": None
    }

# -----------------------------
# MUSIC
# -----------------------------

def get_music_data(title):

    if title in music_cache:
        return music_cache[title]

    try:

        print(f"[ITUNES] Searching for: {title}")

        url = "https://itunes.apple.com/search"

        params = {
            "term": title,
            "limit": 1,
            "media": "music"
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:

            print("[ITUNES] ERROR RESPONSE:")
            print(response.text)

            return {"poster": None, "rating": None}

        data = response.json()

        if data.get("results"):

            song = data["results"][0]

            result = {
                "poster": song.get("artworkUrl100"),
                "rating": None
            }

            music_cache[title] = result
            return result

    except Exception as e:
        print("[ITUNES] EXCEPTION:", e)

    return {"poster": None, "rating": None}

# -----------------------------
# FLASK
# -----------------------------

app = Flask(__name__)

@app.route('/')
def main_page():
    return render_template('main_page.html')

@app.route('/recommend', methods=['POST'])
def recommend():

    try:

        data = request.get_json()

        user_input = data.get("query", "")
        favs = [row["title"] for row in data.get("favs")]
        dis = [row["title"] for row in data.get("dis")]

        print("\n========== USER INPUT ==========")
        print(user_input)
        print("================================\n")

        items = get_recommendations(
            llama3,
            user_input,
            favs,
            dis
        )

        results = []

        for item in items:

            title = item["title"]
            media_type = item["type"]

            print("MEDIA TYPE:", media_type)
            print("TITLE:", title)

            media_data = {
                "poster": None,
                "rating": None
            }

            if media_type == "movie":
                media_data = get_movie_data(title)

            elif media_type == "game":
                media_data = get_game_data(title)

            elif media_type == "book":
                media_data = get_book_data(title)

            elif media_type == "music":
                media_data = get_music_data(title)

            final_item = {
                "title": title,
                "type": media_type,
                "poster": media_data["poster"],
                "rating": media_data["rating"]
            }

            results.append(final_item)


        return jsonify(results)

    except Exception as e:

        print("RECOMMEND ERROR:")
        print(e)

        return jsonify({
            "error": str(e)
        }), 500

# -----------------------------
# RUN
# -----------------------------

if __name__ == '__main__':
    app.run(debug=True)