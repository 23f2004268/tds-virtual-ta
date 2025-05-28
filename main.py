from fastapi import FastAPI
from pydantic import BaseModel
import base64
import json
import os
from transformers import AutoTokenizer
from PIL import Image
import pytesseract
import io

app = FastAPI()

# Load scraped Discourse posts
with open("discourse_posts.json", "r", encoding="utf-8") as f:
    discourse_data = json.load(f)

# Load tokenizer (GPT-2 is a good proxy for GPT-3.5-turbo)
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

class Query(BaseModel):
    question: str
    image: str | None = None

def estimate_token_cost(text: str, cost_per_million: float = 0.5):
    tokens = tokenizer.encode(text)
    num_tokens = len(tokens)
    cost_in_cents = (num_tokens / 1_000_000) * cost_per_million * 100
    return round(cost_in_cents, 7)

def extract_text_from_image(base64_string):
    try:
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(image, lang='eng+jpn')
    except Exception:
        return ""

@app.post("/api/")
async def virtual_ta(query: Query):
    text_query = query.question.strip()

    # If image is provided, extract text and append
    if query.image:
        extracted_text = extract_text_from_image(query.image)
        text_query += " " + extracted_text

    # Handle token cost questions
    if "token" in text_query.lower() and "cost" in text_query.lower():
        import re
        jp_match = re.search(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]+', text_query)
        if jp_match:
            jp_text = jp_match.group(0)
        else:
            jp_text = text_query
        cost = estimate_token_cost(jp_text)
        return {
            "answer": f"The input cost is approximately {cost:.7f} cents.",
            "links": []
        }

    # Keyword match search for other questions
    keywords = [word.lower() for word in text_query.split() if len(word) > 3]
    matches = []
    for post in discourse_data:
        content = post.get("content", "").lower()
        title = post.get("topic_title", "").lower()
        if any(kw in content or kw in title for kw in keywords):
            matches.append(post)

    # Improved answer selection and formatting
    if matches:
        # Find the first post with non-empty content, else fallback to title
        for post in matches:
            if post.get("content", "").strip():
                main_answer = post["content"].strip()
                break
        else:
            main_answer = matches[0].get("topic_title", "Relevant post")
        # If the answer is too long, summarize (simple truncation for now)
        if len(main_answer) > 600:
            main_answer = main_answer[:600] + "..."
        # If answer is still empty, use a fallback
        if not main_answer.strip():
            main_answer = "See the related Discourse post(s) below."
        links = []
        for match in matches[:3]:
            links.append({
                "url": match["url"],
                "text": match.get("topic_title", "Related post")
            })
        # Always return answer and links in required format
        return {
            "answer": main_answer,
            "links": links
        }

    # Fallback: no relevant post found
    return {
        "answer": "Sorry, I couldn't find anything relevant from Discourse. Please try rephrasing your question or provide more details.",
        "links": []
    }
