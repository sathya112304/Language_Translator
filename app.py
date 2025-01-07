from flask import Flask, request, jsonify, render_template
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

# Load the model and tokenizer once to save time
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text):
    input_tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**input_tokens)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    english_text = data.get("text", "")
    if not english_text:
        return jsonify({"error": "No text provided"}), 400
    translated_text = translate_text(english_text)
    return jsonify({"translated_text": translated_text})

if __name__ == "__main__":
    app.run(debug=True)
