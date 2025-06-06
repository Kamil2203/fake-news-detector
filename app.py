from flask import Flask, render_template, request
import joblib
import sqlite3

app = Flask(__name__)

model = joblib.load("fake/model.pkl")
vectorizer = joblib.load("fake/vectorizer.pkl")

def log_to_db(message, prediction, confidence):
    conn = sqlite3.connect("fake_news_log.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO logs (message, prediction, confidence) VALUES (?, ?, ?)", (message, prediction, confidence))
    conn.commit()
    conn.close()

def get_logs():
    conn = sqlite3.connect("fake_news_log.db")
    cursor = conn.cursor()
    cursor.execute("SELECT message, prediction, confidence, timestamp FROM logs ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        text = request.form["message"]
        if len(text.strip().split()) < 5:
            result = "⚠️ Podaj dłuższy tekst do analizy (minimum 5 słów)"
        else:
            vec = vectorizer.transform([text])
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            confidence = round(max(proba) * 100, 2)
            label = "✅ Prawdziwa wiadomość" if pred == 1 else "❌ Fałszywa wiadomość"
            result = f"{label} (pewność: {confidence}%)"
            log_to_db(text, label, confidence)

    return render_template("index.html", result=result)

@app.route("/historia")
def historia():
    logs = get_logs()
    return render_template("historia.html", logs=logs)

if __name__ == "__main__":
    app.run(debug=True)
