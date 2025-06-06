import tkinter as tk
from tkinter import messagebox
import joblib

# Wczytaj zapisany model i vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict():
    text = input_text.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Brak tekstu", "Wprowadź wiadomość do analizy.")
        return
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    label = "✅ Prawdziwa wiadomość" if prediction == 1 else "❌ Fałszywa wiadomość"
    result_label.config(text=f"Wynik: {label}")

# GUI
root = tk.Tk()
root.title("Fake News Detector")

tk.Label(root, text="Wklej wiadomość do analizy:", font=("Arial", 12)).pack(pady=5)
input_text = tk.Text(root, height=10, width=60)
input_text.pack()

tk.Button(root, text="Sprawdź", command=predict, font=("Arial", 12)).pack(pady=10)

result_label = tk.Label(root, text="Wynik: ", font=("Arial", 12, "bold"))
result_label.pack(pady=5)

root.mainloop()
