from flask import Flask, render_template, redirect, url_for
from predict import run_all_models_with_csv

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def run_predictions():
    intervals = ["daily", "weekly", "monthly"]
    results = []
    for interval in intervals:
        try:
            csv_path = f"data/nifty50_{interval}.csv"
            run_all_models_with_csv(csv_path, interval)
            results.append((interval, "Success"))
        except Exception as e:
            results.append((interval, f"Error: {str(e)}"))
    return render_template('results.html', results=results)

if __name__ == "__main__":
    app.run(debug=True)