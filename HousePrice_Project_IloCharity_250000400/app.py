from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# ---------------------------
# Load Trained Pipeline (robust)
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "Model", "house_price.pkl")

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    raise

# Load neighborhoods once
df = pd.read_csv(os.path.join(script_dir, "train.csv"))
neighborhoods = sorted(df["Neighborhood"].dropna().unique().tolist())

# ---------------------------
# Home Page
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html", neighborhoods=neighborhoods)

# ---------------------------
# Prediction Route
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        overall_qual = int(request.form["OverallQual"])
        gr_liv_area = float(request.form["GrLivArea"])
        total_bsmt_sf = float(request.form["TotalBsmtSF"])
        garage_cars = int(request.form["GarageCars"])
        year_built = int(request.form["YearBuilt"])
        neighborhood = request.form["Neighborhood"]

        input_data = pd.DataFrame([{
            "OverallQual": overall_qual,
            "GrLivArea": gr_liv_area,
            "TotalBsmtSF": total_bsmt_sf,
            "GarageCars": garage_cars,
            "YearBuilt": year_built,
            "Neighborhood": neighborhood,
        }])

        prediction = model.predict(input_data)[0]

        return render_template(
            "index.html",
            neighborhoods=neighborhoods,
            prediction_text=f"Estimated House Price: ${prediction:,.2f}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            neighborhoods=neighborhoods,
            prediction_text=f"Error: {str(e)}"
        )

# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
