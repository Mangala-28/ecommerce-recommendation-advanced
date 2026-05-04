from flask import Flask, request, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data.csv")
df = df[['User_ID', 'Product_ID', 'User_Rating']]
df.columns = ['user_id', 'product_id', 'rating']

# Create matrix
user_matrix = df.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)

# ---------------- AUTO NAMES ----------------
# Dynamic product names
product_names = {pid: f"Product {pid}" for pid in user_matrix.columns}

# Dynamic user names
user_names = {uid: f"Customer {uid}" for uid in user_matrix.index}

# ---------------- ML MODEL ----------------
def ml_recommend(user_id):
    user_data = user_matrix.loc[user_id]
    return user_data.sort_values(ascending=False).head(5)

# ---------------- DL MODEL (SIMULATED) ----------------
def dl_recommend(user_id):
    user_data = user_matrix.loc[user_id]
    scores = user_data + np.random.rand(len(user_data))
    return scores.sort_values(ascending=False).head(5)

# ---------------- QML MODEL (SIMULATED) ----------------
def qml_recommend(user_id):
    user_data = user_matrix.loc[user_id]
    scores = np.sin(user_data + np.random.rand(len(user_data)))
    return pd.Series(scores, index=user_data.index).sort_values(ascending=False).head(5)

# ---------------- ROUTE ----------------
@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = None
    customer_name = None
    model_type = None
    error = None

    if request.method == 'POST':
        user_id = request.form['user_id']
        model_type = request.form['model']

        if user_id not in user_matrix.index:
            error = "❌ User not found in dataset"
        else:
            customer_name = user_names.get(user_id, f"Customer {user_id}")

            if model_type == "ML":
                recs = ml_recommend(user_id)
            elif model_type == "DL":
                recs = dl_recommend(user_id)
            else:
                recs = qml_recommend(user_id)

            recommendations = [
                (product_names.get(pid, pid), round(float(score), 2))
                for pid, score in recs.items()
            ]

    return render_template('index.html',
                           recommendations=recommendations,
                           customer_name=customer_name,
                           model_type=model_type,
                           error=error)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)