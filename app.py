from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px

app = Flask(__name__)

# === Load model dan scaler ===
model = load_model("model_sales.h5")
scaler = joblib.load("scaler_sales.pkl")

# === Load data ===
df = pd.read_csv("Online Sales Data.csv")
df['Date'] = pd.to_datetime(df['Date'])

# === Dropdown options ===
categories = df['Product Category'].unique().tolist()
regions = df['Region'].unique().tolist()

def get_filtered_data(category=None, region=None):
    filtered = df.copy()
    if category and category != "Semua":
        filtered = filtered[filtered['Product Category'] == category]
    if region and region != "Semua":
        filtered = filtered[filtered['Region'] == region]
    daily_sales = filtered.groupby('Date').agg({
        'Total Revenue': 'sum',
        'Units Sold': 'sum',
        'Unit Price': 'mean'
    }).reset_index()
    return daily_sales

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_category = request.form.get('category', 'Semua')
    selected_region = request.form.get('region', 'Semua')

    # Data sesuai filter
    daily_sales = get_filtered_data(selected_category, selected_region)

    # Pesan kosong
    message = None
    if daily_sales.empty:
        message = f"Tidak ada data untuk kategori <b>{selected_category}</b> di wilayah <b>{selected_region}</b>."
        fig = go.Figure()
        fig.update_layout(
            title="Data Tidak Ditemukan",
            template='plotly_white',
            xaxis_title="Tanggal",
            yaxis_title="Total Revenue (USD)",
            height=400
        )
        graph_html = fig.to_html(full_html=False)
        total_days = avg_revenue = max_revenue = 0
    else:
        fig = px.line(
            daily_sales,
            x='Date',
            y='Total Revenue',
            title=f"Tren Pendapatan Harian ({selected_category} | {selected_region})",
            template='plotly_white'
        )
        fig.update_traces(line_color='#1e3c72')
        graph_html = fig.to_html(full_html=False)

        total_days = len(daily_sales)
        avg_revenue = daily_sales['Total Revenue'].mean()
        max_revenue = daily_sales['Total Revenue'].max()

    return render_template(
        'index.html',
        graph_html=graph_html,
        total_days=total_days,
        avg_revenue=avg_revenue,
        max_revenue=max_revenue,
        categories=['Semua'] + categories,
        regions=['Semua'] + regions,
        selected_category=selected_category,
        selected_region=selected_region,
        message=message
    )


@app.route('/prediksi', methods=['POST'])
def prediksi():
    selected_category = request.form.get('category', 'Semua')
    selected_region = request.form.get('region', 'Semua')
    daily_sales = get_filtered_data(selected_category, selected_region)

    if daily_sales.empty:
        return "<h3>❌ Tidak ada data untuk prediksi kategori/wilayah ini.</h3>"

    if len(daily_sales) < 40:
        return "<h3>⚠️ Data terlalu sedikit untuk prediksi. Pilih kategori/wilayah lain.</h3>"

    scaled_data = scaler.transform(daily_sales[['Total Revenue', 'Units Sold', 'Unit Price']])
    timesteps = 30
    X = []
    for i in range(timesteps, len(scaled_data)):
        X.append(scaled_data[i - timesteps:i])
    X = np.array(X)

    pred = model.predict(X)
    pred = scaler.inverse_transform(
        np.concatenate((pred, np.zeros((len(pred), 2))), axis=1)
    )[:, 0]

    df_pred = daily_sales.iloc[timesteps:].copy()
    df_pred['Predicted Revenue'] = pred

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_pred['Date'], y=df_pred['Total Revenue'],
        mode='lines', name='Aktual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(
        x=df_pred['Date'], y=df_pred['Predicted Revenue'],
        mode='lines', name='Prediksi LSTM', line=dict(color='orange')))
    fig.update_layout(
        title=f"Prediksi Pendapatan ({selected_category} | {selected_region})",
        xaxis_title='Tanggal',
        yaxis_title='Total Revenue (USD)',
        template='plotly_white',
        height=500
    )
    graph_html = fig.to_html(full_html=False)

    return render_template(
        'prediksi.html',
        graph_html=graph_html,
        selected_category=selected_category,
        selected_region=selected_region
    )


if __name__ == "__main__":
    app.run(debug=True)
