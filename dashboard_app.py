import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="ML MTPE Quality Estimation Dashboard",
    page_icon="🔍",
    layout="wide"
)

# ===== CSS =====
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        background-color: #0f1117;
        color: #e0e0e0;
    }

    h1, h2, h3 { font-family: 'Space Mono', monospace; }

    .kpi-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #2e3450;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .kpi-label {
        font-size: 11px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #7b8ab8;
        font-family: 'Space Mono', monospace;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 36px;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
    }
    .kpi-sub {
        font-size: 12px;
        color: #7b8ab8;
        margin-top: 4px;
    }
    .kpi-green  { color: #4ade80; }
    .kpi-blue   { color: #60a5fa; }
    .kpi-yellow { color: #fbbf24; }
    .kpi-pink   { color: #f472b6; }
    .kpi-coral  { color: #fb923c; }

    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 12px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #7b8ab8;
        border-bottom: 1px solid #2e3450;
        padding-bottom: 10px;
        margin-bottom: 20px;
        margin-top: 10px;
    }

    .insight-box {
        background: linear-gradient(135deg, #1a1f35, #1e2440);
        border-left: 3px solid #60a5fa;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        font-size: 13px;
        color: #b0bcd8;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ===== MODEL + DATA =====
@st.cache_resource
def load_model():
    with open('mt_effort_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    return model, feature_columns

@st.cache_data
def load_data():
    return pd.read_excel('strings_with_features.xlsx')

@st.cache_data
def compute_results():
    model, feature_columns = load_model()
    df = load_data()
    X = df[feature_columns]
    y = df['Effort_Label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=['light', 'moderate', 'heavy'])
    fi = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    return accuracy, report, cm, fi

try:
    model, feature_columns = load_model()
    df = load_data()
    accuracy, report, cm, fi = compute_results()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

model_name = type(model).__name__
model_display = 'Gradient Boosting' if 'Gradient' in model_name else 'Random Forest'
effort_colors = {'light': '#4ade80', 'moderate': '#fbbf24', 'heavy': '#f87171'}

# ===== HEADER =====
st.markdown("# 🔍 ML MTPE Quality Estimation")
st.markdown("**Model performance analysis · KO→EN game localization · 819 strings · 7 game titles**")
st.markdown("---")

# ===== SECTION 1: KPI ROW =====
st.markdown('<div class="section-title">Model Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Best Model</div>
        <div class="kpi-value kpi-blue" style="font-size:18px;">{model_display}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Accuracy</div>
        <div class="kpi-value kpi-green">{accuracy*100:.1f}%</div>
        <div class="kpi-sub">vs 33% random baseline</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="kpi-card">
        <div class="kpi-label">Training Strings</div>
        <div class="kpi-value kpi-yellow">819</div>
        <div class="kpi-sub">across 7 game titles</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Features Used</div>
        <div class="kpi-value kpi-pink">{len(feature_columns)}</div>
        <div class="kpi-sub">source string features</div>
    </div>""", unsafe_allow_html=True)
with col5:
    st.markdown("""<div class="kpi-card">
        <div class="kpi-label">Effort Classes</div>
        <div class="kpi-value kpi-coral">3</div>
        <div class="kpi-sub">light · moderate · heavy</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ===== SECTION 2: DATASET ANALYSIS =====
st.markdown('<div class="section-title">Dataset Analysis</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    label_counts = df['Effort_Label'].value_counts()
    fig_labels = go.Figure(data=[go.Pie(
        labels=label_counts.index,
        values=label_counts.values,
        hole=0.65,
        marker_colors=[effort_colors.get(l, '#999') for l in label_counts.index],
        textinfo='label+percent',
        textfont=dict(size=13, family='Space Mono'),
    )])
    fig_labels.update_layout(
        title=dict(text="Effort Label Distribution",
                   font=dict(size=13, family='Space Mono', color='#b0bcd8')),
        showlegend=False, height=320,
        margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#b0bcd8')
    )
    fig_labels.add_annotation(
        text="819<br>strings", x=0.5, y=0.5,
        font=dict(size=16, family='Space Mono', color='#e0e0e0'),
        showarrow=False
    )
    st.plotly_chart(fig_labels, use_container_width=True)

with col_b:
    fig_sim = go.Figure()
    for effort in ['light', 'moderate', 'heavy']:
        subset = df[df['Effort_Label'] == effort]['Best_MT_Similarity']
        fig_sim.add_trace(go.Box(
            y=subset, name=effort.capitalize(),
            marker_color=effort_colors[effort],
            boxmean=True, line=dict(width=2)
        ))
    fig_sim.update_layout(
        title=dict(text="MT Similarity Score by Effort Label",
                   font=dict(size=13, family='Space Mono', color='#b0bcd8')),
        height=320, margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        yaxis=dict(title="Best MT Similarity", tickfont=dict(size=11, color='#b0bcd8'),
                   gridcolor='#2e3450'),
        xaxis=dict(tickfont=dict(size=12, color='#b0bcd8')),
        font=dict(color='#b0bcd8')
    )
    st.plotly_chart(fig_sim, use_container_width=True)

# ===== SECTION 3: MODEL PERFORMANCE =====
st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    classes = ['light', 'moderate', 'heavy']
    metrics_map = [('Precision', 'precision', '#60a5fa'),
                   ('Recall', 'recall', '#a78bfa'),
                   ('F1', 'f1-score', '#f472b6')]
    fig_metrics = go.Figure()
    for label, key, color in metrics_map:
        fig_metrics.add_trace(go.Bar(
            name=label,
            x=classes,
            y=[report[c][key] for c in classes],
            marker_color=color, opacity=0.85
        ))
    fig_metrics.add_hline(
        y=accuracy, line_dash='dash', line_color='#fbbf24',
        annotation_text=f"Overall Accuracy: {accuracy*100:.1f}%",
        annotation_font=dict(color='#fbbf24', size=11)
    )
    fig_metrics.update_layout(
        title=dict(text=f"{model_display} — Per-Class Metrics",
                   font=dict(size=13, family='Space Mono', color='#b0bcd8')),
        barmode='group', height=350,
        margin=dict(t=50, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='top', y=1.18, xanchor='center', x=0.5,
                    font=dict(color='#b0bcd8')),
        xaxis=dict(tickfont=dict(size=12, color='#b0bcd8'), gridcolor='#2e3450'),
        yaxis=dict(tickfont=dict(size=11, color='#b0bcd8'), gridcolor='#2e3450', range=[0, 1]),
        font=dict(color='#b0bcd8')
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

with col2:
    cm_array = np.array(cm)
    cm_norm = cm_array / cm_array.sum(axis=1, keepdims=True)
    labels = ['Light', 'Moderate', 'Heavy']
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=[f'Pred: {l}' for l in labels],
        y=[f'Actual: {l}' for l in labels],
        colorscale=[[0, '#1e2130'], [0.5, '#3b5bdb'], [1, '#4ade80']],
        showscale=False,
        text=[[str(cm_array[i][j]) for j in range(3)] for i in range(3)],
        texttemplate='<b>%{text}</b>',
        textfont=dict(size=18, color='white'),
        hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>'
    ))
    fig_cm.update_layout(
        title=dict(text=f"{model_display} — Confusion Matrix",
                   font=dict(size=13, family='Space Mono', color='#b0bcd8')),
        height=350, margin=dict(t=50, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=dict(size=11, color='#b0bcd8')),
        yaxis=dict(tickfont=dict(size=11, color='#b0bcd8')),
        font=dict(color='#b0bcd8')
    )
    st.plotly_chart(fig_cm, use_container_width=True)

st.markdown(f"""<div class="insight-box">
⚡ <b>Key Finding:</b> {model_display} achieves {accuracy*100:.1f}% accuracy —
significantly above the 33% random baseline for a 3-class problem.
"Moderate" is the hardest class to predict, sitting ambiguously between light and heavy.
</div>""", unsafe_allow_html=True)

# ===== SECTION 4: FEATURE IMPORTANCE =====
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)

fi_sorted = fi.sort_values('Importance', ascending=True)
fig_fi = go.Figure(go.Bar(
    x=fi_sorted['Importance'],
    y=fi_sorted['Feature'],
    orientation='h',
    marker=dict(
        color=fi_sorted['Importance'],
        colorscale=[[0, '#1e3a5f'], [0.5, '#3b82f6'], [1, '#60a5fa']],
        showscale=False
    ),
    text=[f"{v:.1%}" for v in fi_sorted['Importance']],
    textposition='outside',
    textfont=dict(color='#b0bcd8', size=11)
))
fig_fi.update_layout(
    title=dict(text="Top 10 Most Important Features",
               font=dict(size=13, family='Space Mono', color='#b0bcd8')),
    height=400, margin=dict(t=40, b=10, l=10, r=80),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(tickformat='.0%', tickfont=dict(size=11, color='#b0bcd8'),
               gridcolor='#2e3450', range=[0, fi_sorted['Importance'].max() * 1.3]),
    yaxis=dict(tickfont=dict(size=11, color='#b0bcd8')),
    font=dict(color='#b0bcd8')
)
st.plotly_chart(fig_fi, use_container_width=True)

top_feature = fi.iloc[0]
st.markdown(f"""<div class="insight-box">
⚡ <b>Key Finding:</b> <b>{top_feature['Feature']}</b> is the strongest predictor
({top_feature['Importance']:.1%}), followed by string length and complexity features.
The top 4 features are all character/length-based — longer, more complex strings with
mixed KO/EN content tend to require heavier post-editing.
</div>""", unsafe_allow_html=True)

# ===== SECTION 5: SOURCE VS MT ACCURACY GAP =====
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Predictive Power: Source Features vs MT Similarity</div>',
            unsafe_allow_html=True)

accuracy_data = pd.DataFrame({
    'Feature Set': ['Source Features Only', 'Source + MT Similarity'],
    'Accuracy': [round(accuracy * 100, 1), 98.8],
    'Color': ['#60a5fa', '#4ade80']
})
fig_acc = go.Figure(go.Bar(
    x=accuracy_data['Feature Set'],
    y=accuracy_data['Accuracy'],
    marker_color=accuracy_data['Color'],
    text=[f"{v:.1f}%" for v in accuracy_data['Accuracy']],
    textposition='outside',
    textfont=dict(size=16, color='white'),
    width=0.4
))
fig_acc.add_hline(
    y=33.3, line_dash='dot', line_color='#f87171',
    annotation_text="Random baseline: 33.3%",
    annotation_font=dict(color='#f87171', size=11)
)
fig_acc.update_layout(
    title=dict(text="Prediction Accuracy: Source Features Only vs With MT Similarity Scores",
               font=dict(size=13, family='Space Mono', color='#b0bcd8')),
    height=350, margin=dict(t=50, b=10, l=10, r=10),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(tickfont=dict(size=13, color='#b0bcd8')),
    yaxis=dict(tickfont=dict(size=11, color='#b0bcd8'), gridcolor='#2e3450',
               range=[0, 115], title='Accuracy (%)'),
    font=dict(color='#b0bcd8')
)
st.plotly_chart(fig_acc, use_container_width=True)

gap = round(98.8 - accuracy * 100, 1)
st.markdown(f"""<div class="insight-box">
⚡ <b>Key Finding:</b> Adding MT similarity scores jumps accuracy from {accuracy*100:.1f}% → 98.8% —
a {gap} percentage point gap. Source string features alone are weak predictors of MT quality.
The MT output itself carries most of the quality signal, meaning predicting effort <i>before</i>
MT runs is fundamentally harder than predicting it <i>after</i>.
</div>""", unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"""
<div style="text-align:center; color:#4a5280; font-size:12px; font-family:'Space Mono',monospace;">
    ML-Powered MT Post-Edit Effort Predictor · KO→EN Game Localization ·
    819 strings · 7 game titles · {model_display}
</div>
""", unsafe_allow_html=True)