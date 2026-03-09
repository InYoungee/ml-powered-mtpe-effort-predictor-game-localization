import app as st
import pandas as pd
import re
import difflib
import pickle
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# ===== PAGE CONFIG =====
st.set_page_config(
	page_title="MT Effort Predictor",
	page_icon="🔍 ",
	layout="wide"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background-color: #fafaf8; }

    h1, h2, h3 { font-family: 'DM Mono', monospace; }

    .metric-card {
        background: white;
        border: 1px solid #e8e8e4;
        border-radius: 8px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-label {
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #999;
        font-family: 'DM Mono', monospace;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 600;
        color: #1a1a1a;
        font-family: 'DM Mono', monospace;
    }
    .metric-sub {
        font-size: 12px;
        color: #aaa;
        margin-top: 4px;
    }

    .tag-light {
        background: #e8f5e9; color: #2e7d32;
        padding: 2px 10px; border-radius: 20px;
        font-size: 12px; font-weight: 500;
    }
    .tag-moderate {
        background: #fff8e1; color: #f57f17;
        padding: 2px 10px; border-radius: 20px;
        font-size: 12px; font-weight: 500;
    }
    .tag-heavy {
        background: #fce4ec; color: #c62828;
        padding: 2px 10px; border-radius: 20px;
        font-size: 12px; font-weight: 500;
    }

    .section-header {
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #999;
        border-bottom: 1px solid #e8e8e4;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    .info-box {
        background: #f5f5f0;
        border-left: 3px solid #1a1a1a;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        font-size: 13px;
        color: #555;
        margin-bottom: 16px;
    }

    .stDataFrame { font-size: 13px; }
    div[data-testid="stFileUploader"] { border: 1.5px dashed #ccc; border-radius: 8px; padding: 8px; }
</style>
""", unsafe_allow_html=True)


# ===== LOAD MODEL =====
@st.cache_resource
def load_model():
	with open('mt_effort_model.pkl', 'rb') as f:
		model = pickle.load(f)
	with open('feature_columns.pkl', 'rb') as f:
		feature_columns = pickle.load(f)
	return model, feature_columns


# ===== FEATURE EXTRACTION =====
def extract_features(row, mt_col):
	source = str(row['KO_Source'])
	category = row['Category_Consolidated']
	features = {}

	features['source_char_length'] = len(source)
	features['source_word_count'] = len(source.split())
	words = source.split()
	features['avg_char_per_word'] = len(source) / max(len(words), 1)
	features['has_placeholder'] = int(bool(re.search(r'\{[0-9]+\}|\%[sd]|\$\{[\w]+\}', source)))
	features['placeholder_count'] = len(re.findall(r'\{[0-9]+\}|\%[sd]|\$\{[\w]+\}', source))
	features['has_html'] = int(bool(re.search(r'<[^>]+>', source)))
	features['html_tag_count'] = len(re.findall(r'<[^>]+>', source))
	features['has_numbers'] = int(bool(re.search(r'\d', source)))
	features['number_count'] = len(re.findall(r'\d+', source))
	features['has_quotes'] = int('"' in source or "'" in source)
	features['has_colon'] = int(':' in source)
	features['has_slash'] = int('/' in source)
	features['exclamation_count'] = source.count('!')
	features['question_count'] = source.count('?')
	korean_chars = len(re.findall(r'[가-힣]', source))
	features['korean_char_ratio'] = korean_chars / max(len(source), 1)
	english_chars = len(re.findall(r'[a-zA-Z]', source))
	features['has_mixed_eng_ko'] = int(korean_chars > 0 and english_chars > 0)
	features['english_char_count'] = english_chars
	features['is_ui_system'] = int(category == 'UI_System')
	features['is_game_content'] = int(category == 'Game_Content')
	features['is_narrative'] = int(category == 'Narrative')
	features['is_marketing'] = int(category == 'Marketing')

	return features


def calc_similarity(str1, str2):
	s1 = str(str1).strip().lower()
	s2 = str(str2).strip().lower()
	return round(difflib.SequenceMatcher(None, s1, s2).ratio(), 3)


def run_prediction(df, mt_col, model, feature_columns):
	df = df.copy()
	df['Similarity'] = df.apply(
		lambda r: calc_similarity(r[mt_col], r['EN_Confirmed_Trans']), axis=1
	)
	feature_df = df.apply(lambda r: extract_features(r, mt_col), axis=1, result_type='expand')
	X = feature_df[feature_columns]
	df['Predicted_Effort'] = model.predict(X)
	df['Confidence'] = model.predict_proba(X).max(axis=1).round(3)
	return df


# ===== CHART COLORS =====
EFFORT_COLORS = {
	'light': '#4caf50',
	'moderate': '#ff9800',
	'heavy': '#e53935'
}

# ===== APP HEADER =====
st.markdown("# 🔍  MT Effort Predictor")
st.markdown("**ML-powered post-edit effort estimation for KO→EN game localization**")
st.markdown("---")

# ===== SIDEBAR =====
with st.sidebar:
	st.markdown("### Settings")
	mt_engine = st.radio(
		"MT Engine",
		options=["DeepL", "Google"],
		help="Select which MT engine column to use for prediction"
	)
	mt_col = "MT_DeepL" if mt_engine == "DeepL" else "MT_Google"

	st.markdown("---")
	st.markdown("### Required Columns")
	st.markdown("""
    Your Excel file must include:
    - `KO_Source`
    - `Category_Consolidated`
    - `MT_DeepL` or `MT_Google`
    - `EN_Confirmed_Trans`

    **Category values:**
    `UI_System` · `Game_Content`
    `Narrative` · `Marketing`
    """)

	st.markdown("---")
	st.markdown("### ℹ️ Note on MT Engine")
	st.markdown("""
    Switching MT engine affects **Similarity scores only**.

    Effort predictions (Light / Moderate / Heavy) are based on **KO source string features** — not the MT output itself. \n 
    Switching engines will not change predicted effort labels.
    """)
	st.markdown("""
    🟢 **Light** — Minor edits needed \n
    🟡 **Moderate** — Meaningful editing \n
    🔴 **Heavy** — Significant rewrite
    """)

# ===== MAIN CONTENT =====
try:
	model, feature_columns = load_model()
except:
	st.error(
		"Model files not found. Make sure `mt_effort_model.pkl` and `feature_columns.pkl` are in the same directory.")
	st.stop()

# File uploader
st.markdown('<div class="section-header">Upload File</div>', unsafe_allow_html=True)
st.markdown(
	'<div class="info-box">Upload an Excel file with KO source strings, MT output, and human translations. Your data is processed locally and never stored.</div>',
	unsafe_allow_html=True)

uploaded_file = st.file_uploader(
	"Drop your Excel file here",
	type=["xlsx"],
	label_visibility="collapsed"
)

if uploaded_file:
	try:
		df = pd.read_excel(uploaded_file)

		# Validate columns
		required_cols = ['KO_Source', 'Category_Consolidated', mt_col, 'EN_Confirmed_Trans']
		missing = [c for c in required_cols if c not in df.columns]

		if missing:
			st.error(
				f"Missing columns: {', '.join(missing)}. Please check your file or switch MT engine in the sidebar.")
			st.stop()

		# Run prediction
		with st.spinner("Running predictions..."):
			result_df = run_prediction(df, mt_col, model, feature_columns)

		st.success(f"✅ Predicted effort for {len(result_df)} strings using {mt_engine}")
		st.markdown("---")

		# ===== SUMMARY METRICS =====
		st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)

		counts = result_df['Predicted_Effort'].value_counts()
		avg_sim = result_df['Similarity'].mean()
		avg_conf = result_df['Confidence'].mean()

		col1, col2, col3, col4, col5 = st.columns(5)

		with col1:
			st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Total Strings</div>
                <div class="metric-value">{len(result_df)}</div>
            </div>""", unsafe_allow_html=True)
		with col2:
			st.markdown(f"""<div class="metric-card">
                <div class="metric-label">🟢 Light</div>
                <div class="metric-value" style="color:#4caf50">{counts.get('light', 0)}</div>
                <div class="metric-sub">{counts.get('light', 0) / len(result_df) * 100:.0f}%</div>
            </div>""", unsafe_allow_html=True)
		with col3:
			st.markdown(f"""<div class="metric-card">
                <div class="metric-label">🟡 Moderate</div>
                <div class="metric-value" style="color:#ff9800">{counts.get('moderate', 0)}</div>
                <div class="metric-sub">{counts.get('moderate', 0) / len(result_df) * 100:.0f}%</div>
            </div>""", unsafe_allow_html=True)
		with col4:
			st.markdown(f"""<div class="metric-card">
                <div class="metric-label">🔴 Heavy</div>
                <div class="metric-value" style="color:#e53935">{counts.get('heavy', 0)}</div>
                <div class="metric-sub">{counts.get('heavy', 0) / len(result_df) * 100:.0f}%</div>
            </div>""", unsafe_allow_html=True)
		with col5:
			st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Avg Similarity</div>
                <div class="metric-value">{avg_sim:.2f}</div>
                <div class="metric-sub">MT vs Human</div>
            </div>""", unsafe_allow_html=True)

		st.markdown("<br>", unsafe_allow_html=True)

		# ===== CHARTS =====
		st.markdown('<div class="section-header">Analysis</div>', unsafe_allow_html=True)

		col_a, col_b = st.columns(2)

		with col_a:
			# Effort distribution donut
			fig1 = go.Figure(data=[go.Pie(
				labels=list(counts.index),
				values=list(counts.values),
				hole=0.6,
				marker_colors=[EFFORT_COLORS.get(l, '#999') for l in counts.index],
				textinfo='label+percent',
				textfont_size=13,
			)])
			fig1.update_layout(
				title=dict(text="Effort Distribution", font=dict(size=13, family="DM Mono")),
				showlegend=False,
				height=300,
				margin=dict(t=40, b=10, l=10, r=10),
				paper_bgcolor='white',
				plot_bgcolor='white'
			)
			st.plotly_chart(fig1, use_container_width=True)

		with col_b:
			# Effort by category stacked bar
			cat_effort = pd.crosstab(
				result_df['Category_Consolidated'],
				result_df['Predicted_Effort']
			).reset_index()

			fig2 = go.Figure()
			for effort in ['light', 'moderate', 'heavy']:
				if effort in cat_effort.columns:
					fig2.add_trace(go.Bar(
						name=effort.capitalize(),
						x=cat_effort['Category_Consolidated'],
						y=cat_effort[effort],
						marker_color=EFFORT_COLORS[effort],
					))
			fig2.update_layout(
				title=dict(text="Effort by Category", font=dict(size=13, family="DM Mono")),
				barmode='stack',
				height=300,
				margin=dict(t=80, b=10, l=10, r=10),
				paper_bgcolor='white',
				plot_bgcolor='white',
				legend=dict(orientation='h', yanchor='top', y=1.25, xanchor='center', x=0.5),
				xaxis=dict(tickfont=dict(size=11)),
				yaxis=dict(tickfont=dict(size=11))
			)
			st.plotly_chart(fig2, use_container_width=True)

		col_c, col_d = st.columns(2)

		with col_c:
			# Similarity score distribution by effort
			fig3 = go.Figure()
			for effort in ['light', 'moderate', 'heavy']:
				subset = result_df[result_df['Predicted_Effort'] == effort]['Similarity']
				if not subset.empty:
					fig3.add_trace(go.Box(
						y=subset,
						name=effort.capitalize(),
						marker_color=EFFORT_COLORS[effort],
						boxmean=True
					))
			fig3.update_layout(
				title=dict(text="Similarity Score by Effort", font=dict(size=13, family="DM Mono")),
				height=300,
				margin=dict(t=40, b=10, l=10, r=10),
				paper_bgcolor='white',
				plot_bgcolor='white',
				showlegend=False,
				yaxis=dict(title="Similarity", tickfont=dict(size=11)),
			)
			st.plotly_chart(fig3, use_container_width=True)

		with col_d:
			# Confidence distribution
			fig4 = px.histogram(
				result_df,
				x='Confidence',
				color='Predicted_Effort',
				color_discrete_map=EFFORT_COLORS,
				nbins=20,
				barmode='overlay',
				opacity=0.75
			)
			fig4.update_layout(
				title=dict(text="Model Confidence Distribution", font=dict(size=13, family="DM Mono")),
				height=300,
				margin=dict(t=80, b=10, l=10, r=10),
				paper_bgcolor='white',
				plot_bgcolor='white',
				legend=dict(title='', orientation='h', yanchor='top', y=1.25, xanchor='center', x=0.5),
				xaxis=dict(title="Confidence", tickfont=dict(size=11)),
				yaxis=dict(title="Count", tickfont=dict(size=11))
			)
			st.plotly_chart(fig4, use_container_width=True)

		# ===== RESULTS TABLE =====
		st.markdown("---")
		st.markdown('<div class="section-header">Predictions</div>', unsafe_allow_html=True)

		# Filter
		filter_options = ['All', 'light', 'moderate', 'heavy']
		effort_filter_choice = st.selectbox(
			"Filter by effort label",
			options=filter_options,
			index=0
		)
		effort_filter = ['light', 'moderate', 'heavy'] if effort_filter_choice == 'All' else [effort_filter_choice]

		filtered = result_df[result_df['Predicted_Effort'].isin(effort_filter)]

		display_cols = ['KO_Source', 'Category_Consolidated', mt_col,
						'EN_Confirmed_Trans', 'Similarity', 'Predicted_Effort', 'Confidence']
		available_cols = [c for c in display_cols if c in filtered.columns]

		st.dataframe(
			filtered[available_cols].reset_index(drop=True),
			use_container_width=True,
			height=400,
			column_config={
				'KO_Source': st.column_config.TextColumn('KO Source', width='large'),
				mt_col: st.column_config.TextColumn('MT Output', width='large'),
				'EN_Confirmed_Trans': st.column_config.TextColumn('Human Translation', width='large'),
				'Similarity': st.column_config.ProgressColumn('Similarity', min_value=0, max_value=1, format="%.3f"),
				'Predicted_Effort': st.column_config.TextColumn('Effort'),
				'Confidence': st.column_config.ProgressColumn('Confidence', min_value=0, max_value=1, format="%.3f"),
				'Category_Consolidated': st.column_config.TextColumn('Category', width='small'),
			}
		)

		st.caption(f"Showing {len(filtered)} of {len(result_df)} strings")

	except Exception as e:
		st.error(f"Error processing file: {e}")

else:
	# Empty state
	st.markdown("<br>", unsafe_allow_html=True)
	st.markdown("""
    <div style="text-align:center; padding: 60px 0; color: #bbb;">
        <div style="font-size: 48px; margin-bottom: 16px;">🔍 </div>
        <div style="font-family: 'DM Mono', monospace; font-size: 14px; letter-spacing: 0.05em;">
            Upload an Excel file to get started
        </div>
    </div>
    """, unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#bbb; font-size:12px; font-family:'DM Mono',monospace;">
    ML-Powered MT Post-Edit Effort Predictor · KO→EN Game Localization · 
    Trained on 819 strings from 7 games
</div>
""", unsafe_allow_html=True)