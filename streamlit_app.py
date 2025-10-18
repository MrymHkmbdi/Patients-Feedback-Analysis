import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from PIL import Image


# PAGE CONFIG
st.set_page_config(
    page_title="Patient Feedback Analysis Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# DATA LOADING
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/processed/analyzed_data.csv')
        if df['sentiment'].isna().any():
            st.warning("⚠️ Missing sentiment values replaced with 'neutral'")
            df['sentiment'] = df['sentiment'].fillna('neutral')
        df['sentiment'] = df['sentiment'].astype(str).str.lower()

        insights = pd.read_csv('outputs/complete_cluster_insights.csv')
        recommendations = pd.read_csv('outputs/recommendations/recommendations.csv')
        top_10 = pd.read_csv('outputs/top_10_recurring_issues.csv')
        embeddings_2d = np.load('data/embeddings/umap_2d.npy')
        return df, insights, recommendations, top_10, embeddings_2d
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None, None, None, None

with st.spinner("Loading data..."):
    df, insights_df, recommendations_df, top_10_df, embeddings_2d = load_data()
if df is None:
    st.stop()

# SIDEBAR
st.sidebar.title("🏥 Navigation")
languages = ['All'] + sorted(df['target_language'].unique().tolist())
selected_language = st.sidebar.selectbox("Language", languages)

sentiments = ['All'] + sorted(df['sentiment'].unique().tolist())
selected_sentiment = st.sidebar.selectbox("Sentiment", sentiments)

severities = ["All", "Critical", "High", "Medium", "Low"]
selected_severity = st.sidebar.selectbox("Severity", severities)

df_filtered = df.copy()
if selected_language != 'All':
    df_filtered = df_filtered[df_filtered['target_language'] == selected_language]
if selected_sentiment != 'All':
    df_filtered = df_filtered[df_filtered['sentiment'] == selected_sentiment]
if selected_severity != "All":
    if {'cluster_id', 'severity_level'}.issubset(recommendations_df.columns):
        allowed_clusters = set(
            recommendations_df.loc[
                recommendations_df['severity_level'] == selected_severity, 'cluster_id'
            ].astype(int)
        )
        df_filtered = df_filtered[df_filtered['cluster'].astype(int).isin(allowed_clusters)]
        if 'cluster_id' in insights_df.columns:
            insights_df = insights_df[insights_df['cluster_id'].astype(int).isin(allowed_clusters)]
        if 'cluster_id' in top_10_df.columns:
            top_10_df = top_10_df[top_10_df['cluster_id'].astype(int).isin(allowed_clusters)]
        if df_filtered.empty:
            st.sidebar.info("No data matches this severity filter.")
    else:
        st.sidebar.warning("Severity filter unavailable: 'cluster_id' or 'severity_level' missing in recommendations.")

# if page == "Overview":
st.markdown('<div class="main-header">Patient Feedback Analysis Dashboard</div>', unsafe_allow_html=True)

metrics_html = f"""
<div style='display:flex; justify-content:center; gap:60px; text-align:center;'>
    <div>
        <h4>Total Reviews</h4>
        <h2>{len(df_filtered):,}</h2>
    </div>
    <div>
        <h4>Negative Sentiment</h4>
        <h2>{(df_filtered["sentiment"].eq("negative").mean() * 100):.1f}%</h2>
    </div>
    <div>
        <h4>Average Rating</h4>
        <h2>{(df_filtered["rating_normalized"].mean() * 1.2):.2f}/5.0</h2>
    </div>
</div>
"""
st.markdown(metrics_html, unsafe_allow_html=True)
cluster_stats = []
for cluster_id in sorted(df_filtered['cluster'].unique()):
    cluster_data = df_filtered[df_filtered['cluster'] == cluster_id]
    sentiment_dist = cluster_data['sentiment'].value_counts(normalize=True) * 100
    stats = {
        'Cluster ID': int(cluster_id),
        'Size': int(len(cluster_data)),
        'Negative %': float(sentiment_dist.get('negative', 0)),
        'Neutral %': float(sentiment_dist.get('neutral', 0)),
        'Positive %': float(sentiment_dist.get('positive', 0))
    }
    if 'rating_normalized' in cluster_data.columns and cluster_data['rating_normalized'].notna().any():
        stats['Avg Rating'] = float((cluster_data['rating_normalized'] * 1.2).mean())
    cluster_stats.append(stats)
cluster_stats_df = pd.DataFrame(cluster_stats).sort_values('Cluster ID')

# Tabs
tab_overview, tab_details = st.tabs(["Overview", "Cluster details"])

# Overview Tab
with tab_overview:
    # Cluster Size Chart
    st.subheader("📈 Cluster Size by Cluster")
    fig_size_bar = px.bar(
        cluster_stats_df,
        x='Cluster ID',
        y='Size',
        text='Size',
        labels={'Cluster ID': 'Cluster', 'Size': 'Number of Reviews'},
    )
    fig_size_bar.update_traces(
        marker_color='#e0fe74',
        texttemplate='%{text}',
        textposition='outside'
    )
    fig_size_bar.update_layout(
        height=450,
        xaxis=dict(type='category', title='Cluster ID'),
        yaxis=dict(title='Cluster Size'),
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    st.plotly_chart(fig_size_bar, use_container_width=True)

    if 'rating_normalized' in df.columns:
        st.subheader("⭐ Average Rating by Cluster")
        avg_rating = df_filtered.groupby('cluster')['rating_normalized'].mean().reset_index()
        avg_rating['rating_normalized'] = avg_rating['rating_normalized'] * 1.2
        fig = px.bar(
            avg_rating,
            x='cluster',
            y='rating_normalized',
            labels={'cluster': 'Cluster ID', 'rating_normalized': 'Avg Rating'},
            color='rating_normalized',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    import plotly.express as px

    # Make sure you have your insights_df from your code above
    # It should include columns: ['cluster_id', 'impact_score']

    st.subheader("💥 Cluster Impact Scores")

    fig_impact = px.bar(
        insights_df,
        x='cluster_id',
        y='impact_score',
        text='impact_score',
        color='impact_score',
        color_continuous_scale='Reds',
        labels={'cluster_id': 'Cluster ID', 'impact_score': 'Impact Score'}
    )

    # Make bars look neat and labels clear
    fig_impact.update_traces(
        texttemplate='%{text:.1f}',
        textposition='outside'
    )
    fig_impact.update_layout(
        height=450,
        xaxis=dict(type='category', title='Cluster ID'),
        yaxis=dict(title='Impact Score'),
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    st.plotly_chart(fig_impact, use_container_width=True)

    st.markdown("---")
    st.subheader("🔥 Top 5 Recurring Issues")

    top5 = (
        top_10_df.copy()
        .sort_values('rank', ascending=True)
        .head(5)
    )

    for _, r in top5.iterrows():
        header = f"#{int(r['rank'])} • Cluster {int(r['cluster_id'])} • {int(r['size'])} patients • {r['percentage']:.1f}%"
        with st.expander(header, expanded=False):
            # Small metrics row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Negative %", f"{float(r['negative_pct']):.1f}%")
            with m2:
                st.metric("Neutral %", f"{float(r['neutral_pct']):.1f}%")
            with m3:
                st.metric("Positive %", f"{float(r['positive_pct']):.1f}%")
            with m4:
                if pd.notna(r.get('avg_rating', np.nan)):
                    st.metric("Avg Rating ×1.2", f"{float(r['avg_rating']) * 1.2:.2f}/5.0")
                else:
                    st.metric("Avg Rating ×1.2", "N/A")

            st.markdown("**Summary**")
            st.write(str(r['summary']))

            # Optional: impact score
            if 'impact_score' in r and pd.notna(r['impact_score']):
                st.caption(f"Impact score: {float(r['impact_score']):.1f}")

    st.subheader("📈 KMeans Clusters (UMAP Projection)")
    image = Image.open('//outputs/figures/06_kmeans_clusters_umap.png')
    st.image(
        image,
        use_container_width=True,)



# Cluster Details Tab
with tab_details:
    st.subheader("🔍 Detailed Cluster View")

    # Cluster selector
    available_clusters = sorted([int(c) for c in df_filtered['cluster'].unique()])
    selected_cluster_detail = st.selectbox(
        "Select a cluster to view details",
        available_clusters,
        format_func=lambda x: f"Cluster {x}"
    )

    cluster_detail_data = df_filtered[df_filtered['cluster'] == selected_cluster_detail].copy()

    insight_row = None
    if 'cluster_id' in insights_df.columns:
        match = insights_df[insights_df['cluster_id'] == selected_cluster_detail]
        if not match.empty:
            insight_row = match.iloc[0]

    total_current = len(df_filtered)
    cluster_size = len(cluster_detail_data)
    share_pct = (cluster_size / total_current * 100) if total_current else 0.0

    recs = recommendations_df.copy()
    if 'cluster_id' in recs.columns:
        recs = recs[recs['cluster_id'] == selected_cluster_detail]
    else:
        recs = pd.DataFrame()

    if recs.empty:
        st.info("No recommendations available for this cluster.")
    else:
        recs = recs.copy()


        def _to_int(x):
            try:
                return int(x)
            except:
                return x


    recs['action_priority_num'] = recs['action_priority'].apply(_to_int)

    for (issue_cat, sev_level), g in recs.groupby(['issue_category', 'severity_level'], dropna=False):
        rep = g.sort_values(['action_priority_num'], ascending=[True]).iloc[0]

        sev_text = str(sev_level) if pd.notna(sev_level) else "—"
        issue_text = str(issue_cat) if pd.notna(issue_cat) else "Issue"
        st.markdown(f"### General topic: {issue_text} - {sev_text} Priority")


    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Patients", int(cluster_size))
    with m2:
        st.metric("Share of total", f"{share_pct:.1f}%")
    with m3:
        neg_pct = (cluster_detail_data['sentiment'].eq('negative').mean() * 100) if cluster_size else 0.0
        st.metric("Negative %", f"{neg_pct:.1f}%")
    with m4:
        if 'rating_normalized' in cluster_detail_data.columns and cluster_detail_data['rating_normalized'].notna().any():
            st.metric("Avg Rating", f"{(cluster_detail_data['rating_normalized'] * 1.2).mean():.2f}/5.0")
        else:
            st.metric("Avg Rating", "N/A")
    with m5:
        if insight_row is not None and 'impact_score' in insight_row:
            try:
                st.metric("Impact Score", f"{float(insight_row['impact_score']):.1f}")
            except Exception:
                st.metric("Impact Score", "—")
        else:
            st.metric("Impact Score", "—")

    st.markdown("**Cluster Summary:**")
    if insight_row is not None and 'summary' in insight_row and pd.notna(insight_row['summary']):
        st.info(str(insight_row['summary']))
    else:
        st.info("No summary available for this cluster.")

    recs = recommendations_df.copy()
    if 'cluster_id' in recs.columns:
        recs = recs[recs['cluster_id'] == selected_cluster_detail]
    else:
        recs = pd.DataFrame()

    if recs.empty:
        st.info("No recommendations available for this cluster.")
    else:
        # Prepare
        recs = recs.copy()

        def _to_int(x):
            try:
                return int(x)
            except:
                return x

        recs['action_priority_num'] = recs['action_priority'].apply(_to_int)

        for (issue_cat, sev_level), g in recs.groupby(['issue_category', 'severity_level'], dropna=False):

            st.markdown("**✅ Recommended Action**")
            g_sorted = g.sort_values('action_priority_num')
            lines = []
            for _, row in g_sorted.iterrows():
                pr = row.get('action_priority')
                pr_label = f"priority{int(pr)}" if str(pr).isdigit() else f"priority{pr}"
                action_text = str(row.get('recommended_action', '')).strip() or "—"
                lines.append(f"- **{pr_label}:** {action_text}")
            st.markdown("\n".join(lines))

            st.markdown("---")
    st.caption("Sentiment mix (%)")
    sent_counts = cluster_detail_data['sentiment'].value_counts()
    sent_perc = (sent_counts / sent_counts.sum() * 100).reindex(['negative', 'neutral', 'positive']).fillna(0).reset_index()
    sent_perc.columns = ['sentiment', 'percentage']

    fig_cluster_sent = px.bar(
        sent_perc,
        x='sentiment',
        y='percentage',
        title=None,
        labels={'sentiment': 'Sentiment', 'percentage': 'Percentage (%)'},
        color='sentiment',
        color_discrete_map={'negative': '#d62728', 'neutral': '#ff7f0e', 'positive': '#2ca02c'},
        text=sent_perc['percentage'].round(1).astype(str) + '%'
    )
    fig_cluster_sent.update_traces(textposition='outside')
    fig_cluster_sent.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(range=[0, 100], ticksuffix='%')
    )
    st.plotly_chart(fig_cluster_sent, use_container_width=True)

    st.markdown("**Sample Reviews:**")

    samples = cluster_detail_data.sample(min(4, len(cluster_detail_data)), random_state=42) if len(cluster_detail_data) else pd.DataFrame()

    if samples.empty:
        st.info("No reviews available for this cluster.")
    else:
        sentiment_emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}

        cols = st.columns(2)
        for i, (_, row) in enumerate(samples.iterrows()):
            with cols[i % 2]:
                s_val = (row['sentiment'] or '').lower() if pd.notna(row['sentiment']) else 'unknown'
                emoji = sentiment_emoji.get(s_val, '❓')
                s_label = s_val.title() if isinstance(s_val, str) else 'Unknown'
                lang = str(row.get('target_language', '')).upper() if pd.notna(row.get('target_language', None)) else ''
                text_display = row['text_final'] if pd.notna(row.get('text_final')) else row.get('text_cleaned', 'No text available')
                short_text = str(text_display)[:150]

                st.markdown(
                    f"""
                    <div style='
                        background-color: rgba(255, 255, 255, 0.05);
                        border: 1px solid rgba(255, 255, 255, 0.2);
                        border-radius: 8px;
                        padding: 0.6rem 0.8rem;
                        margin-bottom: 0.8rem;
                        color: white;
                        backdrop-filter: blur(5px);
                        min-height: 130px;
                        font-size: 0.9rem;
                        line-height: 1.3;
                        display: flex;
                        flex-direction: column;
                        justify-content: space-between;
                    '>
                        <div>
                            <strong style='font-size:1rem;'>{emoji} {s_label}</strong>
                            <span style='color:#ccc; font-size:0.8rem;'> ({lang})</span><br>
                            <span style='color:white;'>{short_text}...</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                with st.expander("🔍 See full review"):
                    st.markdown(
                        f"<p style='color:white; font-size:0.9rem;'>{str(text_display)}</p>",
                        unsafe_allow_html=True
                    )





# FOOTER
st.markdown("---")
st.markdown(f"""
<div style='text-align:center; color:#666; padding:2rem;'>
    <p><strong>Patient Feedback Analysis Dashboard</strong></p>
    <p>Powered by NLP, Machine Learning & Multilingual Analysis</p>
    <p>Data processed: {datetime.now().strftime('%Y-%m-%d')}</p>
</div>
""", unsafe_allow_html=True)
