"""
CORD-19 Data Explorer - Streamlit Application
Interactive dashboard for exploring COVID-19 research papers
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

# Page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="🦠",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA WITH CACHING
# ============================================================================

@st.cache_data
def load_data():
    """Load and preprocess the CORD-19 metadata"""

    def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'title' in df.columns:
            df = df[df['title'].notna()]

        if 'publish_time' in df.columns:
            df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
            df = df[df['publish_time'].notna()]
            df['year'] = df['publish_time'].dt.year
        else:
            raise ValueError("Expected 'publish_time' column to exist in the metadata.")

        if 'abstract' in df.columns:
            df['abstract'] = df['abstract'].fillna('')
            df['abstract_word_count'] = df['abstract'].apply(
                lambda x: len(str(x).split()) if pd.notna(x) else 0
            )
            df['has_abstract'] = df['abstract_word_count'] > 0
        else:
            df['abstract'] = ''
            df['abstract_word_count'] = 0
            df['has_abstract'] = False

        df = df[(df['year'] >= 2010) & (df['year'] <= 2024)]
        return df

    try:
        # Load the cleaned data if it exists, otherwise load raw data
        try:
            df = pd.read_csv('metadata_cleaned.csv', low_memory=False)
        except FileNotFoundError:
            df = pd.read_csv('metadata.csv', low_memory=False)

        return preprocess_dataframe(df)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Load data
with st.spinner('Loading data... Please wait.'):
    df = load_data()

# ============================================================================
# HEADER SECTION
# ============================================================================

st.markdown('<h1 class="main-header">🦠 CORD-19 Data Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Interactive exploration of COVID-19 research papers</p>', unsafe_allow_html=True)

# Display key metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Papers", f"{len(df):,}")
with col2:
    st.metric("Year Range", f"{int(df['year'].min())}-{int(df['year'].max())}")
with col3:
    st.metric("With Abstracts", f"{df['has_abstract'].sum():,}")
with col4:
    st.metric("Avg Abstract Length", f"{df['abstract_word_count'].mean():.0f} words")

st.markdown("---")

# ============================================================================
# SIDEBAR FILTERS
# ============================================================================

st.sidebar.header("🔍 Filters")

# Year range slider
year_min, year_max = int(df['year'].min()), int(df['year'].max())
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=year_min,
    max_value=year_max,
    value=(2019, year_max),
    step=1
)

# Abstract filter
abstract_filter = st.sidebar.radio(
    "Abstract Availability",
    ["All Papers", "With Abstract Only", "Without Abstract Only"]
)

# Source filter (if available)
if 'source_x' in df.columns:
    sources = ['All Sources'] + sorted(df['source_x'].dropna().unique().tolist())
    selected_source = st.sidebar.selectbox("Select Source", sources)
else:
    selected_source = 'All Sources'

# Apply filters
filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

if abstract_filter == "With Abstract Only":
    filtered_df = filtered_df[filtered_df['has_abstract'] == True]
elif abstract_filter == "Without Abstract Only":
    filtered_df = filtered_df[filtered_df['has_abstract'] == False]

if selected_source != 'All Sources' and 'source_x' in df.columns:
    filtered_df = filtered_df[filtered_df['source_x'] == selected_source]

st.sidebar.markdown(f"**Filtered Results:** {len(filtered_df):,} papers")

# ============================================================================
# MAIN CONTENT TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Trends", "📚 Content Analysis", "🔎 Data Explorer"])

# TAB 1: OVERVIEW
with tab1:
    st.header("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Publications by Year")
        year_counts = filtered_df['year'].value_counts().sort_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=year_counts.index,
            y=year_counts.values,
            mode='lines+markers',
            marker=dict(size=10),
            line=dict(width=3)
        ))
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Number of Publications",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Abstract Availability")
        abstract_counts = filtered_df['has_abstract'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=['With Abstract', 'Without Abstract'],
            values=abstract_counts.values,
            hole=0.4
        )])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top sources/journals
    st.subheader("Top Publishing Sources")
    if 'source_x' in filtered_df.columns:
        top_sources = filtered_df['source_x'].value_counts().head(15)
        
        fig = px.bar(
            x=top_sources.values,
            y=top_sources.index,
            orientation='h',
            labels={'x': 'Number of Papers', 'y': 'Source'}
        )
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: TRENDS
with tab2:
    st.header("Temporal Trends")
    
    # Monthly publications (if we have detailed dates)
    st.subheader("Publication Trends Over Time")
    
    # Create monthly aggregation
    filtered_df['year_month'] = filtered_df['publish_time'].dt.to_period('M')
    monthly_counts = filtered_df.groupby('year_month').size().reset_index(name='count')
    monthly_counts['year_month'] = monthly_counts['year_month'].astype(str)
    
    fig = px.line(
        monthly_counts,
        x='year_month',
        y='count',
        title='Publications per Month'
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Publications",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Yearly comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Year-over-Year Growth")
        year_counts = filtered_df['year'].value_counts().sort_index()
        growth_rate = year_counts.pct_change() * 100
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=growth_rate.index,
            y=growth_rate.values,
            marker_color=['red' if x < 0 else 'green' for x in growth_rate.values]
        ))
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Growth Rate (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Cumulative Publications")
        cumulative = year_counts.cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cumulative.index,
            y=cumulative.values,
            fill='tozeroy',
            mode='lines'
        ))
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Cumulative Papers",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# TAB 3: CONTENT ANALYSIS
with tab3:
    st.header("Content Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Word Cloud of Titles")
        
        # Generate word cloud
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be'}
        
        all_titles = ' '.join(filtered_df['title'].dropna().astype(str))
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=stopwords,
            max_words=100,
            colormap='viridis'
        ).generate(all_titles)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Most Frequent Title Words")
        
        # Extract and count words
        words = re.findall(r'\b[a-z]+\b', all_titles.lower())
        filtered_words = [w for w in words if w not in stopwords and len(w) > 3]
        word_freq = Counter(filtered_words).most_common(20)
        
        word_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
        
        fig = px.bar(
            word_df,
            x='Frequency',
            y='Word',
            orientation='h'
        )
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Abstract length distribution
    st.subheader("Abstract Length Distribution")
    abstracts_with_text = filtered_df[filtered_df['abstract_word_count'] > 0]['abstract_word_count']
    
    fig = px.histogram(
        abstracts_with_text,
        nbins=50,
        title="Distribution of Abstract Word Counts"
    )
    fig.update_layout(
        xaxis_title="Word Count",
        yaxis_title="Frequency",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{abstracts_with_text.mean():.0f}")
    with col2:
        st.metric("Median", f"{abstracts_with_text.median():.0f}")
    with col3:
        st.metric("Min", f"{abstracts_with_text.min():.0f}")
    with col4:
        st.metric("Max", f"{abstracts_with_text.max():.0f}")

# TAB 4: DATA EXPLORER
with tab4:
    st.header("Data Explorer")
    
    st.subheader("Search Papers")
    search_term = st.text_input("Search in titles:", "")
    
    if search_term:
        search_results = filtered_df[
            filtered_df['title'].str.contains(search_term, case=False, na=False)
        ]
        st.write(f"Found {len(search_results)} papers matching '{search_term}'")
    else:
        search_results = filtered_df
    
    # Display options
    col1, col2 = st.columns(2)
    with col1:
        num_rows = st.selectbox("Number of rows to display", [10, 25, 50, 100], index=0)
    with col2:
        columns_to_show = st.multiselect(
            "Select columns to display",
            options=['title', 'authors', 'journal', 'publish_time', 'abstract', 'url'],
            default=['title', 'publish_time']
        )
    
    if columns_to_show:
        # Filter available columns
        available_cols = [col for col in columns_to_show if col in search_results.columns]
        st.dataframe(
            search_results[available_cols].head(num_rows),
            use_container_width=True,
            height=400
        )
    
    # Download button
    st.subheader("Download Filtered Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name='cord19_filtered_data.csv',
        mime='text/csv'
    )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>CORD-19 Dataset Explorer | Data Science Assignment</p>
        <p>Built with Streamlit 🎈</p>
    </div>
""", unsafe_allow_html=True)