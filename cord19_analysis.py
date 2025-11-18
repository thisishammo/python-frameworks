"""
CORD-19 Data Analysis Script
This script performs exploratory data analysis on the CORD-19 metadata_small.csv file
Author: Hammond
Date: November 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 60)
print("CORD-19 DATA ANALYSIS PROJECT")
print("=" * 60)

# ============================================================================
# PART 1: DATA LOADING AND BASIC EXPLORATION
# ============================================================================

print("\n--- PART 1: DATA LOADING AND BASIC EXPLORATION ---\n")

# Step 1: Load the data
print("Loading metadata_small.csv file...")
df = pd.read_csv('metadata_small.csv', encoding='utf-16', low_memory=False)
print("✓ Data loaded successfully!")

# Step 2: Examine first few rows
print("\n1. First 5 rows of the dataset:")
print(df.head())

# Step 3: Check DataFrame dimensions
print(f"\n2. Dataset dimensions: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Step 4: Identify data types
print("\n3. Data types of each column:")
print(df.dtypes)

# Step 5: Check for missing values
print("\n4. Missing values in each column:")
missing_counts = df.isnull().sum()
missing_percentage = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_counts,
    'Percentage': missing_percentage
})
print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))

# Step 6: Basic statistics
print("\n5. Basic statistics for numerical columns:")
print(df.describe())

# Additional useful info
print("\n6. Column names:")
print(df.columns.tolist())

# ============================================================================
# PART 2: DATA CLEANING AND PREPARATION
# ============================================================================

print("\n\n--- PART 2: DATA CLEANING AND PREPARATION ---\n")

# Step 1: Create a copy for cleaning
df_clean = df.copy()

# Step 2: Handle missing values in important columns
print("1. Handling missing values...")

# Keep only rows with titles (essential for analysis)
initial_rows = len(df_clean)
df_clean = df_clean[df_clean['title'].notna()]
print(f"   - Removed {initial_rows - len(df_clean):,} rows without titles")

# Fill missing abstracts with empty string for word count calculation
df_clean['abstract'] = df_clean['abstract'].fillna('')

# Keep rows with publication dates for time-based analysis
df_clean = df_clean[df_clean['publish_time'].notna()]
print(f"   - Dataset now has {len(df_clean):,} rows")

# Step 3: Convert date column to datetime
print("\n2. Converting dates to datetime format...")
df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')

# Step 4: Extract year from publication date
print("3. Extracting year from publication dates...")
df_clean['year'] = df_clean['publish_time'].dt.year

# Step 5: Create new columns
print("4. Creating additional columns...")

df_clean['abstract_word_count'] = df_clean['abstract'].apply(
    lambda x: len(str(x).split()) if pd.notna(x) else 0
)

df_clean['title_word_count'] = df_clean['title'].apply(
    lambda x: len(str(x).split()) if pd.notna(x) else 0
)

df_clean['has_abstract'] = df_clean['abstract_word_count'] > 0

print(f"   - Created 'year' column")
print(f"   - Created 'abstract_word_count' column")
print(f"   - Created 'title_word_count' column")
print(f"   - Created 'has_abstract' column")

# Display cleaned data info
print("\n5. Cleaned dataset summary:")
print(f"   - Total papers: {len(df_clean):,}")
print(f"   - Year range: {df_clean['year'].min():.0f} to {df_clean['year'].max():.0f}")
print(f"   - Papers with abstracts: {df_clean['has_abstract'].sum():,} ({df_clean['has_abstract'].mean()*100:.1f}%)")
print(f"   - Average abstract length: {df_clean['abstract_word_count'].mean():.1f} words")


# PART 3: DATA ANALYSIS AND VISUALIZATION
print("\n\n--- PART 3: DATA ANALYSIS AND VISUALIZATION ---\n")

# Analysis 1: Count papers by publication year
print("1. Publications by year:")
year_counts = df_clean['year'].value_counts().sort_index()
print(year_counts)

# Filter to reasonable year range (2010-2024)
df_clean = df_clean[(df_clean['year'] >= 2010) & (df_clean['year'] <= 2024)]

# Analysis 2: Top journals
print("\n2. Top 10 journals publishing COVID-19 research:")
if 'journal' in df_clean.columns:
    top_journals = df_clean['journal'].value_counts().head(10)
    for idx, (journal, count) in enumerate(top_journals.items(), 1):
        print(f"   {idx}. {journal}: {count:,} papers")

# Analysis 3: Top sources
print("\n3. Top 10 sources:")
if 'source_x' in df_clean.columns:
    top_sources = df_clean['source_x'].value_counts().head(10)
    for idx, (source, count) in enumerate(top_sources.items(), 1):
        print(f"   {idx}. {source}: {count:,} papers")

# Analysis 4: Most frequent words in titles
print("\n4. Most frequent words in titles (excluding common words):")

# Combine all titles
all_titles = ' '.join(df_clean['title'].dropna().astype(str))

# Remove common words (stopwords)
stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
             'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be',
             'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
             'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}

# Extract words (only alphabetic, lowercase)
words = re.findall(r'\b[a-z]+\b', all_titles.lower())
filtered_words = [w for w in words if w not in stopwords and len(w) > 3]

# Count frequency
word_freq = Counter(filtered_words)
top_words = word_freq.most_common(20)

for idx, (word, count) in enumerate(top_words, 1):
    print(f"   {idx}. {word}: {count:,} times")

# ============================================================================
# VISUALIZATION SECTION
# ============================================================================

print("\n5. Creating visualizations...")

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# Visualization 1: Publications over time
ax1 = plt.subplot(2, 3, 1)
year_counts = df_clean['year'].value_counts().sort_index()
ax1.plot(year_counts.index, year_counts.values, marker='o', linewidth=2, markersize=6)
ax1.set_xlabel('Year', fontsize=10)
ax1.set_ylabel('Number of Publications', fontsize=10)
ax1.set_title('COVID-19 Research Publications Over Time', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Visualization 2: Top 10 journals
if 'journal' in df_clean.columns:
    ax2 = plt.subplot(2, 3, 2)
    top_journals = df_clean['journal'].value_counts().head(10)
    ax2.barh(range(len(top_journals)), top_journals.values)
    ax2.set_yticks(range(len(top_journals)))
    ax2.set_yticklabels([j[:30] + '...' if len(j) > 30 else j for j in top_journals.index], fontsize=8)
    ax2.set_xlabel('Number of Papers', fontsize=10)
    ax2.set_title('Top 10 Journals', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()

# Visualization 3: Distribution by source
if 'source_x' in df_clean.columns:
    ax3 = plt.subplot(2, 3, 3)
    top_sources = df_clean['source_x'].value_counts().head(10)
    ax3.bar(range(len(top_sources)), top_sources.values)
    ax3.set_xticks(range(len(top_sources)))
    ax3.set_xticklabels([s[:15] + '...' if len(s) > 15 else s for s in top_sources.index], 
                         rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('Number of Papers', fontsize=10)
    ax3.set_title('Top 10 Sources', fontsize=12, fontweight='bold')

# Visualization 4: Word cloud of titles
ax4 = plt.subplot(2, 3, 4)
wordcloud = WordCloud(width=800, height=400, background_color='white',
                      stopwords=stopwords, max_words=100).generate(all_titles)
ax4.imshow(wordcloud, interpolation='bilinear')
ax4.axis('off')
ax4.set_title('Word Cloud of Paper Titles', fontsize=12, fontweight='bold')

# Visualization 5: Abstract length distribution
ax5 = plt.subplot(2, 3, 5)
# Filter out papers without abstracts
abstracts_with_text = df_clean[df_clean['abstract_word_count'] > 0]['abstract_word_count']
ax5.hist(abstracts_with_text, bins=50, edgecolor='black', alpha=0.7)
ax5.set_xlabel('Abstract Word Count', fontsize=10)
ax5.set_ylabel('Frequency', fontsize=10)
ax5.set_title('Distribution of Abstract Lengths', fontsize=12, fontweight='bold')
ax5.set_xlim(0, 500)

# Visualization 6: Papers with/without abstracts
ax6 = plt.subplot(2, 3, 6)
abstract_counts = df_clean['has_abstract'].value_counts()
ax6.pie(abstract_counts.values, labels=['With Abstract', 'Without Abstract'],
        autopct='%1.1f%%', startangle=90, colors=['#66c2a5', '#fc8d62'])
ax6.set_title('Papers with vs without Abstracts', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('cord19_analysis_visualizations.png', dpi=300, bbox_inches='tight')
print("   ✓ Visualizations saved as 'cord19_analysis_visualizations.png'")

plt.show()

# ============================================================================
# SAVE CLEANED DATA
# ============================================================================

print("\n6. Saving cleaned dataset...")
df_clean.to_csv('metadata_cleaned.csv', index=False)
print("   ✓ Cleaned data saved as 'metadata_cleaned.csv'")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)
print(f"\nKey Findings:")
print(f"- Analyzed {len(df_clean):,} COVID-19 research papers")
print(f"- Publications span from {df_clean['year'].min():.0f} to {df_clean['year'].max():.0f}")
print(f"- Peak publication year: {year_counts.idxmax()}")
print(f"- {df_clean['has_abstract'].sum():,} papers have abstracts")
print(f"- Most common word in titles: '{top_words[0][0]}'")
print("\nNext steps: Run the Streamlit app to interactively explore the data!")