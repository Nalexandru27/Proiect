import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Financial Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📈 Financial Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Analyzing investment opportunities and expansion possibilities across sectors**")

# Sidebar for file upload and filters
st.sidebar.header("📁 Data Upload & Filters")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=['xlsx', 'xls'])

# Create sample data if no file is uploaded
if uploaded_file is None:
    st.sidebar.info("Using sample data. Upload your Excel file to analyze your dataset.")
    
    # Sample data based on your description
    sample_data = {
        'Ticker': ['CWT', 'BKH', 'NUE', 'UBSI', 'AAPL', 'MSFT', 'JNJ', 'PG', 'KO', 'WMT',
                  'HD', 'BAC', 'JPM', 'XOM', 'CVX', 'PFE', 'MRK', 'T', 'VZ', 'DIS'],
        'Sector': ['Utilities', 'Utilities', 'Basic Materials', 'Financial Services', 'Technology',
                  'Technology', 'Healthcare', 'Consumer Goods', 'Consumer Goods', 'Consumer Services',
                  'Consumer Services', 'Financial Services', 'Financial Services', 'Energy', 'Energy',
                  'Healthcare', 'Healthcare', 'Telecommunications', 'Telecommunications', 'Consumer Services'],
        'Price': [49.55, 60.83, 116.57, 34.20, 175.43, 280.76, 158.91, 145.12, 60.13, 153.78,
                 320.86, 32.15, 142.87, 110.42, 150.27, 38.94, 102.83, 18.45, 40.12, 112.34],
        'Market Cap': [2.86, 4.14, 28.77, 4.61, 2800.5, 2100.3, 420.8, 350.2, 260.4, 410.6,
                      520.3, 265.4, 445.2, 445.8, 290.1, 220.5, 260.8, 130.2, 180.4, 200.5],
        'Current Ratio': [0.69, 0.70, 3.57, 9.00, 1.15, 2.45, 1.32, 0.95, 1.18, 0.82,
                         1.08, 1.05, 1.12, 1.25, 1.18, 1.45, 1.28, 0.75, 0.68, 1.02],
        'Dividend Yield': [2.06, 4.06, 3.30, 4.23, 0.52, 0.68, 2.58, 2.41, 3.15, 1.65,
                          2.08, 2.45, 2.68, 5.85, 3.21, 4.12, 2.95, 7.25, 6.58, 1.95],
        'P/E Ratio': [9.73, 5.72, 1.90, 4.44, 28.5, 32.1, 15.8, 24.2, 25.6, 26.8,
                    22.4, 12.5, 11.2, 15.4, 13.8, 18.9, 14.2, 8.5, 9.2, 25.4],
        'ROE': [3.64, 8.15, 21.61, 7.68, 28.5, 35.2, 24.8, 22.1, 41.2, 19.8,
               25.6, 11.2, 15.4, 18.9, 12.5, 8.9, 12.4, 15.2, 18.6, 22.3],
        'Debt/Capital': [46.37, 57.79, 24.12, 29.38, 15.2, 18.5, 35.4, 28.9, 42.1, 38.7,
                           25.8, 45.2, 38.9, 25.4, 28.7, 42.8, 35.6, 52.1, 48.9, 32.4],
        'Earnings Stability': [True, False, True, True, True, True, True, True, True, False,
                              True, True, False, True, True, True, False, True, True, True]
    }
    
    df = pd.DataFrame(sample_data)
    # Add more rows to reach 70 observations
    additional_rows = []
    sectors = df['Sector'].unique()
    for i in range(50):
        row = {
            'Ticker': f'COMP{i+21}',
            'Sector': np.random.choice(sectors),
            'Price': np.random.uniform(20, 300),
            'Market Cap': np.random.uniform(1, 500),
            'Current Ratio': np.random.uniform(0.5, 10),
            'Dividend Yield': np.random.uniform(0, 8),
            'P/E Ratio': np.random.uniform(5, 40),
            'ROE': np.random.uniform(5, 45),
            'Debt/Capital': np.random.uniform(10, 60),
            'Earnings Stability': np.random.choice([True, False])
        }
        additional_rows.append(row)
    
    df = pd.concat([df, pd.DataFrame(additional_rows)], ignore_index=True)
    
else:
    # Load the uploaded file
    try:
        df = pd.read_excel(uploaded_file)
        st.sidebar.success(f"✅ File uploaded successfully! ({len(df)} rows)")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")
        st.stop()

# Data preprocessing and cleaning
st.header("🔧 Data Processing & Quality Check")

# Handle missing values (Requirement 3)
st.subheader("Missing Values Analysis")
missing_data = df.isnull().sum()
if missing_data.sum() > 0:
    st.warning(f"Found {missing_data.sum()} missing values")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Missing values by column:")
        st.write(missing_data[missing_data > 0])
    with col2:
        # Fill missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        st.success("✅ Missing values filled with median")
else:
    st.success("✅ No missing values found")

# Detect extreme values (Requirement 3)
st.subheader("Extreme Values Detection")
numeric_cols = df.select_dtypes(include=[np.number]).columns
outliers_detected = {}

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    if len(outliers) > 0:
        outliers_detected[col] = len(outliers)

if outliers_detected:
    st.warning(f"Extreme values detected in {len(outliers_detected)} columns")
    st.write(outliers_detected)
else:
    st.success("✅ No extreme outliers detected")

# Encoding methods (Requirement 4)
st.header("🔄 Data Encoding")
categorical_columns = df.select_dtypes(include=['object', 'bool']).columns

if len(categorical_columns) > 0:
    st.subheader("Categorical Variables Encoding")
    
    # Label encoding for categorical variables
    le = LabelEncoder()
    df_encoded = df.copy()
    
    for col in categorical_columns:
        if col in df_encoded.columns:
            df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Original categorical data:")
        st.write(df[categorical_columns].head())
    with col2:
        st.write("Encoded data:")
        encoded_cols = [col for col in df_encoded.columns if col.endswith('_encoded')]
        st.write(df_encoded[encoded_cols].head())
    
    st.success("✅ Categorical variables encoded successfully")

# Scaling methods (Requirement 5)
st.header("📏 Data Scaling")
st.subheader("Feature Scaling for Analysis")

numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_features] = scaler.fit_transform(df[numeric_features])

col1, col2 = st.columns(2)
with col1:
    st.write("Original data statistics:")
    st.write(df[numeric_features].describe().round(2))
with col2:
    st.write("Scaled data statistics:")
    st.write(df_scaled[numeric_features].describe().round(2))

st.success("✅ Features scaled using StandardScaler")

# Statistical processing and grouping (Requirement 6)
st.header("📊 Statistical Analysis & Grouping")

# Group by sector analysis
if 'Sector' in df.columns:
    st.subheader("Sector-wise Analysis")
    
    sector_stats = df.groupby('Sector').agg({
        'Price': ['mean', 'median', 'std'],
        'Market Cap': ['mean', 'sum'],
        'Dividend Yield': 'mean',
        'P/E Ratio': 'mean',
        'ROE': 'mean'
    }).round(2)
    
    sector_stats.columns = ['_'.join(col).strip() for col in sector_stats.columns]
    st.write(sector_stats)
    
    # Sector performance metrics
    st.subheader("Key Sector Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_roe_sector = df.groupby('Sector')['ROE'].mean().idxmax()
        st.metric("Best ROE Sector", best_roe_sector, 
                 f"{df.groupby('Sector')['ROE'].mean().max():.2f}%")
    
    with col2:
        highest_dividend_sector = df.groupby('Sector')['Dividend Yield'].mean().idxmax()
        st.metric("Highest Dividend Sector", highest_dividend_sector,
                 f"{df.groupby('Sector')['Dividend Yield'].mean().max():.2f}%")
    
    with col3:
        largest_market_cap_sector = df.groupby('Sector')['Market Cap'].sum().idxmax()
        st.metric("Largest Market Cap Sector", largest_market_cap_sector,
                 f"${df.groupby('Sector')['Market Cap'].sum().max():.1f}B")

# Graphical representation with matplotlib (Requirement 8)
st.header("📈 Data Visualization")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Sector distribution
if 'Sector' in df.columns:
    sector_counts = df['Sector'].value_counts()
    axes[0, 0].pie(sector_counts.values, labels=sector_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Distribution by Sector')

# 2. Price vs Market Cap scatter plot
axes[0, 1].scatter(df['Price'], df['Market Cap'], alpha=0.6, c=df.index)
axes[0, 1].set_xlabel('Stock Price ($)')
axes[0, 1].set_ylabel('Market Cap (B$)')
axes[0, 1].set_title('Price vs Market Capitalization')

# 3. Dividend Yield distribution
axes[1, 0].hist(df['Dividend Yield'], bins=15, alpha=0.7, color='green')
axes[1, 0].set_xlabel('Dividend Yield (%)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Dividend Yield Distribution')

# 4. ROE vs PE Ratio
axes[1, 1].scatter(df['P/E Ratio'], df['ROE'], alpha=0.6, c='red')
axes[1, 1].set_xlabel('P/E Ratio')
axes[1, 1].set_ylabel('ROE (%)')
axes[1, 1].set_title('P/E Ratio vs Return on Equity')

plt.tight_layout()
st.pyplot(fig)

# Calculează matricea de corelație doar pe coloanele numerice
corr_matrix = df[numeric_features].corr()

# Creează o mască pentru triunghiul superior
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Creează figura
fig, ax = plt.subplots(figsize=(14, 10))

# Desenează heatmap-ul
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",                     # rotunjire la 2 zecimale
    cmap='coolwarm',
    center=0,
    square=True,                  # păstrează pătratele egale
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},     # reduce bara de culoare
    annot_kws={"size": 8}         # font mai mic pentru valori
)

# Ajustează etichetele
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Afișează figura în Streamlit
st.subheader("Correlation Analysis")
st.pyplot(fig)

# Clustering with scikit-learn (Requirement 9)
st.header("🎯 Investment Clustering Analysis")

st.subheader("Company Clustering Based on Financial Characteristics")

# Prepare data for clustering
clustering_features = ['Price', 'Market Cap', 'Dividend Yield', 'P/E Ratio', 'ROE']
available_features = [col for col in clustering_features if col in df.columns]

if len(available_features) >= 3:
    X_cluster = df[available_features].fillna(df[available_features].median())
    X_cluster_scaled = StandardScaler().fit_transform(X_cluster)
    
    # Perform K-means clustering
    n_clusters = st.slider("Number of clusters", 2, 6, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_cluster_scaled)
    
    df['Cluster'] = clusters
    
    # Display cluster characteristics
    st.write("Cluster Characteristics:")
    cluster_summary = df.groupby('Cluster')[available_features].mean().round(2)
    st.write(cluster_summary)
    
    # Visualize clusters
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['P/E Ratio'], df['ROE'], c=df['Cluster'], cmap='viridis', alpha=0.6)
    ax.set_xlabel('P/E Ratio')
    ax.set_ylabel('ROE (%)')
    ax.set_title('Investment Clusters (P/E vs ROE)')
    plt.colorbar(scatter)
    st.pyplot(fig)
    
    # Investment recommendations
    st.subheader("📋 Investment Cluster Insights")
    for cluster_id in range(n_clusters):
        cluster_data = df[df['Cluster'] == cluster_id]
        avg_roe = cluster_data['ROE'].mean()
        avg_pe = cluster_data['P/E Ratio'].mean()
        avg_dividend = cluster_data['Dividend Yield'].mean()
        
        if avg_roe > df['ROE'].median() and avg_pe < df['P/E Ratio'].median():
            cluster_type = "🟢 Value Growth"
        elif avg_dividend > df['Dividend Yield'].median():
            cluster_type = "🔵 Dividend Focus"
        elif avg_pe > df['P/E Ratio'].median():
            cluster_type = "🟡 Growth Premium"
        else:
            cluster_type = "🟠 Balanced"
        
        st.write(f"**Cluster {cluster_id}** ({len(cluster_data)} companies): {cluster_type}")
        st.write(f"- Average ROE: {avg_roe:.2f}%")
        st.write(f"- Average P/E: {avg_pe:.2f}")
        st.write(f"- Average Dividend Yield: {avg_dividend:.2f}%")

# Logistic Regression (Requirement 9)
st.header("🎲 Predictive Analysis")

st.subheader("Dividend Stability Prediction")

if 'Earnings_Stability' in df.columns:
    # Prepare features for logistic regression
    feature_cols = ['P/E Ratio', 'ROE', 'Dividend Yield', 'Debt/Capital']
    available_feature_cols = [col for col in feature_cols if col in df.columns]
    
    if len(available_feature_cols) >= 2:
        X = df[available_feature_cols].fillna(df[available_feature_cols].median())
        y = df['Earnings Stability'].astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        scaler_lr = StandardScaler()
        X_train_scaled = scaler_lr.fit_transform(X_train)
        X_test_scaled = scaler_lr.transform(X_test)
        
        # Train logistic regression
        lr_model = LogisticRegression(random_state=42)
        lr_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = lr_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Accuracy", f"{accuracy:.2%}")
            
        with col2:
            stable_companies = df[df['Earnings Stability'] == True]
            st.metric("Stable Companies", f"{len(stable_companies)}/{len(df)}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': available_feature_cols,
            'Importance': abs(lr_model.coef_[0])
        }).sort_values('Importance', ascending=False)
        
        st.write("📊 Feature Importance for Earnings Stability:")
        st.write(feature_importance)

# Multiple Regression with statsmodels (Requirement 10)
st.header("🔍 Multiple Regression Analysis")

st.subheader("Stock Price Prediction Model")

# Prepare data for multiple regression
target_col = 'Price'
predictor_cols = ['Market Cap', 'P/E Ratio', 'ROE', 'Dividend Yield']
available_predictors = [col for col in predictor_cols if col in df.columns]

if len(available_predictors) >= 2:
    # Prepare regression data
    regression_data = df[available_predictors + [target_col]].dropna()
    
    X_reg = regression_data[available_predictors]
    y_reg = regression_data[target_col]
    
    # Add constant for intercept
    X_reg = sm.add_constant(X_reg)
    
    # Fit multiple regression model
    model = sm.OLS(y_reg, X_reg).fit()
    
    # Display results
    st.subheader("📈 Regression Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R-squared", f"{model.rsquared:.3f}")
        st.metric("Adj. R-squared", f"{model.rsquared_adj:.3f}")
    
    with col2:
        st.metric("F-statistic", f"{model.fvalue:.2f}")
        st.metric("P-value (F-stat)", f"{model.f_pvalue:.4f}")
    
    # Regression coefficients
    st.subheader("📊 Coefficient Analysis")
    coef_df = pd.DataFrame({
        'Variable': model.params.index,
        'Coefficient': model.params.values,
        'P-value': model.pvalues.values,
        'Significant': model.pvalues.values < 0.05
    }).round(4)
    
    st.write(coef_df)
    
    # Residual plot
    st.subheader("🎯 Model Diagnostics")
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Residuals vs Fitted
    ax1.scatter(fitted_values, residuals, alpha=0.6)
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted Values')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    st.pyplot(fig)

# Investment Recommendations Dashboard
st.header("💼 Investment Expansion Recommendations")

st.subheader("📊 Dashboard Summary")

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_roe = df['ROE'].mean()
    st.metric("Average ROE", f"{avg_roe:.2f}%", 
              delta=f"{(avg_roe - df['ROE'].median()):.2f}%")

with col2:
    avg_dividend = df['Dividend Yield'].mean()
    st.metric("Average Dividend Yield", f"{avg_dividend:.2f}%",
              delta=f"{(avg_dividend - 3.0):.2f}%" if avg_dividend > 3.0 else None)

with col3:
    total_market_cap = df['Market Cap'].sum()
    st.metric("Total Market Cap", f"${total_market_cap:.1f}B")

with col4:
    num_sectors = df['Sector'].nunique() if 'Sector' in df.columns else 0
    st.metric("Sectors Covered", num_sectors)

# Top recommendations
st.subheader("🎯 Top Investment Opportunities")

if all(col in df.columns for col in ['ROE', 'Dividend Yield', 'P/E Ratio']):
    # Calculate investment score
    df['Investment_Score'] = (
        (df['ROE'] / df['ROE'].max()) * 0.4 +
        (df['Dividend Yield'] / df['Dividend Yield'].max()) * 0.3 +
        ((df['P/E Ratio'].max() - df['P/E Ratio']) / df['P/E Ratio'].max()) * 0.3
    )
    
    top_investments = df.nlargest(5, 'Investment_Score')[
        ['Ticker', 'Sector', 'Price', 'ROE', 'Dividend Yield', 'P/E Ratio', 'Investment_Score']
    ].round(2)
    
    st.write("**Top 5 Investment Opportunities:**")
    st.write(top_investments)

# Final insights
st.subheader("🔍 Key Insights & Expansion Strategy")

insights = []

if 'Sector' in df.columns:
    dominant_sector = df['Sector'].value_counts().index[0]
    insights.append(f"**Sector Dominance**: {dominant_sector} represents the largest portion of analyzed companies")

if 'ROE' in df.columns:
    high_roe_companies = len(df[df['ROE'] > df['ROE'].quantile(0.75)])
    insights.append(f"**High Performance**: {high_roe_companies} companies show exceptional ROE (top quartile)")

if 'Dividend_Yield' in df.columns:
    dividend_paying = len(df[df['Dividend Yield'] > 2.0])
    insights.append(f"**Income Generation**: {dividend_paying} companies offer attractive dividend yields (>2%)")

for insight in insights:
    st.write(insight)

# Footer
st.markdown("---")
st.markdown("**Analysis completed using:** Python, Streamlit, Pandas, Matplotlib, Scikit-learn, Statsmodels")