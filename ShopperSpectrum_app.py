# ==========================================================
# Shopper Spectrum 
# Customer Segmentation & Product Recommendation System
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ----------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# TITLE
# ----------------------------------------------------------

st.title("🛒 Shopper Spectrum")
st.subheader("Customer Segmentation & Product Recommendation System")

# ----------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        "compressed_data.csv.gz",
        compression="gzip",
        encoding="latin1"
    )

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Remove missing Customer IDs
    df = df.dropna(subset=["CustomerID"])

    # Remove cancelled invoices
    df = df[
        ~df["InvoiceNo"].astype(str).str.startswith("C")
    ]

    # Remove invalid Quantity and Price
    df = df[
        (df["Quantity"] > 0) &
        (df["UnitPrice"] > 0)
    ]

    # Convert data types
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        dayfirst=True
    )

    # Feature Engineering
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    return df

df = load_data()

# ----------------------------------------------------------
# LOAD TRAINED MODELS
# ----------------------------------------------------------

kmeans = pickle.load(
    open("kmeans.pkl", "rb")
)

scaler = pickle.load(
    open("scaler.pkl", "rb")
)

similarity_df = pickle.load(
    open("similarity.pkl", "rb")
)

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------

st.sidebar.title("🧭 Navigation")

menu = st.sidebar.radio(

    "Select Module",

    [

        "🏠 Home",

        "📊 Dashboard Overview",

        "📈 RFM Analysis",

        "🎯 Product Recommendation",

        "👤 Customer Segmentation"

    ]

)

st.sidebar.markdown("---")

st.sidebar.info("""

### 🛒 Shopper Spectrum

Customer Segmentation

Product Recommendation

RFM Analysis

Interactive Dashboard

""")

st.sidebar.success("Version 1.0")

# ==========================================================
# HOME PAGE
# ==========================================================

if menu == "🏠 Home":

    st.header("🏠 Welcome to Shopper Spectrum")

    st.write("""
Shopper Spectrum is an E-Commerce Analytics Project.

✔ Customer Segmentation using RFM Analysis

✔ Product Recommendation System

✔ Business Insights Dashboard

✔ Interactive Data Visualization

✔ Machine Learning using KMeans Clustering
""")

    st.markdown("---")

    # ------------------------------------------------------
    # DATASET SUMMARY
    # ------------------------------------------------------

    st.subheader("📊 Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Customers",
            df["CustomerID"].nunique()
        )

    with col2:
        st.metric(
            "Products",
            df["Description"].nunique()
        )

    with col3:
        st.metric(
            "Countries",
            df["Country"].nunique()
        )

    with col4:
        st.metric(
            "Transactions",
            len(df)
        )

    st.markdown("---")

    # ------------------------------------------------------
    # CUSTOMER SEGMENT TABLE
    # ------------------------------------------------------

    st.subheader("📌 Customer Segments at a Glance")

    segment_df = pd.DataFrame({

        "Segment":[
            "🟢 High-Value",
            "🟡 Regular",
            "🔵 Occasional",
            "🔴 At-Risk"
        ],

        "Recency":[
            "Very Recent",
            "Recent",
            "Recent",
            "Long Ago"
        ],

        "Frequency":[
            "Very High",
            "Medium",
            "Low",
            "Very Low"
        ],

        "Monetary":[
            "Very High",
            "Medium",
            "Low",
            "Low"
        ],

        "Description":[
            "Frequent Premium Customers",
            "Steady Buyers",
            "Occasional Buyers",
            "Need Retention"
        ]

    })

    st.table(segment_df)

    st.info("👈 Use the sidebar to navigate through all modules.")

    st.markdown("---")

    # ------------------------------------------------------
    # PROJECT FEATURES
    # ------------------------------------------------------

    st.subheader("🚀 Project Features")

    feature1, feature2 = st.columns(2)

    with feature1:

        st.success("✔ Data Cleaning")

        st.success("✔ Exploratory Data Analysis")

        st.success("✔ RFM Analysis")

        st.success("✔ KMeans Clustering")

    with feature2:

        st.success("✔ Product Recommendation")

        st.success("✔ Customer Segmentation")

        st.success("✔ Business Dashboard")

        st.success("✔ Interactive Visualizations")

    st.markdown("---")

    with st.expander("📖 About This Project"):

        st.write("""

Shopper Spectrum is an end-to-end E-Commerce Analytics project.

This project performs:

• Data Cleaning

• Exploratory Data Analysis (EDA)

• RFM Analysis

• Customer Segmentation

• Product Recommendation

• Business Insights Dashboard

• Interactive Streamlit Application

""")

# ==========================================================
# DASHBOARD OVERVIEW
# ==========================================================

elif menu == "📊 Dashboard Overview":

    st.header("📊 Dashboard Overview")

    # ------------------------------------------------------
    # KPI CARDS
    # ------------------------------------------------------

    total_customers = df["CustomerID"].nunique()
    total_products = df["Description"].nunique()
    total_countries = df["Country"].nunique()
    total_revenue = round(df["TotalAmount"].sum(), 2)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("👥 Customers", total_customers)
    c2.metric("📦 Products", total_products)
    c3.metric("🌍 Countries", total_countries)
    c4.metric("💰 Revenue (£)", f"{total_revenue:,.0f}")

    st.markdown("---")

    # ------------------------------------------------------
    # DATASET PREVIEW
    # ------------------------------------------------------

    st.subheader("📋 Dataset Preview")

    st.dataframe(df.head(10))

    st.write("Rows :", df.shape[0])
    st.write("Columns :", df.shape[1])

    st.markdown("---")

    # ------------------------------------------------------
    # TOP 10 SELLING PRODUCTS
    # ------------------------------------------------------

    st.subheader("🏆 Top 10 Selling Products")

    top_products = (
        df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10,6))

    ax.barh(
        top_products.index,
        top_products.values
    )

    ax.set_xlabel("Quantity Sold")
    ax.set_ylabel("Products")
    ax.set_title("Top Selling Products")

    st.pyplot(fig)

    # ------------------------------------------------------
    # TOP COUNTRIES
    # ------------------------------------------------------

    st.subheader("🌍 Top 10 Countries by Revenue")

    country_sales = (
        df.groupby("Country")["TotalAmount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10,5))

    ax.bar(
        country_sales.index,
        country_sales.values
    )

    plt.xticks(rotation=45)

    ax.set_xlabel("Country")
    ax.set_ylabel("Revenue")

    st.pyplot(fig)

    # ------------------------------------------------------
    # MONTHLY SALES TREND
    # ------------------------------------------------------

    st.subheader("📈 Monthly Sales Trend")

    monthly_sales = (
        df.groupby(df["InvoiceDate"].dt.to_period("M"))
        ["TotalAmount"]
        .sum()
    )

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        monthly_sales.index.astype(str),
        monthly_sales.values,
        marker="o"
    )

    plt.xticks(rotation=90)

    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")

    st.pyplot(fig)

    # ------------------------------------------------------
    # REVENUE DISTRIBUTION
    # ------------------------------------------------------

    st.subheader("💰 Revenue Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(
        df["TotalAmount"],
        bins=40,
        edgecolor="black"
    )

    ax.set_xlabel("Revenue")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    # ------------------------------------------------------
    # TOP CUSTOMERS
    # ------------------------------------------------------

    st.subheader("👑 Top 10 Customers")

    top_customers = (
        df.groupby("CustomerID")["TotalAmount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    st.dataframe(top_customers)

    # ------------------------------------------------------
    # DOWNLOAD CLEAN DATASET
    # ------------------------------------------------------

    st.subheader("📥 Download Clean Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="Cleaned_OnlineRetail.csv",
        mime="text/csv"
    )

# ==========================================================
# RFM ANALYSIS
# ==========================================================

elif menu == "📈 RFM Analysis":

    st.header("📈 RFM Analysis")

    # ------------------------------------------------------
    # CREATE RFM DATASET
    # ------------------------------------------------------

    latest_date = df["InvoiceDate"].max()
    reference_date = latest_date + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg({

        "InvoiceDate": lambda x: (reference_date - x.max()).days,

        "InvoiceNo": "nunique",

        "TotalAmount": "sum"

    })

    rfm.columns = [

        "Recency",

        "Frequency",

        "Monetary"

    ]

    st.subheader("📋 RFM Dataset")

    st.dataframe(rfm.head(10))

    # ------------------------------------------------------
    # RECENCY DISTRIBUTION
    # ------------------------------------------------------

    st.subheader("📅 Recency Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(
        rfm["Recency"],
        bins=30,
        edgecolor="black"
    )

    ax.set_xlabel("Recency (Days)")
    ax.set_ylabel("Customers")

    st.pyplot(fig)

    # ------------------------------------------------------
    # FREQUENCY DISTRIBUTION
    # ------------------------------------------------------

    st.subheader("🔁 Frequency Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(
        rfm["Frequency"],
        bins=30,
        edgecolor="black"
    )

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Customers")

    st.pyplot(fig)

    # ------------------------------------------------------
    # MONETARY DISTRIBUTION
    # ------------------------------------------------------

    st.subheader("💰 Monetary Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(
        rfm["Monetary"],
        bins=30,
        edgecolor="black"
    )

    ax.set_xlabel("Monetary (£)")
    ax.set_ylabel("Customers")

    st.pyplot(fig)

    # ------------------------------------------------------
    # RFM BOXPLOTS
    # ------------------------------------------------------

    st.subheader("📦 RFM Boxplots")

    fig, ax = plt.subplots(figsize=(10,5))

    rfm[["Recency","Frequency","Monetary"]].boxplot(ax=ax)

    st.pyplot(fig)

    # ------------------------------------------------------
    # CORRELATION HEATMAP
    # ------------------------------------------------------

    st.subheader("🔥 Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(6,5))

    sns.heatmap(
        rfm.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    # ------------------------------------------------------
    # MONTHLY SALES TREND
    # ------------------------------------------------------

    st.subheader("📈 Monthly Sales Trend")

    monthly_sales = (
        df.groupby(df["InvoiceDate"].dt.to_period("M"))["TotalAmount"]
        .sum()
    )

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        monthly_sales.index.astype(str),
        monthly_sales.values,
        marker="o"
    )

    plt.xticks(rotation=90)

    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (£)")

    st.pyplot(fig)

    # ------------------------------------------------------
    # STANDARDIZE RFM
    # ------------------------------------------------------

    scaler_rfm = StandardScaler()

    rfm_scaled = scaler_rfm.fit_transform(
        rfm[["Recency","Frequency","Monetary"]]
    )

    st.markdown("---")

    st.header("🎯 Customer Segmentation Analysis")

# ==========================================================
# RFM ANALYSIS
# ==========================================================

elif menu == "📈 RFM Analysis":

    st.header("📈 RFM Analysis")

    # ------------------------------------------------------
    # CREATE RFM DATASET
    # ------------------------------------------------------

    latest_date = df["InvoiceDate"].max()
    reference_date = latest_date + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg({

        "InvoiceDate": lambda x: (reference_date - x.max()).days,

        "InvoiceNo": "nunique",

        "TotalAmount": "sum"

    })

    rfm.columns = [

        "Recency",

        "Frequency",

        "Monetary"

    ]

    st.subheader("📋 RFM Dataset")

    st.dataframe(rfm.head(10))

    # ------------------------------------------------------
    # RECENCY DISTRIBUTION
    # ------------------------------------------------------

    st.subheader("📅 Recency Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(
        rfm["Recency"],
        bins=30,
        edgecolor="black"
    )

    ax.set_xlabel("Recency (Days)")
    ax.set_ylabel("Customers")

    st.pyplot(fig)

    # ------------------------------------------------------
    # FREQUENCY DISTRIBUTION
    # ------------------------------------------------------

    st.subheader("🔁 Frequency Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(
        rfm["Frequency"],
        bins=30,
        edgecolor="black"
    )

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Customers")

    st.pyplot(fig)

    # ------------------------------------------------------
    # MONETARY DISTRIBUTION
    # ------------------------------------------------------

    st.subheader("💰 Monetary Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(
        rfm["Monetary"],
        bins=30,
        edgecolor="black"
    )

    ax.set_xlabel("Monetary (£)")
    ax.set_ylabel("Customers")

    st.pyplot(fig)

    # ------------------------------------------------------
    # RFM BOXPLOTS
    # ------------------------------------------------------

    st.subheader("📦 RFM Boxplots")

    fig, ax = plt.subplots(figsize=(10,5))

    rfm[["Recency","Frequency","Monetary"]].boxplot(ax=ax)

    st.pyplot(fig)

    # ------------------------------------------------------
    # CORRELATION HEATMAP
    # ------------------------------------------------------

    st.subheader("🔥 Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(6,5))

    sns.heatmap(
        rfm.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    # ------------------------------------------------------
    # MONTHLY SALES TREND
    # ------------------------------------------------------

    st.subheader("📈 Monthly Sales Trend")

    monthly_sales = (
        df.groupby(df["InvoiceDate"].dt.to_period("M"))["TotalAmount"]
        .sum()
    )

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        monthly_sales.index.astype(str),
        monthly_sales.values,
        marker="o"
    )

    plt.xticks(rotation=90)

    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (£)")

    st.pyplot(fig)

    # ------------------------------------------------------
    # STANDARDIZE RFM
    # ------------------------------------------------------

    scaler_rfm = StandardScaler()

    rfm_scaled = scaler_rfm.fit_transform(
        rfm[["Recency","Frequency","Monetary"]]
    )

    st.markdown("---")

    st.header("🎯 Customer Segmentation Analysis")

    # ------------------------------------------------------
    # ELBOW METHOD
    # ------------------------------------------------------

    st.subheader("📈 Elbow Method")

    inertia = []

    K = range(2, 11)

    for k in K:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(rfm_scaled)

        inertia.append(model.inertia_)

    fig, ax = plt.subplots(figsize=(8,5))

    ax.plot(
        list(K),
        inertia,
        marker="o"
    )

    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")

    st.pyplot(fig)

    # ------------------------------------------------------
    # SILHOUETTE SCORE
    # ------------------------------------------------------

    st.subheader("⭐ Silhouette Scores")

    scores = []

    for k in K:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(rfm_scaled)

        score = silhouette_score(
            rfm_scaled,
            labels
        )

        scores.append(score)

    score_df = pd.DataFrame({

        "Clusters": list(K),

        "Silhouette Score": scores

    })

    st.dataframe(score_df)

    fig, ax = plt.subplots(figsize=(8,5))

    ax.plot(
        score_df["Clusters"],
        score_df["Silhouette Score"],
        marker="o"
    )

    ax.set_xlabel("Clusters")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score")

    st.pyplot(fig)

    # ------------------------------------------------------
    # FINAL KMEANS MODEL
    # ------------------------------------------------------

    kmeans_model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    rfm["Cluster"] = kmeans_model.fit_predict(rfm_scaled)

    cluster_map = {

        0: "🟡 Regular",

        1: "🔴 At-Risk",

        2: "🔵 Occasional",

        3: "🟢 High-Value"

    }

    rfm["Segment"] = rfm["Cluster"].map(cluster_map)

    # ------------------------------------------------------
    # CUSTOMER SEGMENT DISTRIBUTION
    # ------------------------------------------------------

    st.subheader("📊 Customer Segment Distribution")

    segment_count = rfm["Segment"].value_counts()

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(
        segment_count.index,
        segment_count.values
    )

    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Number of Customers")

    plt.xticks(rotation=20)

    st.pyplot(fig)

    # ------------------------------------------------------
    # CLUSTER SCATTER PLOT
    # ------------------------------------------------------

    st.subheader("🎯 Customer Cluster Scatter Plot")

    fig, ax = plt.subplots(figsize=(8,6))

    scatter = ax.scatter(

        rfm["Recency"],

        rfm["Monetary"],

        c=rfm["Cluster"],

        cmap="viridis"

    )

    ax.set_xlabel("Recency")
    ax.set_ylabel("Monetary (£)")
    ax.set_title("Customer Segments")

    plt.colorbar(scatter)

    st.pyplot(fig)

    # ------------------------------------------------------
    # CLUSTER PROFILE
    # ------------------------------------------------------

    st.subheader("📋 Cluster Profile")

    profile = (
        rfm.groupby("Segment")[
            ["Recency", "Frequency", "Monetary"]
        ]
        .mean()
        .round(2)
    )

    st.dataframe(profile)

    # ------------------------------------------------------
    # SUMMARY STATISTICS
    # ------------------------------------------------------

    st.subheader("📑 RFM Summary Statistics")

    st.dataframe(rfm.describe())

# ==========================================================
# PRODUCT RECOMMENDATION
# ==========================================================

elif menu == "🎯 Product Recommendation":

    st.header("🎯 Product Recommendation System")

    st.write("""
Uses **Item-Based Collaborative Filtering (Cosine Similarity)**
to recommend products that are frequently purchased together.
""")

    # ------------------------------------------------------
    # SELECT PRODUCT
    # ------------------------------------------------------

    product_list = sorted(similarity_df.columns.tolist())

    product = st.selectbox(
        "📦 Select Product",
        product_list
    )

    if st.button("🔍 Get Recommendations"):

        recommendations = (
            similarity_df[product]
            .sort_values(ascending=False)
            .iloc[1:6]
        )

        st.success(f"Top 5 recommendations for **{product}**")

        recommendation_df = pd.DataFrame({

            "Recommended Product": recommendations.index,

            "Similarity Score": recommendations.values.round(3)

        })

        st.dataframe(
            recommendation_df,
            use_container_width=True
        )

        # --------------------------------------------------
        # BAR CHART
        # --------------------------------------------------

        st.subheader("📊 Similarity Scores")

        fig, ax = plt.subplots(figsize=(10,5))

        ax.barh(
            recommendation_df["Recommended Product"],
            recommendation_df["Similarity Score"]
        )

        ax.set_xlabel("Similarity Score")
        ax.set_ylabel("Product")
        ax.set_title("Top 5 Recommended Products")

        st.pyplot(fig)

        # --------------------------------------------------
        # HEATMAP
        # --------------------------------------------------

        st.subheader("🔥 Product Similarity Heatmap")

        heatmap_products = [product] + list(recommendations.index)

        heatmap_df = similarity_df.loc[
            heatmap_products,
            heatmap_products
        ]

        fig, ax = plt.subplots(figsize=(8,6))

        sns.heatmap(
            heatmap_df,
            annot=True,
            cmap="YlGnBu",
            ax=ax
        )

        st.pyplot(fig)

    st.markdown("---")

    st.info(
        "💡 Recommendation is based on customers who purchased similar products."
    )

# ==========================================================
# CUSTOMER SEGMENTATION
# ==========================================================

elif menu == "👤 Customer Segmentation":

    st.header("👤 Customer Segmentation Predictor")

    st.write("""
Enter the customer's **Recency**, **Frequency**, and **Monetary**
values to predict the customer segment using the trained
KMeans model.
""")

    st.markdown("---")

    st.subheader("📖 Segment Reference Guide")

    reference = pd.DataFrame({

        "Segment":[
            "🟢 High-Value",
            "🟡 Regular",
            "🔵 Occasional",
            "🔴 At-Risk"
        ],

        "Description":[
            "Frequent premium customers",
            "Steady customers",
            "Occasional buyers",
            "Customers needing retention"
        ]

    })

    st.table(reference)

    st.markdown("---")

    st.subheader("📝 Enter RFM Values")

    recency = st.number_input(

        "📅 Recency (days)",

        min_value=0,

        value=30

    )

    frequency = st.number_input(

        "🔁 Frequency",

        min_value=1,

        value=10

    )

    monetary = st.number_input(

        "💰 Monetary (£)",

        min_value=0.0,

        value=300.0

    )

    if st.button("Predict Customer Segment"):

        sample = pd.DataFrame(

            [[recency, frequency, monetary]],

            columns=[
                "Recency",
                "Frequency",
                "Monetary"
            ]

        )

        sample_scaled = scaler.transform(sample)

        prediction = kmeans.predict(sample_scaled)[0]

        cluster_map = {

            0: "🟡 Regular",

            1: "🔴 At-Risk",

            2: "🔵 Occasional",

            3: "🟢 High-Value"

        }

        st.success(
            f"Predicted Segment : {cluster_map[prediction]}"
        )

        st.markdown("---")

        st.subheader("📋 Input Summary")

        summary = pd.DataFrame({

            "Metric":[
                "Recency",
                "Frequency",
                "Monetary (£)"
            ],

            "Value":[
                recency,
                frequency,
                monetary
            ]

        })

        st.table(summary)

        st.markdown("---")

        if prediction == 3:

            st.success("""
🟢 High-Value Customer

✔ Recent Purchases

✔ High Spending

✔ Loyal Customer

✔ Premium Buyer
""")

        elif prediction == 0:

            st.info("""
🟡 Regular Customer

✔ Active Customer

✔ Moderate Spending

✔ Regular Purchases
""")

        elif prediction == 2:

            st.warning("""
🔵 Occasional Customer

✔ Purchases Occasionally

✔ Medium Spending

✔ Can Be Improved
""")

        else:

            st.error("""
🔴 At-Risk Customer

✔ Long Time Since Last Purchase

✔ Low Spending

✔ Needs Retention Campaign
""")

    st.markdown("---")

    st.subheader("🔍 Explore Existing Customers")

    segment_filter = st.selectbox(

        "Filter by Segment",

        [
            "All",
            "🟢 High-Value",
            "🟡 Regular",
            "🔵 Occasional",
            "🔴 At-Risk"
        ]

    )

    latest_date = df["InvoiceDate"].max()
    reference_date = latest_date + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg({

        "InvoiceDate": lambda x: (reference_date - x.max()).days,

        "InvoiceNo": "nunique",

        "TotalAmount": "sum"

    })

    rfm.columns = [
        "Recency",
        "Frequency",
        "Monetary"
    ]

    rfm_scaled = scaler.transform(rfm)

    rfm["Cluster"] = kmeans.predict(rfm_scaled)

    cluster_map = {

        0: "🟡 Regular",

        1: "🔴 At-Risk",

        2: "🔵 Occasional",

        3: "🟢 High-Value"

    }

    rfm["Segment"] = rfm["Cluster"].map(cluster_map)

    if segment_filter != "All":

        rfm = rfm[
            rfm["Segment"] == segment_filter
        ]

    st.write(f"Showing {len(rfm)} customers")

    st.dataframe(rfm.head(50))

# ==========================================================
# FINAL FOOTER (SINGLE CLEAN VERSION)
# ==========================================================

st.markdown("---")

st.markdown("""
<div style='text-align:center; padding:25px; font-size:16px; color:gray;'>

🚀 <b>Created by Rajeswari Rachapalli</b><br>
🛒 Shopper Spectrum Project<br><br>

✔ Customer Segmentation  
✔ Product Recommendation  
✔ RFM Analysis  
✔ Interactive Dashboard  

<br>
  Thank you for visiting!

</div>""", unsafe_allow_html=True

)# ==========================================================
# SIDEBAR PROJECT DETAILS
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Project Details")

st.sidebar.write("**Project Name**")
st.sidebar.success("🛒 Shopper Spectrum")

st.sidebar.write("**Modules**")

st.sidebar.write("🏠 Home")
st.sidebar.write("📊 Dashboard Overview")
st.sidebar.write("📈 RFM Analysis")
st.sidebar.write("🎯 Product Recommendation")
st.sidebar.write("👤 Customer Segmentation")

st.sidebar.markdown("---")

st.sidebar.info("""
Version : 1.0

Developed using

• Python

• Streamlit

• Scikit-Learn

• Pandas

• KMeans
""")

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background-color:#f5f7fb;
}

h1{
    color:#0E6EFD;
    text-align:center;
}

h2{
    color:#1F77B4;
}

h3{
    color:#2E86C1;
}

div[data-testid="metric-container"]{
    background:#ffffff;
    border-radius:12px;
    padding:15px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.10);
}

.stButton>button{
    width:100%;
    border-radius:8px;
    height:45px;
    font-size:16px;
    font-weight:bold;
}

footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# ABOUT PROJECT
# ==========================================================

with st.expander("📖 About This Project"):

    st.write("""

### Shopper Spectrum

An End-to-End Machine Learning Project for E-Commerce Analytics.

This project includes:

✔ Data Cleaning

✔ Exploratory Data Analysis (EDA)

✔ Feature Engineering

✔ RFM Analysis

✔ KMeans Customer Segmentation

✔ Product Recommendation using Collaborative Filtering

✔ Interactive Dashboard using Streamlit

✔ Business Insights

✔ Customer Prediction Module

✔ Professional Visualizations

""")

# ==========================================================
# THANK YOU
# ==========================================================

st.success("🎉 Thank you for using Shopper Spectrum!")