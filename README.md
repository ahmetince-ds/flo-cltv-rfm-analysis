# 📊 Customer Segmentation & CLTV Analysis (RFM + CLTV)

## 🎯 Project Overview

This project focuses on **customer segmentation and customer lifetime value (CLTV) analysis** using retail transaction data.

The main objective is to transform raw customer transaction data into actionable business insights by using data science techniques such as RFM analysis and CLTV modeling.

---

## 📌 Business Problem

Companies aim to answer the following business questions:

- Who are my most valuable customers?
- Which customers are at risk of churning?
- Which customers should be targeted with marketing campaigns?
- How can customer retention and lifetime value be improved?

---

## 📂 Dataset

This project uses the **Online Retail II dataset**:

Dataset Link:  
https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

It contains:

- Customer ID (master_id)
- Invoice information
- Product quantities
- Prices
- Transaction dates

---

## 🧠 Methodology

### 1. Exploratory Data Analysis (EDA)

- Missing value analysis
- Data type checks
- Outlier and distribution analysis
- Feature engineering

---

### 2. Feature Engineering

Created new variables:

- `total_order` → total number of purchases (online + offline)
- `total_value` → total spending
- `avg_order_value` → average basket value per customer

---

### 3. RFM Analysis

Customers are segmented based on:

- **Recency** → How recently a customer purchased
- **Frequency** → How often they purchase
- **Monetary** → How much they spend

### 🔹 RFM Segments:

- Champions
- Loyal Customers
- Potential Loyalists
- At Risk
- Hibernating
- Need Attention
- New Customers
- Cant Lose Customers

---

## 💰 CLTV Analysis

Customer Lifetime Value is calculated using:

- Average Order Value
- Purchase Frequency
- Total Customer Revenue

Then scaled using MinMaxScaler and segmented into:

- A (High value customers)
- B
- C
- D (Low value customers)

---

## 📊 Key Insights

- High-value customers (Champions) identified for loyalty strategies
- At-risk customers detected for reactivation campaigns
- New customers identified for onboarding campaigns
- Customer base segmented for targeted marketing strategies
- Revenue potential clearly measurable per segment

---

## 📈 Visualizations

### RFM Segment Distribution
![RFM Segments](images/rfm_segments.png)

### CLTV Segment Distribution
![CLTV Segments](images/cltv_segments.png)

---

## 📁 Project Structure

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Lifetimes

---

## 🚀 How to Run

Clone the repository:

```bash
git clone https://github.com/ahmetince-ds/flo-cltv-rfm-analysis
