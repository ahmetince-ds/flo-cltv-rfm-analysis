import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import datetime as dt 
from sklearn.preprocessing import MinMaxScaler
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from eda import grab_col_names,cat_summary,num_summary
import sys
import os

os.makedirs("images", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

sys.path.append(os.path.dirname(__file__))

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format', lambda x:'%.5f' % x)

# 1 OKUMA / READ
df = pd.read_csv('C:\\Users\Win10\Desktop\my_analysis\FLO Projesi\data.csv')

def check_df(dataframe,head=5):
    print('### İLK 5 BİLGİ',dataframe.head(head))
    print('### SATIR VE SUTUN BİLGİSİ',dataframe.shape)
    print('### TİP BİLGİSİ',dataframe.dtypes)
    print('### SON 5 BILGI',dataframe.tail(head))
    print('### EKSIK DEGER BILGISI',dataframe.isnull().sum())
    print('### QUANTİLES BILGISI ###',dataframe.describe([0.0,0.05,0.50,0.95,0.99,1]).T)


check_df(df)

# 3 KOLONLAR 
df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df['total_order'] = df['order_num_total_ever_offline'] + df['order_num_total_ever_online']
df['total_value'] = df['customer_value_total_ever_online'] + df['customer_value_total_ever_offline']

# 4 EDA 
cat_cols, num_cols, cat_but_car = grab_col_names(df)

print("Cat cols:", len(cat_cols))
print("Num cols:", len(num_cols))

#---------------------------------------------
# TÜM ÖZELLİKLİ TEK FONKSİYON

for col in cat_cols[:5]:
    cat_summary(df, col, plot=True)

print('*********************** SAYISAL DEĞİŞKENLER *****************************************')

# SAYISAL DEĞİŞKEN ANALİZİ DEVAMMM....

for col in num_cols:
    num_summary(df,col,plot=True)

# Tüm numericleri tek grafikte görmek
df[num_cols].hist(figsize=(15,10))
plt.show()


# Korelasyon matrisi (ÇOK ÖNEMLİ)
plt.figure(figsize=(10,8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
plt.show()


# KORELASYON FONKSİYONLARI 

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):

    num_cols = dataframe.select_dtypes(include=['int64','float64']).columns

    corr = dataframe[num_cols].corr()
    cor_matrix = corr.abs()

    upper_triangle_matrix = cor_matrix.where(
        np.triu(np.ones(cor_matrix.shape), k=1).astype(bool)
    )

    drop_list = [col for col in upper_triangle_matrix.columns 
                 if any(upper_triangle_matrix[col] > corr_th)]

    print(f"{len(drop_list)} columns will be dropped: {drop_list}")
             

    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize':(10,10)})
        sns.heatmap(corr, cmap='RdBu', annot=True)
        plt.show()

    return drop_list

drop_list = high_correlated_cols(df, plot=True)

df = df.drop(drop_list, axis=1)

# 🔥 yeniden hesapla
cat_cols, num_cols, cat_but_car = grab_col_names(df)

print("Dropped columns:", drop_list)
print("Final shape:", df.shape)
print("Remaining numeric:", len(num_cols))
print("Remaining categorical:", len(cat_cols))
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

df.to_csv("flo_final_dataset.csv", index=False)

#-------------------------------------------------------------------------------

# RFM ANALİZ

# tüm süreçlerin FONKSİYONLAŞTIRILMASI

def create_rfm(dataframe, csv=False):

    df = dataframe.copy()

    # tarih dönüşümü
    df["last_order_date"] = pd.to_datetime(df["last_order_date"])

    # total feature'lar
    df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["total_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

    # analiz tarihi
    today_date = df["last_order_date"].max() + pd.Timedelta(days=1)

    # RFM
    rfm = df.groupby("master_id").agg({
        "last_order_date": lambda date: (today_date - date.max()).days,
        "total_order": "sum",
        "total_value": "sum"
    })

    rfm.columns = ["recency", "frequency", "monetary"]

    # skorlar
    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5,4,3,2,1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])
    rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1,2,3,4,5])

    # RFM SCORE
    rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) + 
                        rfm["frequency_score"].astype(str))

    # segment mapping
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map, regex=True)

    
    if csv:
        import os
        os.makedirs("outputs", exist_ok=True)

        rfm.to_csv("outputs/rfm.csv", index=True)
        print("RFM CSV kaydedildi → outputs/rfm.csv")

    return rfm

rfm = create_rfm(df, csv=True)

rfm_summary = rfm.groupby("segment").agg({
    "recency": "mean",
    "frequency": "mean",
    "monetary": "mean",
    "segment": "count"
}).rename(columns={"segment": "count"})

print(rfm_summary)
print(rfm["segment"].value_counts())

# 📊 GRAFİK (BURAYA)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.countplot(x=rfm["segment"], order=rfm["segment"].value_counts().index)
plt.xticks(rotation=45)
plt.title("RFM Segment Distribution")
plt.show()


# ---- CLTV KISMI

# ==============================
# CLTV HESAPLAMA (FLO DATASET)
# ==============================

def create_cltv(dataframe, csv=False):

    df = dataframe.copy()

    # =========================
    # FEATURE ENGINEERING
    # =========================

    df["total_order"] = (
        df["order_num_total_ever_online"] +
        df["order_num_total_ever_offline"]
    )

    df["total_value"] = (
        df["customer_value_total_ever_online"] +
        df["customer_value_total_ever_offline"]
    )

    # sıfıra bölme hatasını önle
    df = df[df["total_order"] > 0]

    # ortalama sipariş değeri
    df["avg_order_value"] = df["total_value"] / df["total_order"]

    # frequency (müşteri alışveriş sayısı)
    df["frequency"] = df["total_order"]

    # =========================
    # CLTV
    # =========================

    df["cltv"] = df["avg_order_value"] * df["frequency"]

    # =========================
    # SCALING
    # =========================

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    df["cltv_score"] = scaler.fit_transform(df[["cltv"]])

    # =========================
    # SEGMENTATION
    # =========================

    df["cltv_segment"] = pd.qcut(
        df["cltv_score"],
        4,
        labels=["D", "C", "B", "A"],
        duplicates="drop"
    )

    # =========================
    # FINAL DF
    # =========================

    cltv_df = df[[
        "master_id",
        "total_order",
        "total_value",
        "avg_order_value",
        "cltv",
        "cltv_score",
        "cltv_segment"
    ]]

    # =========================
    # SAVE
    # =========================

    if csv:
        import os
        os.makedirs("outputs", exist_ok=True)

        cltv_df.to_csv("outputs/cltv.csv", index=False)
        print("CLTV CSV kaydedildi → outputs/cltv.csv")

    return cltv_df

cltv_df = create_cltv(df, csv=True)

print(cltv_df.head())
print(cltv_df["cltv_segment"].value_counts())

# 📊 GRAFİK (FINAL CLEAN)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
sns.countplot(x=cltv_df["cltv_segment"], order=["A","B","C","D"])
plt.title("CLTV Segment Distribution")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# FINAL MERGE
final_df = rfm.merge(cltv_df, on="master_id", how="left")

print(final_df.head())
print(final_df.shape)
print(final_df.isnull().sum().sum())

#-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<

# 1. RFM SEGMENT ANALİZİ (EN ÖNEMLİ)
# ==============================
# RFM SEGMENT ANALYSIS
# ==============================

rfm_summary = rfm.groupby("segment").agg({
    "recency": "mean",
    "frequency": "mean",
    "monetary": "mean",
    "segment": "count"
}).rename(columns={"segment": "count"}).sort_values("monetary", ascending=False)

print("\nRFM SEGMENT SUMMARY")
print(rfm_summary)

# -<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<

# 💎 2. EN DEĞERLİ MÜŞTERİLER (CLTV + RFM)
# ==============================
# TOP CUSTOMERS (RFM + CLTV)
# ==============================

print("\nTOP 10 CUSTOMERS BY MONETARY (RFM)")
print(rfm.sort_values("monetary", ascending=False).head(10))

print("\nTOP 10 CUSTOMERS BY CLTV")
print(final_df.sort_values("cltv", ascending=False).head(10))

# -<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<

# ⚠ 3. KAMPANYA HEDEF GRUPLARI
# ==============================
# CAMPAIGN TARGET GROUPS
# ==============================

at_risk_customers = rfm[rfm["segment"] == "at_risk"]
cant_loose_customers = rfm[rfm["segment"] == "cant_loose"]

potential_customers = rfm[rfm["segment"] == "potential_loyalists"]

print("\nAT RISK CUSTOMER COUNT:", len(at_risk_customers))
print("CANT LOSE CUSTOMER COUNT:", len(cant_loose_customers))
print("POTENTIAL LOYALISTS COUNT:", len(potential_customers))

# -<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<

# 💡 4. EN AKTİF VS EN PASİF MÜŞTERİLER
# ==============================
# CUSTOMER ACTIVITY INSIGHT
# ==============================

print("\nMOST ACTIVE CUSTOMERS")
print(rfm.sort_values("frequency", ascending=False).head(10))

print("\nMOST INACTIVE / LOST VALUE CUSTOMERS")
print(rfm.sort_values("recency", ascending=False).head(10))

# -<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<

# 📊 5. CLTV SEGMENT ANALİZİ
# ==============================
# CLTV SEGMENT ANALYSIS
# ==============================

print("\nCLTV SEGMENT DISTRIBUTION")
print(cltv_df["cltv_segment"].value_counts())

# -<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-

# 🎯 6. (OPSİYONEL AMA PRO SEVİYE) KAMPANYA ÖNERİSİ TABLOSU
# ==============================
# BUSINESS RECOMMENDATION SUMMARY
# ==============================

campaign_strategy = pd.DataFrame({
    "Segment": ["champions", "loyal_customers", "potential_loyalists", "at_risk", "cant_loose"],
    "Strategy": [
        "VIP rewards, early access",
        "Loyalty program incentives",
        "Discount + onboarding campaigns",
        "Win-back campaigns",
        "Urgent retention offers"
    ]
})

print("\nCAMPAIGN STRATEGY")
print(campaign_strategy)

# -<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<

plt.savefig("images/rfm_segments.png", dpi=300, bbox_inches="tight")

plt.savefig("images/cltv_segments.png", dpi=300, bbox_inches="tight")
