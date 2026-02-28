
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="MN Income Shares + GAI* Demo", layout="wide")
st.title("MN Income Shares (Statute Table) + GAI* Demo Dashboard")
if table_file is None and not table_path.exists():
    st.error("Missing default MN Basic Support Table CSV. Please upload it in the sidebar.")
    st.stop()

table_df = pd.read_csv(table_file) if table_file is not None else load_table(table_path)

@st.cache_data
def load_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["pics_min","pics_max","children_count","bs_total"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {path}.")
    df = df.copy()
    df["pics_min"] = pd.to_numeric(df["pics_min"])
    df["pics_max"] = pd.to_numeric(df["pics_max"])
    df["children_count"] = pd.to_numeric(df["children_count"]).astype(int)
    df["bs_total"] = pd.to_numeric(df["bs_total"])
    return df.sort_values(["children_count","pics_min"]).reset_index(drop=True)

@st.cache_data
def load_fpl(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["year","hh_size","fpl_annual"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {path}.")
    df["year"] = pd.to_numeric(df["year"]).astype(int)
    df["hh_size"] = pd.to_numeric(df["hh_size"]).astype(int)
    df["fpl_annual"] = pd.to_numeric(df["fpl_annual"])
    return df

def get_fpl_annual(fpl_df: pd.DataFrame, year: int, hh_size: int) -> float:
    sub = fpl_df[(fpl_df["year"]==year) & (fpl_df["hh_size"]==hh_size)]
    if len(sub)==0:
        near = fpl_df.iloc[(fpl_df["year"]-year).abs().argsort()[:1]]
        return float(near["fpl_annual"].iloc[0])
    return float(sub["fpl_annual"].iloc[0])

def lookup_basic_support_total(table: pd.DataFrame, pics_combined_cap: float, children: int) -> float:
    sub = table[table["children_count"]==children]
    hit = sub[(sub["pics_min"]<=pics_combined_cap) & (pics_combined_cap<=sub["pics_max"])]
    if len(hit)==0:
        if pics_combined_cap < sub["pics_min"].min():
            return float(sub.loc[sub["pics_min"].idxmin(),"bs_total"])
        return float(sub.loc[sub["pics_max"].idxmax(),"bs_total"])
    return float(hit["bs_total"].iloc[0])

def compute_income_shares(table, ncp_gross_m, cp_gross_m, children, ssr_ncp, ssr_cp,
                          deductions_ncp=0.0, deductions_cp=0.0, cap_combined_pics=20000.0):
    pics_ncp = max(0.0, ncp_gross_m - ssr_ncp - deductions_ncp)
    pics_cp  = max(0.0, cp_gross_m  - ssr_cp  - deductions_cp)
    pics_combined = pics_ncp + pics_cp
    pics_cap = min(pics_combined, cap_combined_pics)
    bs_total = lookup_basic_support_total(table, pics_cap, children)
    share_ncp = (pics_ncp / pics_combined) if pics_combined>0 else np.nan
    bs_ncp = share_ncp * bs_total if np.isfinite(share_ncp) else np.nan
    return dict(PICS_NCP=pics_ncp, PICS_CP=pics_cp, PICS_Combined=pics_combined, PICS_Cap=pics_cap,
                BS_Total_Table=bs_total, Share_NCP=share_ncp, IS_BS_NCP=bs_ncp)

def score_band(value, bands):
    for lo, hi, sc in bands:
        if value>=lo and value<=hi:
            return float(sc)
    if value < bands[0][0]:
        return float(bands[0][2])
    return float(bands[-1][2])

def la_score(wage_ratio, burden):
    wage_sc = score_band(wage_ratio, [
        (1.10, 1e9, 100),(0.80, 1.099999, 80),(0.60, 0.799999, 60),(0.40, 0.599999, 40),(0.00, 0.399999, 20),
    ])
    burden_sc = score_band(burden, [
        (0.00, 0.149999, 100),(0.15, 0.249999, 80),(0.25, 0.349999, 60),(0.35, 0.449999, 40),(0.45, 1e9, 20),
    ])
    return 0.5*wage_sc + 0.5*burden_sc

def pp_score(ncp_gross_m, order_m, ssr_ncp, ssr_applied, excess_imputation):
    disp = ncp_gross_m - ssr_ncp
    if disp <= 0:
        base = 10.0
    else:
        disp_burden = order_m / disp
        base = score_band(disp_burden, [
            (0.00, 0.299999, 100),(0.30, 0.399999, 70),(0.40, 0.499999, 40),(0.50, 1e9, 20),
        ])
    base = min(100.0, base + (10.0 if ssr_applied==1 else 0.0))
    base = max(0.0, base - (15.0 if excess_imputation==1 else 0.0))
    return base

def cs_score(collection_rate, missed_rate):
    cr_sc = score_band(collection_rate, [
        (0.90, 1.0, 100),(0.75, 0.899999, 80),(0.60, 0.749999, 60),(0.40, 0.599999, 40),(0.00, 0.399999, 20),
    ])
    pr = 1.0 - missed_rate
    pr_sc = score_band(pr, [
        (0.90, 1.0, 100),(0.75, 0.899999, 80),(0.60, 0.749999, 60),(0.40, 0.599999, 40),(0.00, 0.399999, 20),
    ])
    return 0.6*cr_sc + 0.4*pr_sc

def cp_score(arrears_to_wage, wage_to_200fpl):
    atw = score_band(arrears_to_wage, [
        (0.00, 0.249999, 100),(0.25, 0.749999, 80),(0.75, 1.249999, 60),(1.25, 1.999999, 40),(2.00, 1e9, 20),
    ])
    w2f = score_band(wage_to_200fpl, [
        (2.50, 1e9, 100),(2.00, 2.499999, 80),(1.50, 1.999999, 60),(1.25, 1.499999, 40),(0.00, 1.249999, 20),
    ])
    return 0.5*atw + 0.5*w2f

def eq_score(relative_risk):
    val = 100.0 - 100.0*abs(relative_risk - 1.0)
    return float(np.clip(val, 0, 100))

def rag(label, x, green_lo, amber_lo):
    if x >= green_lo: return f"🟢 {label}: Green"
    if x >= amber_lo: return f"🟡 {label}: Amber"
    return f"🔴 {label}: Red"

def logistic(z): 
    return 1.0/(1.0+np.exp(-z))

def predict_compliance(gai_no_cs, burden, ssr_applied, excess_imp, cp_scaled, district_fe=0.0, coefs=None):
    if coefs is None:
        coefs = {"alpha": -0.3, "b_gai": 0.03, "b_burden": -1.5, "b_ssr": 0.15, "b_eximp": -0.20, "b_cp": 0.8}
    eta = (coefs["alpha"]
           + coefs["b_gai"]*gai_no_cs
           + coefs["b_burden"]*burden
           + coefs["b_ssr"]*ssr_applied
           + coefs["b_eximp"]*excess_imp
           + coefs["b_cp"]*cp_scaled
           + district_fe)
    return float(logistic(eta))

def apply_burden_cap(order_m, income_m, cap=None):
    if cap is None or income_m<=0:
        return order_m
    return min(order_m, cap*income_m)

with st.sidebar:
    st.header("Data Inputs")
    table_file = st.file_uploader("Upload MN Basic Support Table CSV", type=["csv"])
    fpl_file = st.file_uploader("Upload FPL table CSV (optional)", type=["csv"])
    st.divider()

    st.header("Case Inputs (single-case demo)")
    ncp_gross_m = st.number_input("NCP gross monthly income ($)", min_value=0.0, value=4000.0, step=50.0)
    cp_gross_m  = st.number_input("CP gross monthly income ($)", min_value=0.0, value=3000.0, step=50.0)
    children = st.number_input("# joint children (1–6)", min_value=1, max_value=6, value=2, step=1)
    order_actual_m = st.number_input("Actual court order (basic support) monthly ($)", min_value=0.0, value=900.0, step=25.0)

    ssr_ncp = st.number_input("SSR (NCP) monthly ($)", min_value=0.0, value=1359.0, step=10.0)
    ssr_cp  = st.number_input("SSR (CP) monthly ($)", min_value=0.0, value=1359.0, step=10.0)
    deductions_ncp = st.number_input("NCP deductions for PICS (monthly $) (optional)", min_value=0.0, value=0.0, step=10.0)
    deductions_cp  = st.number_input("CP deductions for PICS (monthly $) (optional)", min_value=0.0, value=0.0, step=10.0)

    st.divider()
    st.header("GAI Inputs (demo)")
    county_median_wage_m = st.number_input("County median wage (monthly $)", min_value=0.0, value=5000.0, step=50.0)
    ncp_yearly_wage = st.number_input("NCP yearly wage ($)", min_value=0.0, value=35000.0, step=500.0)
    arrears_balance = st.number_input("Arrears balance ($)", min_value=0.0, value=12000.0, step=250.0)
    collection_rate = st.slider("Current support collection rate", 0.0, 1.0, 0.81, 0.01)
    missed_rate = st.slider("Missed payment rate", 0.0, 1.0, 0.12, 0.01)
    ssr_applied = st.selectbox("SSR applied? (indicator)", [0,1], index=1)
    excess_imputation = st.selectbox("Excess imputation? (indicator)", [0,1], index=0)
    relative_risk = st.slider("Equity relative risk (RR) for group in district", 0.2, 2.0, 1.0, 0.01)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
table_path = BASE_DIR / "data" / "mn_basic_support_table_template.csv"
fpl_path = "data/fpl_table_template.csv"
table_df = pd.read_csv(table_file) if table_file is not None else load_table(table_path)
fpl_df = pd.read_csv(fpl_file) if fpl_file is not None else load_fpl(fpl_path)

tabs = st.tabs(["Income Shares (Statute)", "GAI Scoring", "Side-by-side + RAG", "Simulation Lab"])

with tabs[0]:
    st.subheader("Income Shares (Statute Table) Calculator")
    res = compute_income_shares(table_df, ncp_gross_m, cp_gross_m, children, ssr_ncp, ssr_cp, deductions_ncp, deductions_cp)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("PICS (NCP)", f"${res['PICS_NCP']:,.0f}")
    c2.metric("PICS (CP)", f"${res['PICS_CP']:,.0f}")
    c3.metric("Combined PICS", f"${res['PICS_Combined']:,.0f}")
    c4.metric("Combined PICS (cap)", f"${res['PICS_Cap']:,.0f}")
    c5,c6,c7 = st.columns(3)
    c5.metric("Table Total Basic Support", f"${res['BS_Total_Table']:,.0f}")
    c6.metric("NCP Share", f"{(res['Share_NCP']*100 if np.isfinite(res['Share_NCP']) else np.nan):.1f}%")
    presumptive = res["IS_BS_NCP"] if np.isfinite(res["IS_BS_NCP"]) else 0.0
    c7.metric("Presumptive NCP Basic Support", f"${presumptive:,.0f}")
    st.metric("Actual − Presumptive (Gap)", f"${(order_actual_m - presumptive):,.0f}")
    with st.expander("Show currently loaded table CSV"):
        st.dataframe(table_df, use_container_width=True)

with tabs[1]:
    st.subheader("GAI* Component Scoring (demo thresholds)")
    income_m = ncp_gross_m
    wage_m = ncp_yearly_wage/12.0 if ncp_yearly_wage>0 else 0.0
    wage_ratio = wage_m / county_median_wage_m if county_median_wage_m>0 else 0.0
    burden = order_actual_m / income_m if income_m>0 else 1.0
    LA = la_score(wage_ratio, burden)
    PP = pp_score(income_m, order_actual_m, ssr_ncp, ssr_applied, excess_imputation)
    CS = cs_score(collection_rate, missed_rate)
    arrears_to_wage = arrears_balance / ncp_yearly_wage if ncp_yearly_wage>0 else 10.0
    year = int(pd.Timestamp.today().year)
    fpl_annual = get_fpl_annual(fpl_df, year, 1)
    wage_to_200fpl = ncp_yearly_wage / (2.0*fpl_annual) if fpl_annual>0 else 0.0
    CP = cp_score(arrears_to_wage, wage_to_200fpl)
    EQ = eq_score(relative_risk)
    colA,colB,colC,colD,colE = st.columns(5)
    colA.metric("LA", f"{LA:.1f}")
    colB.metric("PP", f"{PP:.1f}")
    colC.metric("CS", f"{CS:.1f}")
    colD.metric("CP", f"{CP:.1f}")
    colE.metric("EQ", f"{EQ:.1f}")
    GAI_full = 0.2*(LA+PP+CS+CP+EQ)
    GAI_no_cs = 0.25*(LA+PP+CP+EQ)
    st.divider()
    c1,c2 = st.columns(2)
    c1.metric("GAI* (full)", f"{GAI_full:.1f}")
    c2.metric("GAI(-CS)*", f"{GAI_no_cs:.1f}")

with tabs[2]:
    st.subheader("Side-by-side + RAG Status")
    res = compute_income_shares(table_df, ncp_gross_m, cp_gross_m, children, ssr_ncp, ssr_cp, deductions_ncp, deductions_cp)
    presumptive = res["IS_BS_NCP"] if np.isfinite(res["IS_BS_NCP"]) else 0.0
    gap = order_actual_m - presumptive
    gap_pct = abs(gap)/presumptive if presumptive>0 else np.nan

    align_rag = "🟢 Alignment: Green"
    if np.isfinite(gap_pct):
        if gap_pct > 0.25: align_rag = "🔴 Alignment: Red"
        elif gap_pct > 0.10: align_rag = "🟡 Alignment: Amber"
    else:
        align_rag = "🟡 Alignment: Amber (no presumptive computed)"

    disp_income = ncp_gross_m - ssr_ncp
    disp_burden = (order_actual_m/disp_income) if disp_income>0 else 10.0
    aff_rag = "🟢 Affordability: Green" if disp_burden < 0.30 else ("🟡 Affordability: Amber" if disp_burden <= 0.40 else "🔴 Affordability: Red")

    c1,c2,c3 = st.columns(3)
    c1.metric("Presumptive (Statute) Basic Support", f"${presumptive:,.0f}")
    c2.metric("Actual Basic Support", f"${order_actual_m:,.0f}")
    c3.metric("Gap (Actual − Presumptive)", f"${gap:,.0f}")
    st.write(f"- {align_rag}")
    st.write(f"- {aff_rag} (Disposable burden = {disp_burden:.2f})")

    income_m = ncp_gross_m
    wage_m = ncp_yearly_wage/12.0 if ncp_yearly_wage>0 else 0.0
    wage_ratio = wage_m / county_median_wage_m if county_median_wage_m>0 else 0.0
    burden = order_actual_m / income_m if income_m>0 else 1.0
    LA = la_score(wage_ratio, burden)
    PP = pp_score(income_m, order_actual_m, ssr_ncp, ssr_applied, excess_imputation)
    CS = cs_score(collection_rate, missed_rate)
    arrears_to_wage = arrears_balance / ncp_yearly_wage if ncp_yearly_wage>0 else 10.0
    year = int(pd.Timestamp.today().year)
    fpl_annual = get_fpl_annual(fpl_df, year, 1)
    wage_to_200fpl = ncp_yearly_wage / (2.0*fpl_annual) if fpl_annual>0 else 0.0
    CP = cp_score(arrears_to_wage, wage_to_200fpl)
    EQ = eq_score(relative_risk)
    GAI_full = 0.2*(LA+PP+CS+CP+EQ)
    GAI_no_cs = 0.25*(LA+PP+CP+EQ)

    st.divider()
    st.markdown("### GAI RAG Status")
    st.write(rag("LA", LA, 70, 55))
    st.write(rag("PP", PP, 70, 55))
    st.write(rag("CS", CS, 75, 60))
    st.write(rag("CP", CP, 70, 55))
    st.write(rag("EQ", EQ, 80, 60))

    reds = sum([LA<55, PP<55, CS<60, CP<55, EQ<60])
    overall = "🟢 Overall: Green"
    if (PP<55) or (CP<55) or (reds>=2):
        overall = "🔴 Overall: Red"
    else:
        ambers = sum([(55<=LA<70),(55<=PP<70),(60<=CS<75),(55<=CP<70),(60<=EQ<80)])
        if (55<=PP<70) or (55<=CP<70) or (ambers>=2):
            overall = "🟡 Overall: Amber"
    st.divider()
    st.write(overall)
    st.write(f"GAI*={GAI_full:.1f} | GAI(-CS)*={GAI_no_cs:.1f}")

with tabs[3]:
    st.subheader("Simulation Lab: Moving thresholds → predicted compliance")
    income_m = ncp_gross_m
    burden_base = order_actual_m / income_m if income_m>0 else 1.0
    wage_m = ncp_yearly_wage/12.0 if ncp_yearly_wage>0 else 0.0
    wage_ratio = wage_m / county_median_wage_m if county_median_wage_m>0 else 0.0
    LA = la_score(wage_ratio, burden_base)
    PP = pp_score(income_m, order_actual_m, ssr_ncp, ssr_applied, excess_imputation)
    arrears_to_wage = arrears_balance / ncp_yearly_wage if ncp_yearly_wage>0 else 10.0
    year = int(pd.Timestamp.today().year)
    fpl_annual = get_fpl_annual(fpl_df, year, 1)
    wage_to_200fpl = ncp_yearly_wage / (2.0*fpl_annual) if fpl_annual>0 else 0.0
    CP = cp_score(arrears_to_wage, wage_to_200fpl)
    EQ = eq_score(relative_risk)
    GAI_no_cs = 0.25*(LA+PP+CP+EQ)
    p_base = predict_compliance(GAI_no_cs, burden_base, ssr_applied, excess_imputation, CP/100.0)

    c1,c2,c3 = st.columns(3)
    c1.metric("Baseline burden", f"{burden_base:.2f}")
    c2.metric("GAI(-CS)*", f"{GAI_no_cs:.1f}")
    c3.metric("Predicted compliance", f"{p_base*100:.1f}%")

    st.divider()
    cap = st.select_slider("Set burden cap (Order/Income)", options=[None, 0.35, 0.30, 0.25, 0.20], value=0.30,
                           format_func=lambda x: "No cap" if x is None else f"{int(x*100)}%")
    new_order = apply_burden_cap(order_actual_m, income_m, cap)
    new_burden = new_order / income_m if income_m>0 else 1.0
    LA2 = la_score(wage_ratio, new_burden)
    PP2 = pp_score(income_m, new_order, ssr_ncp, ssr_applied, excess_imputation)
    GAI2 = 0.25*(LA2+PP2+CP+EQ)
    p2 = predict_compliance(GAI2, new_burden, ssr_applied, excess_imputation, CP/100.0)

    d1,d2,d3,d4 = st.columns(4)
    d1.metric("New order", f"${new_order:,.0f}")
    d2.metric("New burden", f"{new_burden:.2f}")
    d3.metric("New GAI(-CS)*", f"{GAI2:.1f}")
    d4.metric("Predicted compliance", f"{p2*100:.1f}%")
    st.success(f"Δ predicted compliance: {(p2-p_base)*100:+.2f} percentage points")
