import streamlit as st, pandas as pd, numpy as np, plotly.express as px
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent/"src"))
from fraud_model import clean,target_col,run,y_encode

st.set_page_config(page_title="Fraud Detection & Risk",page_icon="🛡️",layout="wide")
st.title("🛡️ End-to-End Fraud Detection & Risk Analysis")
st.caption("Transaction analytics, imbalance-aware classification, threshold tuning and investigation queue.")

uploaded=st.file_uploader("Upload transaction CSV (or use demo data)",type=["csv"])
raw=pd.read_csv(uploaded) if uploaded else pd.read_csv("data/fraud_demo.csv")
df=clean(raw); target=target_col(df)
if target is None:
    st.error("No fraud target found. Expected Fraud, is_fraud, Class or target.")
    st.stop()

threshold=st.sidebar.slider("Fraud alert threshold",0.05,0.95,0.50,0.01)
y=y_encode(df[target])
c1,c2,c3,c4=st.columns(4)
c1.metric("Transactions",f"{len(df):,}")
c2.metric("Fraud cases",f"{int(y.sum()):,}")
c3.metric("Fraud rate",f"{y.mean():.2%}")
c4.metric("Missing cells",f"{int(df.isna().sum().sum()):,}")

tab1,tab2,tab3,tab4=st.tabs(["EDA","Model Lab","Error Analysis","Risk Queue"])
with tab1:
    st.subheader("Fraud pattern exploration")
    a,b=st.columns(2)
    with a:
        st.plotly_chart(px.histogram(df,x="Amount",color=target,nbins=40,title="Transaction amount by class"),
                        use_container_width=True)
    with b:
        if "Hour" in df.columns:
            h=df.assign(_y=y).groupby("Hour")._y.mean().reset_index()
            st.plotly_chart(px.line(h,x="Hour",y="_y",markers=True,title="Fraud rate by hour"),
                            use_container_width=True)
    st.dataframe(pd.DataFrame({"dtype":df.dtypes.astype(str),"missing":df.isna().sum(),"unique":df.nunique()}),
                 use_container_width=True)

with tab2:
    results,fitted=run(df,target,threshold)
    st.dataframe(results.style.format({c:"{:.3f}" for c in results.columns if c!="Model"}),use_container_width=True)
    st.plotly_chart(px.bar(results.melt(id_vars="Model",value_vars=["Precision","Recall","F1","ROC-AUC","PR-AUC"]),
                           x="Model",y="value",color="variable",barmode="group",title="Fraud model comparison"),
                    use_container_width=True)
    st.success(f"Recommended model: **{results.iloc[0].Model}**")

with tab3:
    results,fitted=run(df,target,threshold)
    best=results.iloc[0].Model
    pipe,Xt,yt,proba,pred=fitted[best]
    err=Xt.copy()
    err["Actual_Fraud"]=yt.values
    err["Predicted_Fraud"]=pred
    err["Fraud_Probability"]=proba
    fp=err[(err.Actual_Fraud==0)&(err.Predicted_Fraud==1)]
    fn=err[(err.Actual_Fraud==1)&(err.Predicted_Fraud==0)]
    a,b=st.columns(2)
    a.metric("False positives",len(fp))
    b.metric("False negatives",len(fn))
    st.write("False positives = legitimate transactions flagged for review. False negatives = fraud that escaped the selected threshold.")
    st.dataframe(err.sort_values("Fraud_Probability",ascending=False).head(50),use_container_width=True)

with tab4:
    results,fitted=run(df,target,threshold)
    best=results.iloc[0].Model
    pipe=fitted[best][0]
    X=df.drop(columns=[target])
    ids=X["Transaction_ID"] if "Transaction_ID" in X.columns else pd.Series(np.arange(len(X)),name="Transaction_ID")
    Xin=X.drop(columns=["Transaction_ID"]) if "Transaction_ID" in X.columns else X
    proba=pipe.predict_proba(Xin)[:,1]
    queue=df.copy()
    queue["Fraud_Probability"]=proba
    queue["Risk"]=pd.cut(proba,[0,.25,.60,.85,1],labels=["Low","Medium","High","Critical"],include_lowest=True)
    st.plotly_chart(px.histogram(queue,x="Fraud_Probability",color="Risk",nbins=30,title="Risk distribution"),
                    use_container_width=True)
    st.dataframe(queue.sort_values("Fraud_Probability",ascending=False).head(100),use_container_width=True)
    st.download_button("Download investigation queue",queue.sort_values("Fraud_Probability",ascending=False).to_csv(index=False),
                       "fraud_investigation_queue.csv","text/csv")
    st.markdown("### Practical controls")
    st.write("- Route Critical transactions to immediate manual review or step-up authentication.")
    st.write("- Tune the threshold according to the relative cost of missed fraud vs false alerts.")
    st.write("- Monitor model drift, fraud prevalence and alert-review capacity after deployment.")
    st.write("- Use PR-AUC when fraud is rare; ROC-AUC alone can look optimistic under severe imbalance.")
