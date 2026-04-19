"""
=============================================================
  Credit Card Fraud Detection — Training Pipeline
  Muhammad Zeeshan | B01799050 | UWS COMP11128
  Dataset: ULB Credit Card Fraud (Dal Pozzolo et al., 2015)
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
=============================================================

Run:
    python train.py
    python train.py --data path/to/creditcard.csv

Expected results (identical every run):
    Logistic Regression : Acc=0.9745  Prec=0.0581  Rec=0.9082  F1=0.1093  AUC=0.9719
    Random Forest       : Acc=0.9988  Prec=0.5931  Rec=0.8776  F1=0.7078  AUC=0.9782
    SVM                 : Acc=0.9817  Prec=0.0780  Rec=0.8878  F1=0.1433  AUC=0.9805
"""

# ── Lock ALL random seeds FIRST ───────────────────────────────
import os, sys
os.environ["PYTHONHASHSEED"] = "42"
import random
random.seed(42)
import numpy as np
np.random.seed(42)

# ── Other imports ─────────────────────────────────────────────
import time, argparse, warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, learning_curve)
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score, roc_auc_score,
                                     confusion_matrix, roc_curve,
                                     classification_report)
import joblib

CLR = {"Logistic Regression":"#2563EB","Random Forest":"#16A34A","SVM":"#DC2626"}


def load_data(path):
    print(f"\n{'='*60}\n  STEP 1 — Loading dataset:  {path}\n{'='*60}")
    if not os.path.exists(path):
        print(f"\n  ERROR: '{path}' not found.")
        print("  Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("  Save as:  data/creditcard.csv"); sys.exit(1)
    df = pd.read_csv(path)
    nf = int(df["Class"].sum()); nl = int((df["Class"]==0).sum())
    print(f"  Rows: {len(df):,}  |  Fraud: {nf} ({nf/len(df)*100:.4f}%)  |  Legit: {nl:,}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    return df


def eda_figures(df):
    print(f"\n{'='*60}\n  STEP 2 — EDA Figures\n{'='*60}")
    os.makedirs("reports", exist_ok=True)
    fraud = df[df["Class"]==1]; legit = df[df["Class"]==0]

    # Fig 1
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4))
    a1.bar(["Legitimate","Fraud"],[len(legit),len(fraud)],color=["#2563EB","#DC2626"],edgecolor="white",width=0.5)
    a1.set_title("Class Distribution — Count",fontsize=12,fontweight="bold"); a1.set_ylabel("Transactions")
    for i,v in enumerate([len(legit),len(fraud)]): a1.text(i,v*1.01,f"{v:,}",ha="center",fontsize=10,fontweight="bold")
    a1.spines[["top","right"]].set_visible(False)
    pcts=[(1-df["Class"].mean())*100,df["Class"].mean()*100]
    a2.bar(["Legitimate","Fraud"],pcts,color=["#2563EB","#DC2626"],edgecolor="white",width=0.5)
    a2.set_title("Class Distribution — Percentage",fontsize=12,fontweight="bold"); a2.set_ylabel("Percentage (%)")
    a2.text(0,pcts[0]+0.3,f"{pcts[0]:.2f}%",ha="center",fontsize=10,fontweight="bold")
    a2.text(1,pcts[1]+0.02,f"{pcts[1]:.4f}%",ha="center",fontsize=10,fontweight="bold")
    a2.spines[["top","right"]].set_visible(False)
    plt.suptitle("Fig 1. Class Distribution — ULB Credit Card Fraud Dataset\nDal Pozzolo et al. (2015)",fontsize=11,style="italic")
    plt.tight_layout(); plt.savefig("reports/fig1_class_distribution.png",dpi=150,bbox_inches="tight"); plt.close()
    print("  Fig 1 saved")

    # Fig 2
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4))
    a1.hist(fraud["Amount"],bins=40,color="#DC2626",alpha=0.85,edgecolor="white")
    a1.axvline(fraud["Amount"].mean(),color="black",lw=2,ls="--",label=f"Mean: \u20ac{fraud['Amount'].mean():.2f}")
    a1.set_title("Amount \u2014 Fraud",fontsize=12,fontweight="bold"); a1.set_xlabel("Amount (\u20ac)"); a1.set_ylabel("Frequency")
    a1.legend(fontsize=9); a1.spines[["top","right"]].set_visible(False)
    a2.hist(legit["Amount"].clip(0,500),bins=40,color="#2563EB",alpha=0.85,edgecolor="white")
    a2.axvline(legit["Amount"].mean(),color="black",lw=2,ls="--",label=f"Mean: \u20ac{legit['Amount'].mean():.2f}")
    a2.set_title("Amount \u2014 Legitimate (clipped \u20ac500)",fontsize=12,fontweight="bold"); a2.set_xlabel("Amount (\u20ac)"); a2.set_ylabel("Frequency")
    a2.legend(fontsize=9); a2.spines[["top","right"]].set_visible(False)
    plt.suptitle("Fig 2. Transaction Amount Distribution by Class",fontsize=11,style="italic")
    plt.tight_layout(); plt.savefig("reports/fig2_amount_by_class.png",dpi=150,bbox_inches="tight"); plt.close()
    print("  Fig 2 saved")

    # Fig 3
    fh=(fraud["Time"]%86400)/3600; lh=(legit["Time"]%86400)/3600
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4))
    a1.hist(fh,bins=24,color="#DC2626",alpha=0.85,edgecolor="white"); a1.set_title("Hour of Day \u2014 Fraud",fontsize=12,fontweight="bold"); a1.set_xlabel("Hour (0\u201324)"); a1.set_ylabel("Frequency"); a1.spines[["top","right"]].set_visible(False)
    a2.hist(lh,bins=24,color="#2563EB",alpha=0.85,edgecolor="white"); a2.set_title("Hour of Day \u2014 Legitimate",fontsize=12,fontweight="bold"); a2.set_xlabel("Hour (0\u201324)"); a2.set_ylabel("Frequency"); a2.spines[["top","right"]].set_visible(False)
    plt.suptitle("Fig 3. Transaction Time Distribution by Class",fontsize=11,style="italic")
    plt.tight_layout(); plt.savefig("reports/fig3_time_by_class.png",dpi=150,bbox_inches="tight"); plt.close()
    print("  Fig 3 saved")


def engineer_features(df):
    print(f"\n{'='*60}\n  STEP 3 — Feature Engineering\n{'='*60}")
    df=df.copy()
    df["log_amount"]  = np.log1p(df["Amount"])
    df["hour_of_day"] = (df["Time"]%86400)/3600
    df["is_night"]    = ((df["hour_of_day"]<6)|(df["hour_of_day"]>22)).astype(int)
    feat_cols=[f"V{i}" for i in range(1,29)]+["log_amount","hour_of_day","is_night"]
    print(f"  Features: {len(feat_cols)} total (28 PCA + 3 engineered)")
    return df, feat_cols


def preprocess(df, feat_cols):
    print(f"\n{'='*60}\n  STEP 4 — Preprocessing\n{'='*60}")
    X=df[feat_cols].values; y=df["Class"].values

    # Stratified 80/20 split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    print(f"  Train: {X_train.shape} | Fraud: {y_train.sum()} ({y_train.mean()*100:.4f}%)")
    print(f"  Test:  {X_test.shape}  | Fraud: {y_test.sum()} ({y_test.mean()*100:.4f}%)")

    # Scale — fitted on train only
    scaler=StandardScaler(); X_train_sc=scaler.fit_transform(X_train); X_test_sc=scaler.transform(X_test)
    os.makedirs("models",exist_ok=True); joblib.dump(scaler,"models/scaler.pkl")
    print("  Scaler saved -> models/scaler.pkl")

    # ONE fixed RNG for ALL sampling — this is the key to reproducibility
    RNG=np.random.RandomState(42)
    fraud_idx=np.where(y_train==1)[0]; legit_idx=np.where(y_train==0)[0]

    # 10:1 for LR + RF
    l10=RNG.choice(legit_idx,size=len(fraud_idx)*10,replace=False)
    i10=RNG.permutation(np.concatenate([fraud_idx,l10]))
    X_bal=X_train_sc[i10]; y_bal=y_train[i10]

    # 3:1 for SVM
    l3=RNG.choice(legit_idx,size=len(fraud_idx)*3,replace=False)
    i3=RNG.permutation(np.concatenate([fraud_idx,l3]))
    X_svm=X_train_sc[i3]; y_svm=y_train[i3]

    print(f"\n  Balanced 10:1 (LR+RF): {X_bal.shape} | Fraud:{y_bal.sum()} Legit:{(y_bal==0).sum()}")
    print(f"  Balanced  3:1 (SVM):   {X_svm.shape} | Fraud:{y_svm.sum()} Legit:{(y_svm==0).sum()}")
    return X_bal,y_bal,X_svm,y_svm,X_test_sc,y_test


def train_models(X_bal,y_bal,X_svm,y_svm):
    print(f"\n{'='*60}\n  STEP 5 — Training Models\n{'='*60}")
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    configs=[
        ("Logistic Regression",LogisticRegression(C=0.01,max_iter=1000,class_weight="balanced",random_state=42,n_jobs=-1),X_bal,y_bal),
        ("Random Forest",RandomForestClassifier(n_estimators=100,max_depth=12,min_samples_leaf=2,class_weight="balanced",n_jobs=-1,random_state=42),X_bal,y_bal),
        ("SVM",SVC(C=1,kernel="rbf",gamma="scale",class_weight="balanced",probability=True,random_state=42),X_svm,y_svm),
    ]
    trained={}
    for name,model,Xtr,ytr in configs:
        print(f"\n  Training: {name} ...")
        t0=time.perf_counter(); model.fit(Xtr,ytr); elapsed=time.perf_counter()-t0
        cvf1=cross_val_score(model,Xtr,ytr,cv=cv,scoring="f1",n_jobs=-1)
        print(f"  Time: {elapsed:.1f}s  |  CV F1: {cvf1.mean():.4f} +/- {cvf1.std():.4f}")
        fname=name.lower().replace(" ","_"); joblib.dump(model,f"models/{fname}.pkl")
        print(f"  Saved -> models/{fname}.pkl")
        trained[name]={"model":model,"cv_f1_mean":float(cvf1.mean()),"cv_f1_std":float(cvf1.std())}
    return trained


def evaluate(trained,X_test,y_test):
    print(f"\n{'='*60}\n  STEP 6 — Evaluation  (test:{len(y_test):,}  fraud:{y_test.sum()})\n{'='*60}")
    results={}
    for name,info in trained.items():
        m=info["model"]; yp=m.predict(X_test); ypr=m.predict_proba(X_test)[:,1]; cm=confusion_matrix(y_test,yp)
        results[name]={"accuracy":float(accuracy_score(y_test,yp)),"precision":float(precision_score(y_test,yp,zero_division=0)),
            "recall":float(recall_score(y_test,yp,zero_division=0)),"f1":float(f1_score(y_test,yp,zero_division=0)),
            "auc":float(roc_auc_score(y_test,ypr)),"cm":cm,"yp":yp,"ypr":ypr,
            "cv_f1_mean":info["cv_f1_mean"],"cv_f1_std":info["cv_f1_std"]}
        r=results[name]
        print(f"\n  -- {name} --")
        print(f"  Accuracy:{r['accuracy']:.4f}  Precision:{r['precision']:.4f}  Recall:{r['recall']:.4f}  F1:{r['f1']:.4f}  AUC:{r['auc']:.4f}")
        print(f"  CV F1:{r['cv_f1_mean']:.4f}+/-{r['cv_f1_std']:.4f}")
        print(f"  TN={cm[0,0]:,}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")
        print(classification_report(y_test,yp,target_names=["Legit","Fraud"],zero_division=0))
    return results


def result_figures(results,trained,y_test,X_bal,y_bal,X_svm,y_svm):
    print(f"\n{'='*60}\n  STEP 7 — Figures\n{'='*60}")
    names=list(results.keys()); cv5=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

    # Fig 4 Confusion matrices
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,name in zip(axes,names):
        cm=results[name]["cm"]; im=ax.imshow(cm,cmap="Blues")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Legit","Fraud"],fontsize=9); ax.set_yticklabels(["Legit","Fraud"],fontsize=9,rotation=90,va="center")
        ax.set_xlabel("Predicted",fontsize=9); ax.set_ylabel("Actual",fontsize=9)
        ax.set_title(f"{name}\nConfusion Matrix",fontsize=10,fontweight="bold")
        for i in range(2):
            for j in range(2): ax.text(j,i,f"{cm[i,j]:,}",ha="center",va="center",fontsize=11,fontweight="bold",color="white" if cm[i,j]>cm.max()/2 else "black")
        plt.colorbar(im,ax=ax,shrink=0.8)
    plt.suptitle("Fig 4. Confusion Matrices \u2014 All Three Models\nULB Credit Card Fraud Dataset (Dal Pozzolo et al., 2015)",fontsize=11,style="italic")
    plt.tight_layout(); plt.savefig("reports/fig4_confusion_matrices.png",dpi=150,bbox_inches="tight"); plt.close(); print("  Fig 4 saved")

    # Fig 5 ROC curves
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,name in zip(axes,names):
        fpr,tpr,_=roc_curve(y_test,results[name]["ypr"]); c=CLR[name]
        ax.plot(fpr,tpr,color=c,lw=2,label=f"AUC={results[name]['auc']:.4f}"); ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.4); ax.fill_between(fpr,tpr,alpha=0.07,color=c)
        ax.legend(loc="lower right",fontsize=9); ax.set_xlabel("False Positive Rate",fontsize=9); ax.set_ylabel("True Positive Rate",fontsize=9)
        ax.set_title(f"{name}\nROC Curve",fontsize=10,fontweight="bold"); ax.set_xlim([0,1]); ax.set_ylim([0,1.02]); ax.grid(alpha=0.3)
    plt.suptitle("Fig 5. ROC Curves \u2014 All Three Models",fontsize=11,style="italic")
    plt.tight_layout(); plt.savefig("reports/fig5_roc_curves.png",dpi=150,bbox_inches="tight"); plt.close(); print("  Fig 5 saved")

    # Fig 6 Metrics bar chart
    mk=["accuracy","precision","recall","f1","auc"]; ml=["ACCURACY","PRECISION","RECALL","F1","AUC"]
    x=np.arange(len(mk)); w=0.25
    fig,ax=plt.subplots(figsize=(13,5))
    for i,name in enumerate(names):
        vals=[results[name][m] for m in mk]; bars=ax.bar(x+i*w,vals,w,label=name,color=CLR[name],alpha=0.85,edgecolor="white")
        for bar,val in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.004,f"{val:.3f}",ha="center",va="bottom",fontsize=7.5,fontweight="bold")
    ax.set_xticks(x+w); ax.set_xticklabels(ml,fontsize=10); ax.set_ylim(0,1.18); ax.set_ylabel("Score",fontsize=11)
    ax.set_title("Fig 6. Comparative Model Performance \u2014 All Five Metrics\nULB Credit Card Fraud Dataset",fontsize=11,fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y",alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); plt.savefig("reports/fig6_metrics_comparison.png",dpi=150,bbox_inches="tight"); plt.close(); print("  Fig 6 saved")

    # Fig 7 Learning curves
    sizes=np.linspace(0.1,1.0,6)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,name in zip(axes,names):
        Xu=X_svm if name=="SVM" else X_bal; yu=y_svm if name=="SVM" else y_bal; model=trained[name]["model"]
        ts,tr_sc,cv_sc=learning_curve(model,Xu,yu,cv=cv5,train_sizes=sizes,scoring="f1",n_jobs=-1)
        tm=tr_sc.mean(1); tsd=tr_sc.std(1); cm=cv_sc.mean(1); csd=cv_sc.std(1); c=CLR[name]
        ax.plot(ts,tm,"o-",color=c,lw=2,label="Training score"); ax.fill_between(ts,tm-tsd,tm+tsd,alpha=0.1,color=c)
        ax.plot(ts,cm,"s--",color="gray",lw=2,label="CV score"); ax.fill_between(ts,cm-csd,cm+csd,alpha=0.1,color="gray")
        ax.set_xlabel("Training size",fontsize=9); ax.set_ylabel("F1 Score",fontsize=9)
        ax.set_title(f"{name}\nLearning Curve",fontsize=10,fontweight="bold"); ax.legend(loc="lower right",fontsize=8); ax.set_ylim([0.5,1.05]); ax.grid(alpha=0.3)
    plt.suptitle("Fig 7. Learning Curves \u2014 All Three Models",fontsize=11,style="italic")
    plt.tight_layout(); plt.savefig("reports/fig7_learning_curves.png",dpi=150,bbox_inches="tight"); plt.close(); print("  Fig 7 saved")

    # Fig 8 Feature importance
    rf=trained["Random Forest"]["model"]; feat_cols=[f"V{i}" for i in range(1,29)]+["log_amount","hour_of_day","is_night"]
    fi=pd.Series(rf.feature_importances_,index=feat_cols); top10=fi.sort_values(ascending=False).head(10)
    fig,ax=plt.subplots(figsize=(10,5))
    bars=ax.barh(top10.index[::-1],top10.values[::-1],color="#16A34A",alpha=0.85,edgecolor="white")
    for bar,val in zip(bars,top10.values[::-1]): ax.text(val+0.0003,bar.get_y()+bar.get_height()/2,f"{val:.4f}",va="center",fontsize=9)
    ax.set_xlabel("Feature Importance (Gini)",fontsize=11); ax.set_title("Fig 8. Top 10 Feature Importances \u2014 Random Forest\nULB Credit Card Fraud Dataset",fontsize=11,fontweight="bold"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); plt.savefig("reports/fig8_feature_importance.png",dpi=150,bbox_inches="tight"); plt.close(); print("  Fig 8 saved")


def save_results(results):
    rows=[]
    for name,r in results.items():
        cm=r["cm"]
        rows.append({"Model":name,"Accuracy":round(r["accuracy"],4),"Precision":round(r["precision"],4),
            "Recall":round(r["recall"],4),"F1":round(r["f1"],4),"AUC":round(r["auc"],4),
            "CV_F1_Mean":round(r["cv_f1_mean"],4),"CV_F1_Std":round(r["cv_f1_std"],4),
            "TN":int(cm[0,0]),"FP":int(cm[0,1]),"FN":int(cm[1,0]),"TP":int(cm[1,1]),"False_Alarms":int(cm[0,1])})
    out=pd.DataFrame(rows).set_index("Model"); out.to_csv("reports/results_summary.csv")
    print(f"\n{'='*60}\n  FINAL RESULTS\n{'='*60}")
    print(out[["Accuracy","Precision","Recall","F1","AUC","CV_F1_Mean","False_Alarms"]].to_string())

    # Auto-verify against report values
    expected={"Logistic Regression":dict(Accuracy=0.9745,Precision=0.0581,Recall=0.9082,F1=0.1093,AUC=0.9719),
               "Random Forest":dict(Accuracy=0.9988,Precision=0.5931,Recall=0.8776,F1=0.7078,AUC=0.9782),
               "SVM":dict(Accuracy=0.9817,Precision=0.0780,Recall=0.8878,F1=0.1433,AUC=0.9805)}
    print(f"\n{'='*60}\n  VERIFICATION vs REPORT\n{'='*60}")
    all_ok=True
    for mn,exp in expected.items():
        for metric,ev in exp.items():
            gv=round(float(out.loc[mn,metric]),4); ok=abs(gv-ev)<0.0001
            if not ok: all_ok=False
            print(f"  {mn[:22]:<22} {metric:<10} expected={ev:.4f}  got={gv:.4f}  [{'PASS' if ok else 'FAIL'}]")
    print()
    if all_ok: print("  ALL VALUES MATCH REPORT - results are fully reproducible.")
    else: print("  Some values differ. Check dataset is the full 284,807-row ULB file.")


def main():
    parser=argparse.ArgumentParser(description="Credit Card Fraud Detection Pipeline")
    parser.add_argument("--data",default="data/creditcard.csv")
    args=parser.parse_args()
    print("\n"+"="*60+"\n  CREDIT CARD FRAUD DETECTION PIPELINE\n  Muhammad Zeeshan | B01799050 | UWS COMP11128\n  Dataset: Dal Pozzolo et al. (2015) - ULB Kaggle\n"+"="*60)
    df=load_data(args.data)
    eda_figures(df)
    df,feat_cols=engineer_features(df)
    X_bal,y_bal,X_svm,y_svm,X_test,y_test=preprocess(df,feat_cols)
    trained=train_models(X_bal,y_bal,X_svm,y_svm)
    results=evaluate(trained,X_test,y_test)
    result_figures(results,trained,y_test,X_bal,y_bal,X_svm,y_svm)
    save_results(results)
    print("\n"+"="*60+"\n  COMPLETE\n  models/  -> 3 classifiers + scaler\n  reports/ -> 8 figures + results_summary.csv\n"+"="*60+"\n")

if __name__=="__main__":
    main()
