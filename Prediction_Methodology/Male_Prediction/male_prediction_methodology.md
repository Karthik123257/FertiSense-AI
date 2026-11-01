<div align="center">

# 🔬 Male Infertility AI Diagnostic Engine
### *Explainable Machine Learning for Andrology & Semen Quality Evaluation*

[![Standard](https://img.shields.io/badge/Clinical_Standard-WHO_6th_Edition-06b6d4?style=for-the-badge&logo=medscape)](https://www.who.int/)
[![Model](https://img.shields.io/badge/Classifier-XGBoost_Ensemble-10b981?style=for-the-badge&logo=xgboost)](https://xgboost.readthedocs.io/)
[![Explainability](https://img.shields.io/badge/XAI_Engine-Tree__SHAP-8b5cf6?style=for-the-badge)](https://shap.readthedocs.io/)
[![ROC-AUC](https://img.shields.io/badge/Test_ROC--AUC-0.805-f59e0b?style=for-the-badge)]()

---

```
   ┌─────────────────────────────────────────────────────────┐
   │          PATIENT LAB REPORT (.PDF / SCAN)               │
   └────────────────────────────┬────────────────────────────┘
                                │
                                ▼  [pdfplumber Regex Parser]
   ┌─────────────────────────────────────────────────────────┐
   │ EXTRACTED WHO BIOMARKERS: Count, Motility, Morph, Vol.  │
   └────────────────────────────┬────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
     ┌───────────────────────┐       ┌──────────────────────┐
     │  XGBoost Risk Scorer  │       │  SHAP TreeExplainer  │
     │  P(Subfertile | x)    │       │  Local Attributions  │
     └──────────┬────────────┘       └──────────┬───────────┘
                │                               │
                └───────────────┬───────────────┘
                                ▼
   ┌─────────────────────────────────────────────────────────┐
   │         DUAL-TRACK EXPLAINABLE TAKEAWAY REPORT          │
   │  🟢 "What Looks Great"     🔴 "Areas Needing Care"      │
   │  (Protective Baselines)    (Root Causes & Lifestyle)   │
   └─────────────────────────────────────────────────────────┘
```

</div>

---

## 📊 1. Core Clinical Biomarkers Monitored

Our XGBoost classifier evaluates patient parameters against the **World Health Organization (WHO) 6th Edition reference bounds**:

| Biomarker | Clinical Unit | WHO 6th Ed. Baseline | Diagnostic Significance |
| :--- | :---: | :---: | :--- |
| **Sperm Concentration** | $10^6\text{ / mL}$ | **$\ge 16.0$ M/mL** | Density of gametes produced per milliliter of ejaculate |
| **Progressive Motility** | $\%$ | **$\ge 30.0\%$** | Percentage of sperm actively swimming in rapid, linear paths |
| **Total Motility** | $\%$ | **$\ge 42.0\%$** | Combined proportion of progressive and non-progressive swimmers |
| **Normal Morphology** | $\%$ | **$\ge 4.0\%$** | Strict Kruger criteria assessment of acrosome, head, and tail shapes |
| **Ejaculate Volume** | $\text{mL}$ | **$\ge 1.4$ mL** | Accessory gland secretory contribution (seminal vesicles & prostate) |
| **Total Testosterone** | $\text{ng/dL}$ | **$300 - 850$ ng/dL** | Endocrine fuel driving active spermatogenesis in Leydig cells |
| **Daily Sitting Duration** | $\text{hrs/day}$ | **$< 8.0$ hrs/day** | Behavioral risk factor for scrotal hyperthermia and DNA fragmentation |

---

## 🧮 2. Diagnostic Calculations & Formulas

```
 1. TOTAL MOTILE SPERM COUNT (TMSC)
    TMSC = Volume (mL) × Concentration (M/mL) × (Progressive Motility % / 100)
    ↳ Clinical Goal: ≥ 20 Million indicates optimal natural conception potential.

 2. MODEL ENSEMBLE LOG-ODDS
    ẑ(x) = z₀ + ∑ [ η × fₖ(x) ]   (for k = 1 to 120 trees, learning rate η = 0.04)

 3. PROBABILITY CALIBRATION
    P(Subfertile | x) = 1 / [ 1 + e^(-ẑ(x)) ]
    Conception Readiness Score = [ 1 - P(Subfertile | x) ] × 100%

 4. SHAP DUAL-COLUMN BREAKDOWN
    ẑ(x) = φ₀ + ∑ φᵢ(x)
    ↳ If φᵢ < 0 : Protective Strength  ──► Categorized as "What Looks Great"
    ↳ If φᵢ > 0 : Risk / Blocker       ──► Categorized as "Areas Needing Attention"
```

---

## 📈 3. Visual Performance Snapshot

<div align="center">

| Confusion Matrix ($73.1\%$ Accuracy) | Relative Feature Importance |
| :---: | :---: |
| <img src="../../02_Evaluation_and_Visual_Results/Male_Results/charts/01_male_confusion_matrix.png" width="380" /> | <img src="../../02_Evaluation_and_Visual_Results/Male_Results/charts/02_male_feature_importance_bar.png" width="380" /> |

| Diagnostic Cohort Distribution | ROC Curve ($\text{AUC} = 0.800$) |
| :---: | :---: |
| <img src="../../02_Evaluation_and_Visual_Results/Male_Results/charts/03_male_diagnosis_pie_chart.png" width="380" /> | <img src="../../02_Evaluation_and_Visual_Results/Male_Results/charts/04_male_roc_and_confidence.png" width="380" /> |

</div>

---

## 💡 4. Real-World Diagnostic Presets

* **Case A — Normozoospermia (Optimal):**
  $\text{Volume} = 3.5\text{ mL}$, $\text{Count} = 55\text{ M/mL}$, $\text{Motility} = 45\%$, $\text{Morph} = 6.5\%$ $\rightarrow$ **$88\%$ Readiness**.
* **Case B — Asthenozoospermia (Functional Motility Delay):**
  $\text{Volume} = 3.2\text{ mL}$, $\text{Count} = 48\text{ M/mL}$, $\text{Motility} = 18\%$, $\text{Morph} = 6.0\%$ $\rightarrow$ **$64\%$ Readiness** (Identifies treatable antioxidant & thermal triggers).
* **Case C — Oligozoospermia (Low Density):**
  $\text{Volume} = 1.8\text{ mL}$, $\text{Count} = 8.5\text{ M/mL}$, $\text{Motility} = 15\%$, $\text{Morph} = 2.0\%$ $\rightarrow$ **$22\%$ Readiness** (Recommends varicocele and hormonal screening).