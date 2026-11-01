<div align="center">

# 🌸 Female Infertility & Ovulatory AI Engine
### *Explainable Endocrine, Ovarian Reserve & Ultrasound Follicle Evaluation*

[![Standard](https://img.shields.io/badge/Clinical_Standard-ASRM_%26_Rotterdam-ec4899?style=for-the-badge&logo=medscape)](https://www.asrm.org/)
[![Model](https://img.shields.io/badge/Classifier-XGBoost_Ensemble-10b981?style=for-the-badge&logo=xgboost)](https://xgboost.readthedocs.io/)
[![Dataset](https://img.shields.io/badge/Cohort-Kaggle_PCOS_541_Patients-0284c7?style=for-the-badge)](https://www.kaggle.com/)
[![ROC-AUC](https://img.shields.io/badge/Test_ROC--AUC-0.733-f59e0b?style=for-the-badge)]()

---

```
   ┌─────────────────────────────────────────────────────────┐
   │       DAY-3 BLOODWORK & ULTRASOUND FOLICULAR SCAN       │
   └────────────────────────────┬────────────────────────────┘
                                │
                                ▼  [Automated Parameter Extraction]
   ┌─────────────────────────────────────────────────────────┐
   │ PARSED BIOMARKERS: AMH, TSH, FSH, LH, PRL, AFC, Cycle   │
   └────────────────────────────┬────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
     ┌───────────────────────┐       ┌──────────────────────┐
     │  Endocrine Predictor  │       │  SHAP Explainer      │
     │  P(Ovulatory Delay)   │       │  Biomarker Weights   │
     └──────────┬────────────┘       └──────────┬───────────┘
                │                               │
                └───────────────┬───────────────┘
                                ▼
   ┌─────────────────────────────────────────────────────────┐
   │             EXPLAINABLE CLINICAL ROADMAP                │
   │  🟢 Favorable Reserve Indicators (AMH, AFC)             │
   │  🔴 Treatable Endocrine Signals (Thyroid, Prolactin)    │
   └─────────────────────────────────────────────────────────┘
```

</div>

---

## 🧬 1. Monitored Endocrine & Ovarian Biomarkers

Validated against guidelines from the **American Society for Reproductive Medicine (ASRM)** and **ESHRE Rotterdam Criteria**:

| Biomarker | Clinical Unit | Fertility Target Range | Physiological Mechanism |
| :--- | :---: | :---: | :--- |
| **AMH (Anti-Müllerian)** | $\text{ng/mL}$ | **$1.2 - 4.0$ ng/mL** | Secreted by pre-antral follicles; primary quantitative proxy for egg reserve |
| **TSH (Preconception)** | $\text{mIU/L}$ | **$0.5 - 2.5$ mIU/L** | Tighter preconception window to ensure timely ovulation and early implantation |
| **FSH (Day 3)** | $\text{mIU/mL}$ | **$3.5 - 10.0$ mIU/mL** | Pituitary signaling required to recruit and grow the month's follicle cohort |
| **LH (Day 3)** | $\text{mIU/mL}$ | **$2.5 - 8.0$ mIU/mL** | Triggers final oocyte maturation; normally synchronized $\approx 1:1$ with FSH |
| **Prolactin (PRL)** | $\text{ng/mL}$ | **$< 24.0$ ng/mL** | Pituitary hormone; elevations directly disrupt pulsatile GnRH secretion |
| **Antral Follicles (AFC)** | $\text{count}$ | **$10 - 22$ Follicles** | Total resting follicles visible on Day-3 transvaginal ultrasound (both ovaries) |
| **Body Mass Index (BMI)** | $\text{kg/m}^2$ | **$18.5 - 24.9$** | Evaluates peripheral estrogen aromatization and insulin resistance risks |
| **Menstrual Cycle Length** | $\text{days}$ | **$26 - 32$ Days** | Regularity indicator; cycles $>35$ days strongly signal anovulation |

---

## 📐 2. Key Diagnostic Formulas Applied

```
 1. GONADOTROPIN SYNCHRONIZATION RATIO
    R(LH/FSH) = LH (mIU/mL) / FSH (mIU/mL)
    ↳ If Ratio > 2.0 : Strong clinical indicator of anovulatory PCOS follicle arrest.

 2. ANTRAL FOLLICULAR TOTAL
    AFC_Total = Follicles(Left Ovary) + Follicles(Right Ovary)
    ↳ High AFC (> 22) + High AMH (> 4.5) ──► Polycystic Ovarian Pattern
    ↳ Low AFC (< 6)   + Low AMH (< 1.0)  ──► Diminished Ovarian Reserve (DOR)

 3. THYROID FERTILITY PENALTY
    Δ_TSH = max(0, TSH - 2.5 mIU/L)
    ↳ Subclinical hypothyroidism directly suppresses peak mid-cycle LH surge.

 4. LOCAL SHAP EXPLAINABILITY MAPPING
    ẑ(x) = φ₀ + ∑ φᵢ(x)
    ↳ Generates transparent patient-friendly root causes (e.g., TSH leveling pathway).
```

---

## 📊 3. Visual Performance Snapshot

<div align="center">

| Confusion Matrix ($68.0\%$ Accuracy) | Relative Feature Importance |
| :---: | :---: |
| <img src="../../02_Evaluation_and_Visual_Results/Female_Results/charts/01_female_confusion_matrix.png" width="380" /> | <img src="../../02_Evaluation_and_Visual_Results/Female_Results/charts/02_female_feature_importance_bar.png" width="380" /> |

| Endocrine Cohort Breakdown | ROC Curve ($\text{AUC} = 0.755$) |
| :---: | :---: |
| <img src="../../02_Evaluation_and_Visual_Results/Female_Results/charts/03_female_diagnosis_pie_chart.png" width="380" /> | <img src="../../02_Evaluation_and_Visual_Results/Female_Results/charts/04_female_roc_and_confidence.png" width="380" /> |

</div>

---

## 👩‍⚕️ 4. Clinical Diagnostic Presets

* **Preset 1 — Optimal Ovulatory Baseline:**
  $\text{AMH} = 2.8\text{ ng/mL}$, $\text{TSH} = 1.85\text{ mIU/L}$, $\text{PRL} = 14\text{ ng/mL}$, $\text{AFC} = 16$ $\rightarrow$ **$89\%$ Readiness**.
* **Preset 2 — Thyroid & Ovulatory Synchronization Delay:**
  $\text{AMH} = 3.4\text{ ng/mL}$, $\text{TSH} = 4.8\text{ mIU/L}$, $\text{PRL} = 29\text{ ng/mL}$, $\text{AFC} = 16$ $\rightarrow$ **$61\%$ Readiness** (Highlights strong egg count alongside treatable thyroid elevation).
* **Preset 3 — Diminished Ovarian Reserve (DOR):**
  $\text{AMH} = 0.75\text{ ng/mL}$, $\text{FSH} = 12.4\text{ mIU/mL}$, $\text{AFC} = 5$ $\rightarrow$ **$28\%$ Readiness** (Recommends urgent reproductive timeline consultation).