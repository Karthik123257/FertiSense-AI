# 🧬 FertilAI — Explainable AI Reproductive Diagnostic Platform
> **End-to-End Machine Learning Pipeline & Clinical Decision Support System for Explainable Infertility Triage**

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI_v0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/ML_Engine-XGBoost_v2.0-FF6F00?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP_TreeExplainer-38BDF8?style=for-the-badge)](https://shap.readthedocs.io/)
[![Clinical Standards](https://img.shields.io/badge/Benchmarks-WHO_6th_Ed_%26_ASRM-10B981?style=for-the-badge)](#clinical-guidelines--reference-standards)
[![Frontend](https://img.shields.io/badge/UI-Tailwind_CSS_%2B_Glassmorphism-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)](#frontend-features)

---

## 📌 Executive Summary
**FertilAI** is an explainable AI reproductive health platform designed to transform complex laboratory reports into clear, patient-friendly insights. By taking unstructured clinical lab reports (PDFs), the platform:
1. **Parses & Auto-Fills Patient Data:** Automatically extracts clinical biomarkers and demographic details using `pdfplumber` and medical regex patterns without manual data entry.
2. **Evaluates Risk via Calibrated ML:** Employs sex-specific gradient-boosted decision trees trained on clinical reproductive cohorts.
3. **Provides Local Explainability (SHAP):** Decomposes black-box predictions into positive baseline strengths ("What Looks Great") and actionable root causes ("Areas Needing Attention").
4. **Generates Downloadable Clinical Reports:** Compiles structured diagnostic summaries with explicit **Fertile vs. Infertile (Subfertile)** decision verdicts via ReportLab.

---

## 🏗️ System Architecture & Workflow

```text
       [Patient Uploads Semen Analysis or Hormone Lab PDF]
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │ 1. PDF Parser & Regex Extractor              │
        │    (pdf_extractor.py / pdfplumber)           │
        └──────────────────────┬───────────────────────┘
                               │ Extracted Values & Patient Name
                               ▼
            [Sex-Specific Clinical Routing]
             /                            \
            ▼                              ▼
  ┌────────────────────────┐      ┌────────────────────────┐
  │ Male Diagnostics       │      │ Female Diagnostics     │
  │ • WHO 6th Ed. Metrics  │      │ • ASRM & Rotterdam     │
  │ • Model: XGBoost Male  │      │ • Model: XGBoost Female│
  │ • Explainer: SHAP Tree │      │ • Explainer: SHAP Tree │
  └───────────┬────────────┘      └───────────┬────────────┘
              \                            /
               ▼                          ▼
        ┌──────────────────────────────────────────────┐
        │ 2. Plain-English Clinical Attribution Engine │
        │    • Positive SHAP → "What Looks Great"      │
        │    • Negative SHAP → "Areas Needing Attention│
        │    • Clinical Roadmap Recommendations        │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │ 3. Dual Delivery Presentation                │
        │    ├── Interactive Web Dashboard UI          │
        │    └── Downloadable Clinical PDF Report      │
        │        (With Plain-English Verdict Box)      │
        └──────────────────────────────────────────────┘


🔬 Clinical Guidelines & Reference Standards1. Male Diagnostics (WHO 6th Edition Standard)Calibrated on the multi-parametric SEED-ML cohort (infertility_man_data-v2.csv):Progressive Motility (PR): $\ge 30\%$ (Forward linear swimming speed towards ovum)Sperm Concentration: $\ge 16.0\text{ M/mL}$ (Viable cell density minimum)Total Motility (PR + NP): $\ge 42\%$ (Combined moving sperm threshold)Normal Morphology: $\ge 4.0\%$ (Kruger strict criteria for ovum penetration)Semen Volume: $\ge 1.4\text{ mL}$ (Accessory gland fluid sufficiency)Total Testosterone: $300\text{--}850\text{ ng/dL}$ (Primary endocrine driver)Sedentary Risk Indicator: $< 8.0\text{ hrs/day}$ (Avoidance of scrotal hyperthermia)2. Female Diagnostics (ASRM & Rotterdam Consensus)Calibrated on clinical patient records across 10 centers (PCOS_infertility.csv):Anti-Müllerian Hormone (AMH): $1.2\text{--}4.0\text{ ng/mL}$ (Primordial follicle egg reserve)Preconception TSH: $0.5\text{--}2.5\text{ mIU/L}$ (Thyroid window for ovulation & implantation)Prolactin (PRL): $< 25.0\text{ ng/mL}$ (Avoidance of luteinizing hormone suppression)Antral Follicle Count (AFC): $10\text{--}20\text{ follicles}$ (Sonographic bilateral total)Menstrual Cycle Regularity: $26\text{--}32\text{ days}$ (Ovulatory rhythm indicator)LH/FSH Ratio: $< 2.0$ (Screening marker for anovulatory PCOS)📊 Model Performance & Evaluation HighlightsModel PipelineROC-AUCOverall AccuracyPrimary PredictorsMale XGBoost Classifier0.80073.1%Progressive Motility (43.4%), Normal Morphology (16.0%), Sperm Concentration (10.7%)Female XGBoost Classifier0.75568.0%Prolactin (20.5%), Thyroid TSH (18.8%), Menstrual Cycle Length (14.2%)Detailed confusion matrices, precision-recall breakdowns, bar charts, and diagnostic pie charts are available inside docs_and_results/02_Evaluation_and_Visual_Results/.📁 Repository Directory StructurePlaintextFertilAI/
├── fertilai-backend/
│   ├── data/
│   │   ├── infertility_man_data-v2.csv         # Male training dataset (SEED-ML)
│   │   └── PCOS_infertility.csv                # Female training dataset (Kaggle)
│   ├── models/
│   │   ├── male_model.joblib                   # Serialized male XGBoost model
│   │   ├── male_explainer.joblib               # SHAP TreeExplainer for male pipeline
│   │   ├── female_model.joblib                 # Serialized female XGBoost model
│   │   └── female_explainer.joblib             # SHAP TreeExplainer for female pipeline
│   ├── app.py                                  # FastAPI REST server & inference router
│   ├── pdf_extractor.py                        # pdfplumber OCR & clinical regex parser
│   ├── report_generator.py                     # ReportLab PDF engine with verdict box
│   └── requirements.txt                        # Backend Python dependencies
│
├── frontend/
│   └── index.html                              # Tailwind CSS interactive dashboard
│
├── docs_and_results/
│   ├── 01_Prediction_Methodology/
│   │   ├── Male_Prediction/README.md           # Formulas (TMSC, tree ensemble, SHAP)
│   │   └── Female_Prediction/README.md         # Formulas (LH/FSH, BMI, endocrine thresholds)
│   ├── 02_Evaluation_and_Visual_Results/
│   │   ├── Male_Results/charts/                # Confusion matrix, ROC, pie & bar charts
│   │   └── Female_Results/charts/              # Confusion matrix, ROC, pie & bar charts
│   └── 03_System_Architecture_and_Pipeline/
│       └── README.md                           # Comprehensive end-to-end pipeline specs
│
├── .gitignore                                  # Git exclusion configuration
└── README.md                                   # Root project documentation
🚀 Quickstart & Installation GuidePrerequisitesPython 3.9+ installedModern Web Browser (Chrome, Edge, Firefox)1. Clone the RepositoryBashgit clone [https://github.com/](https://github.com/)<your-username>/FertilAI.git
cd FertilAI
2. Set Up the Backend EnvironmentBashcd fertilai-backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
3. Launch the Backend ServerBashpython -m uvicorn app:app --reload --port 8000
API will be live at: http://127.0.0.1:8000Swagger Documentation: http://127.0.0.1:8000/docs4. Open the Web DashboardOpen frontend/index.html in your browser (or use VS Code Live Server on port 5500).Verify that the bottom-left status pill turns green and shows Backend Online.🔒 Medical Device DisclaimerFertilAI is an automated assistive triage tool designed for educational, research, and preparatory clinical screening. It does not replace definitive medical judgment or diagnostic evaluation by a licensed urologist, reproductive endocrinologist, or obstetrician.
