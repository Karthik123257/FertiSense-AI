<div align="center">

# ⚙️ System Architecture & End-to-End Pipeline
### *High-Throughput Clinical PDF Ingestion, Explainable AI Inference & Dynamic Reporting Engine*

[![Backend](https://img.shields.io/badge/API-FastAPI_v0.115-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![OCR Engine](https://img.shields.io/badge/Extraction-pdfplumber_Regex-0284c7?style=for-the-badge)](https://github.com/jsvine/pdfplumber)
[![ML Framework](https://img.shields.io/badge/Inference-XGBoost_Dual_Trained-10b981?style=for-the-badge&logo=xgboost)](https://xgboost.readthedocs.io/)
[![XAI Attribution](https://img.shields.io/badge/Explainability-Tree__SHAP-8b5cf6?style=for-the-badge)](https://shap.readthedocs.io/)
[![PDF Generator](https://img.shields.io/badge/Report-ReportLab_Engine-f43f5e?style=for-the-badge)](https://www.reportlab.com/)

---

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 1. CLIENT DASHBOARD (HTML5 / Tailwind CSS / Vanilla JS)               │
  │    • Drag & Drop Patient Lab PDF (.pdf)                                │
  │    • Sex Routing Selector (Male Semen / Female Endocrine)              │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │  Multipart/Form-Data
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 2. FASTAPI REST SERVER (app.py)                                        │
  │    • Ingests PDF in-memory buffer via io.BytesIO                       │
  │    • Validates payload schemas & routes to clinical parsers            │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 3. REGEX & OCR EXTRACTION ENGINE (pdf_extractor.py)                   │
  │    • Patient Demographic Parser ("Patient Name:", "Age")               │
  │    • Context-Aware Non-Digit Delimiter Regular Expressions             │
  │    • Outputs Structured Biomarker Vector: x ∈ ℝ⁷ (Male) or ℝ⁸ (Female)  │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            │ Sex-Specific Dispatch                             │
            ▼                                                   ▼
 ┌───────────────────────────────┐       ┌───────────────────────────────┐
 │ MALE ML & XAI SUBSYSTEM       │       │ FEMALE ML & XAI SUBSYSTEM     │
 │ • Model: male_model.joblib    │       │ • Model: female_model.joblib  │
 │ • Explainer: TreeExplainer    │       │ • Explainer: TreeExplainer    │
 │ • Norms: WHO 6th Edition      │       │ • Norms: ASRM / Rotterdam     │
 └──────────────┬────────────────┘       └──────────────┬────────────────┘
                │                                       │
                └───────────────────┬───────────────────┘
                                    │
                                    ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 4. PLAIN-ENGLISH CLINICAL TRANSLATION ENGINE                           │
  │    • Negative SHAP (φᵢ < 0) ──► "What Looks Great" (Strengths)         │
  │    • Positive SHAP (φᵢ > 0) ──► "Areas Needing Attention" (Blockers)   │
  │    • Generates 74-day spermatogenesis or ovulatory recovery roadmaps   │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
 ┌───────────────────────────────┐       ┌───────────────────────────────┐
 │ LIVE DASHBOARD UPDATE         │       │ DOWNLOADABLE CLINICAL REPORT  │
 │ • Conception Readiness Gauge  │       │ • report_generator.py         │
 │ • Real-Time Input Auto-Fill   │       │ • Styled ReportLab PDF        │
 │ • Protective / Risk Cards     │       │ • Prominent Final Verdict Box │
 └───────────────────────────────┘       └───────────────────────────────┘
```

</div>

---

## 🧩 2. Core Architecture Pipeline Stages

### Stage 1: Context-Aware PDF Parsing & OCR (`pdf_extractor.py`)
Standard lab reports format biomarkers across inconsistent layouts (e.g., `AMH: 3.4`, `AMH (ng/mL) = 3.4`, or `Anti-Mullerian Hormone - 3.4`). The parser employs **boundary-resilient, non-digit lookahead patterns**:
* **Name Regex:** Matches patient labels across multi-line breaks without swallowing adjacent fields like `Age`, `Sex`, or `Date`.
* **Biomarker Regex:** Uses `[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)` to isolate numbers regardless of intervening unit symbols.

### Stage 2: Dual-Stream Machine Learning Inference
FertilAI routes requests into dedicated classification paths to avoid confounding biological signals between male spermatogenesis and female ovulatory endocrinology:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ MALE: x = [Vol, Conc, ProgMot, TotMot, Morph, Testo, SittingHours]      │
│ ↳ Prediction: P(Subfertile | x) via 120-tree XGBoost Ensemble           │
├─────────────────────────────────────────────────────────────────────────┤
│ FEMALE: x = [AMH, TSH, FSH, LH, PRL, AFC_Total, BMI, CycleLength]       │
│ ↳ Prediction: P(Ovulatory Delay | x) via 120-tree XGBoost Ensemble      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Stage 3: Feature Attribution to Clinical Translation (`app.py`)
Raw model probabilities cannot tell a patient or clinician *why* a particular prediction was generated. FertilAI computes exact **SHapley Additive exPlanations (SHAP)** values for each feature:
$$\hat{z}(x) = \phi_0 + \sum_{i=1}^{D} \phi_i(x)$$
* **$\phi_i \le 0$ (Protective Driver):** Biomarker sits comfortably within reference bounds. Mapped directly into patient-accessible terminology under **"What Looks Great"** (e.g., *"Healthy Egg Reserve: AMH of 3.4 ng/mL shows optimal primordial follicular pool"*).
* **$\phi_i > 0$ (Pathological Driver):** Biomarker deviates from guidelines. Mapped directly into root-cause rationales under **"Areas Needing Attention"** (e.g., *"Elevated Thyroid: TSH of 4.8 mIU/L is outside preconception window, creating ovulatory delay"*).

### Stage 4: Dynamic Clinical PDF Export (`report_generator.py`)
Compiles a finalized clinical evaluation document featuring:
1. **Patient Demographic Header & Date Stamps**
2. **Tabular Biomarker Audit:** Side-by-side comparison of measured values against WHO 6th Ed. and ASRM reference thresholds.
3. **Dual-Track Takeaways:** Plain-English strengths and root causes.
4. **Actionable Care Roadmap:** Recommended lifestyle adjustments, dietary supplements, and specialist follow-up intervals.
5. **Final Verdict Banner:** Highlighted green (`FINAL AI PREDICTION RESULT: FERTILE`) or red/crimson (`FINAL AI PREDICTION RESULT: INFERTILE`) summary box.

---

## 📡 3. REST API Endpoint Specification

| Method | Endpoint | Description | Payload | Return Type |
| :---: | :--- | :--- | :--- | :--- |
| `GET` | `/` | Health check & model registry status | *None* | `JSON` |
| `POST` | `/analyze-report` | Full pipeline: Ingests PDF, parses values, runs inference & SHAP | `MultipartForm` (`file`, `gender`, `patient_name`) | `JSON` |
| `POST` | `/download-pdf-report` | Generates & streams branded clinical takeaway PDF | `FormData` (scores, strengths, blockers, metrics) | `application/pdf` |
| `GET` | `/datasets/{type}/preview` | Returns schema, column list, and top 5 rows for web explorer | URL Path (`male` or `female`) | `JSON` |
| `GET` | `/datasets/{type}/download` | Direct 1-click download of training dataset CSVs | URL Path (`male` or `female`) | `text/csv` |

---

## 🚀 4. Local Execution & Startup Guide

```bash
# 1. Install Dependencies
pip install fastapi uvicorn python-multipart pdfplumber reportlab xgboost shap scikit-learn pandas numpy joblib

# 2. Train Models & Build SHAP Explainers
python train_models.py

# 3. Launch FastAPI Server
python -m uvicorn app:app --reload --port 8000

# 4. Open Frontend
# Open frontend/index.html in your web browser (or via VS Code Live Server)
```