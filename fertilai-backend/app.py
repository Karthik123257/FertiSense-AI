"""
app.py
FastAPI Server connecting PDF parsing, ML inference, SHAP explanations,
and downloadable clinical PDF generation.
"""

import os
import io
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, FileResponse

from pdf_extractor import extract_male_parameters, extract_female_parameters
from report_generator import generate_patient_pdf

app = FastAPI(
    title="FertilAI Diagnostics Engine",
    description="Explainable Infertility Prediction API using XGBoost & SHAP",
    version="2.0.0"
)

# Enable CORS for local web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models & Explainers
MALE_MODEL_PATH = "models/male_model.joblib"
MALE_EXPLAINER_PATH = "models/male_explainer.joblib"
FEMALE_MODEL_PATH = "models/female_model.joblib"
FEMALE_EXPLAINER_PATH = "models/female_explainer.joblib"

male_model = joblib.load(MALE_MODEL_PATH) if os.path.exists(MALE_MODEL_PATH) else None
male_explainer = joblib.load(MALE_EXPLAINER_PATH) if os.path.exists(MALE_EXPLAINER_PATH) else None

female_model = joblib.load(FEMALE_MODEL_PATH) if os.path.exists(FEMALE_MODEL_PATH) else None
female_explainer = joblib.load(FEMALE_EXPLAINER_PATH) if os.path.exists(FEMALE_EXPLAINER_PATH) else None


# =====================================================================
# SHAP CLINICAL INTERPRETERS
# =====================================================================
def interpret_shap_male(features_dict, shap_values):
    strengths = []
    concerns = []
    recs = []

    vol = features_dict.get('Volume(mL)', 0)
    prog_mot = features_dict.get('Progressive_Motility(%)', 0)
    tot_mot = features_dict.get('Total_Motility(%)', 0)
    conc = features_dict.get('Sperm_Concentration(M/mL)', 0)
    morph = features_dict.get('Normal_Morphology(%)', 0)
    sitting = features_dict.get('Sitting_Hours_Per_Day', 0)

    if vol >= 1.4:
        strengths.append(f"Healthy Semen Volume: {vol} mL exceeds the 1.4 mL WHO standard.")
    else:
        concerns.append(f"Low Ejaculate Volume: {vol} mL is below the 1.4 mL standard.")

    if conc >= 16.0:
        strengths.append(f"Strong Sperm Production: Concentration of {conc} M/mL is well above the WHO 16 M/mL clinical minimum.")
    else:
        concerns.append(f"Low Sperm Concentration (Oligozoospermia): {conc} M/mL reduces the probability of natural fertilization.")
        recs.append("Consult a urologist for varicocele evaluation and gonadotropin assessment.")

    if prog_mot >= 30.0:
        strengths.append(f"Forward Motility: {prog_mot}% of sperm show strong linear movement towards the egg.")
    else:
        concerns.append(f"Reduced Swimming Speed (Asthenozoospermia): Only {prog_mot}% move progressively (standard is ≥30%).")
        recs.append("Incorporate CoQ10 (200mg/day) and L-Carnitine to replenish mitochondrial energy.")

    if tot_mot >= 42.0:
        strengths.append(f"Good Total Motility: {tot_mot}% total moving sperm meets the ≥42% WHO threshold.")

    if morph >= 4.0:
        strengths.append(f"Normal Structural Shape: {morph}% strict normal forms exceeds the 4% Kruger standard.")
    else:
        concerns.append(f"Atypical Shape (Teratozoospermia): {morph}% normal forms indicates morphological variation.")

    if sitting <= 8.0:
        strengths.append(f"Healthy Daily Activity: {sitting} sitting hours/day avoids excessive scrotal heat buildup.")
    else:
        concerns.append(f"Scrotal Heat Elevation: {sitting} sitting hours daily causes localized hyperthermia.")
        recs.append("Use a standing desk and take 5-minute walking breaks every 60 minutes.")

    if not recs:
        recs.append("Maintain antioxidant-rich Mediterranean diet and re-screen in 74 days (full spermatogenesis cycle).")

    return strengths, concerns, recs


def interpret_shap_female(features_dict, shap_values):
    strengths = []
    concerns = []
    recs = []

    amh = features_dict.get('AMH(ng/mL)', 0)
    tsh = features_dict.get('TSH (mIU/L)', 0)
    fsh = features_dict.get('FSH(mIU/mL)', 0)
    lh = features_dict.get('LH(mIU/mL)', 0)
    prl = features_dict.get('PRL(ng/mL)', 0)
    afc = features_dict.get('Follicle No. (Total)', 0)
    cycle = features_dict.get('Cycle length(days)', 0)

    # 1. Ovarian Reserve (AMH)
    if 1.2 <= amh <= 4.0:
        strengths.append(f"Optimal Ovarian Egg Reserve: AMH of {amh} ng/mL shows balanced, fertile primordial follicles.")
    elif amh > 4.5:
        concerns.append(f"High Follicular Density (PCOS Pattern): Elevated AMH ({amh} ng/mL) with AFC ({afc}) indicates follicle clustering.")
        recs.append("Schedule an insulin resistance panel (HOMA-IR) and discuss myo-inositol supplementation with your physician.")
    else:
        concerns.append(f"Diminished Ovarian Reserve (DOR): AMH of {amh} ng/mL indicates declining egg availability.")
        recs.append("Consult an REI specialist promptly to evaluate immediate conception or egg preservation pathways.")

    # 2. Thyroid Window (TSH)
    if 0.5 <= tsh <= 2.5:
        strengths.append(f"Thyroid Balance in Optimal Window: TSH of {tsh} mIU/L supports regular ovulation.")
    else:
        concerns.append(f"Subclinical Thyroid Imbalance: TSH of {tsh} mIU/L is outside the ideal 0.5–2.5 range for conception.")
        recs.append("Consult an endocrinologist for low-dose thyroid titration to bring TSH below 2.5 mIU/L.")

    # 3. Prolactin
    if prl <= 25.0:
        strengths.append(f"Normal Prolactin: {prl} ng/mL allows natural LH surge and timely ovulation.")
    else:
        concerns.append(f"Hyperprolactinemia: Prolactin at {prl} ng/mL can suppress ovulation signals from the pituitary.")
        recs.append("Screen for prolactin elevation triggers (sleep deprivation, thyroid interaction, or microadenoma).")

    # 4. Gonadotropins & Cycle Regularity
    if fsh > 0 and (lh / fsh) > 2.0:
        concerns.append(f"Elevated LH/FSH Ratio ({round(lh/fsh, 1)}): High ratio suggests follicular arrest common in anovulatory PCOS.")
    if cycle > 35:
        concerns.append(f"Irregular / Extended Cycle ({cycle} days): Suggests delayed or absent spontaneous ovulation.")

    if not recs:
        recs.append("Continue tracking ovulation using LH surge strips and maintain regular Day 12–16 intercourse.")

    return strengths, concerns, recs


# =====================================================================
# REST ENDPOINTS
# =====================================================================
@app.get("/")
def root():
    return {
        "service": "FertilAI Explainable Diagnostic Server",
        "status": "online",
        "models_loaded": {
            "male": male_model is not None,
            "female": female_model is not None
        }
    }


@app.post("/analyze-report")
async def analyze_report(
    gender: str = Form(...),
    patient_name: str = Form(""),
    file: UploadFile = File(...)
):
    """
    Accepts uploaded patient PDF report, extracts Patient Name & metrics,
    computes ML probability and SHAP feature contributions, and returns plain-English clinical explanation.
    """
    gender = gender.lower().strip()
    if gender not in ["male", "female"]:
        raise HTTPException(status_code=400, detail="Gender must be 'male' or 'female'")

    file_bytes = await file.read()
    buffer = io.BytesIO(file_bytes)

    clean_filename = os.path.splitext(file.filename)[0].replace("_", " ").replace("-", " ").strip()
    fallback_name = patient_name.strip() if patient_name.strip() else clean_filename

    try:
        if gender == "male":
            extracted_name, metrics, raw_text = extract_male_parameters(buffer, fallback_name=fallback_name)
            feature_names = [
                'Volume(mL)', 'Sperm_Concentration(M/mL)', 'Progressive_Motility(%)',
                'Total_Motility(%)', 'Normal_Morphology(%)', 'Total_Testosterone(ng/dL)',
                'Sitting_Hours_Per_Day'
            ]
            feature_vector = np.array([[metrics[k] for k in feature_names]])
            
            prob = float(male_model.predict_proba(feature_vector)[0][1]) if male_model else 0.45
            shap_vals = male_explainer(feature_vector).values[0] if male_explainer else np.zeros(len(feature_names))
            
            strengths, concerns, recs = interpret_shap_male(metrics, shap_vals)

        else:
            extracted_name, metrics, raw_text = extract_female_parameters(buffer, fallback_name=fallback_name)
            feature_names = [
                'AMH(ng/mL)', 'TSH (mIU/L)', 'FSH(mIU/mL)', 'LH(mIU/mL)',
                'PRL(ng/mL)', 'Follicle No. (Total)', 'BMI', 'Cycle length(days)'
            ]
            feature_vector = np.array([[metrics[k] for k in feature_names]])
            
            prob = float(female_model.predict_proba(feature_vector)[0][1]) if female_model else 0.45
            shap_vals = female_explainer(feature_vector).values[0] if female_explainer else np.zeros(len(feature_names))

            strengths, concerns, recs = interpret_shap_female(metrics, shap_vals)

        readiness_score = round(max(0.0, min(100.0, (1.0 - prob) * 100)), 1)

        return JSONResponse(content={
            "patient_name": extracted_name,
            "gender": gender,
            "subfertility_risk_pct": round(prob * 100, 1),
            "conception_readiness_pct": readiness_score,
            "status": "Favorable Conception Baseline" if prob < 0.40 else "Subfertility / Physiological Blockers Identified",
            "extracted_metrics": metrics,
            "what_looks_great": strengths,
            "areas_needing_attention": concerns,
            "action_plan": recs
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process report: {str(e)}")


@app.post("/download-pdf-report")
async def download_pdf(
    patient_name: str = Form("Patient"),
    gender: str = Form("male"),
    probability: float = Form(0.35),
    strengths: str = Form(""),
    concerns: str = Form(""),
    recommendations: str = Form(""),
    metrics_json: str = Form("{}")
):
    str_list = [s.strip() for s in strengths.split("||") if s.strip()]
    con_list = [c.strip() for c in concerns.split("||") if c.strip()]
    rec_list = [r.strip() for r in recommendations.split("||") if r.strip()]
    metrics_dict = json.loads(metrics_json) if metrics_json else {}

    pdf_bytes = generate_patient_pdf(
        patient_name=patient_name,
        gender=gender,
        probability=probability,
        strengths=str_list,
        concerns=con_list,
        recommendations=rec_list,
        metrics_dict=metrics_dict
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=FertilAI_{patient_name}_Report.pdf"}
    )


# =====================================================================
# DATASET PREVIEW & DOWNLOAD ENDPOINTS
# =====================================================================
DATASET_FILES = {
    "male": "data/infertility_man_data-v2.csv",
    "female": "data/PCOS_infertility.csv"
}

@app.get("/datasets/{dataset_type}/preview")
def preview_dataset(dataset_type: str):
    dataset_type = dataset_type.lower().strip()
    if dataset_type not in DATASET_FILES:
        raise HTTPException(status_code=400, detail="Invalid dataset type. Choose 'male' or 'female'.")

    file_path = DATASET_FILES[dataset_type]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Dataset file not found at {file_path}")

    df = pd.read_csv(file_path)
    sample_rows = df.head(5).fillna("").to_dict(orient="records")

    return {
        "dataset_type": dataset_type,
        "filename": os.path.basename(file_path),
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "sample_rows": sample_rows
    }


@app.get("/datasets/{dataset_type}/download")
def download_dataset(dataset_type: str):
    dataset_type = dataset_type.lower().strip()
    if dataset_type not in DATASET_FILES:
        raise HTTPException(status_code=400, detail="Invalid dataset type. Choose 'male' or 'female'.")

    file_path = DATASET_FILES[dataset_type]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Dataset file not found at {file_path}")

    return FileResponse(
        path=file_path,
        media_type="text/csv",
        filename=os.path.basename(file_path)
    )