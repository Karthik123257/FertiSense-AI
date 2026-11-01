"""
pdf_extractor.py
Extracts Patient Name and clinical parameters from uploaded patient reports
using pdfplumber & regex for both Male and Female reports.
"""

import re
import pdfplumber

def extract_patient_name(text, fallback="Patient"):
    """
    Scans PDF text for 'Patient Name:', 'Patient:', or 'Name:' fields.
    """
    name_patterns = [
        r'Patient\s*Name\s*[:\-\|]\s*([A-Za-z\s\.]+?)(?:\n|\r|Age|Sex|Gender|Date|ID|\||$)',
        r'Patient\s*[:\-\|]\s*([A-Za-z\s\.]+?)(?:\n|\r|Age|Sex|Gender|Date|ID|\||$)',
        r'(?:^|\n)Name\s*[:\-\|]\s*([A-Za-z\s\.]+?)(?:\n|\r|Age|Sex|Gender|Date|ID|\||$)'
    ]
    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            clean_name = match.group(1).strip()
            # Avoid matching generic headers
            if len(clean_name) > 1 and clean_name.lower() not in ["report", "male", "female", "test"]:
                return clean_name
    return fallback


def extract_male_parameters(pdf_path_or_bytes, fallback_name="Patient"):
    text = ""
    with pdfplumber.open(pdf_path_or_bytes) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"

    patient_name = extract_patient_name(text, fallback=fallback_name)

    patterns = {
        'Volume(mL)': [
            r'(?:Ejaculate\s*)?Volume[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)',
            r'Volume\s*[:\-\|]?\s*([0-9]+\.?[0-9]*)'
        ],
        'Sperm_Concentration(M/mL)': [
            r'(?:Sperm\s*)?Concentration[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)',
            r'Sperm\s*Count[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)'
        ],
        'Progressive_Motility(%)': [
            r'Progressive\s*(?:Motility|\(PR\))?[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)',
            r'PR\s*\(Rapid\s*\+\s*Slow\)[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)'
        ],
        'Total_Motility(%)': [
            r'Total\s*Motility[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)',
            r'Motility\s*\(PR\s*\+\s*NP\)[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)'
        ],
        'Normal_Morphology(%)': [
            r'(?:Normal\s*)?Morphology[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)',
            r'Strict\s*Normal\s*Forms[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)'
        ],
        'Total_Testosterone(ng/dL)': [
            r'(?:Total\s*)?Testosterone[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)',
            r'Total\s*T[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)'
        ],
        'Sitting_Hours_Per_Day': [
            r'Sitting\s*(?:Hours|Time)?[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)',
            r'Desk\s*Hours[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)'
        ]
    }

    extracted = {
        'Volume(mL)': 2.5,
        'Sperm_Concentration(M/mL)': 35.0,
        'Progressive_Motility(%)': 38.0,
        'Total_Motility(%)': 52.0,
        'Normal_Morphology(%)': 5.0,
        'Total_Testosterone(ng/dL)': 520.0,
        'Sitting_Hours_Per_Day': 6.0
    }

    for param, regex_list in patterns.items():
        for reg in regex_list:
            match = re.search(reg, text, re.IGNORECASE)
            if match:
                try:
                    extracted[param] = float(match.group(1))
                    break
                except ValueError:
                    pass

    return patient_name, extracted, text


def extract_female_parameters(pdf_path_or_bytes, fallback_name="Patient"):
    text = ""
    with pdfplumber.open(pdf_path_or_bytes) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"

    patient_name = extract_patient_name(text, fallback=fallback_name)

    patterns = {
        'AMH(ng/mL)': [
            r'AMH[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)',
            r'Anti[\s\-]M[uü]llerian\s*Hormone[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)'
        ],
        'TSH (mIU/L)': [
            r'TSH[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)',
            r'Thyroid\s*Stimulating\s*Hormone[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)'
        ],
        'FSH(mIU/mL)': [
            r'FSH[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)',
            r'Follicle\s*Stimulating\s*Hormone[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)'
        ],
        'LH(mIU/mL)': [
            r'LH[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)',
            r'Luteinizing\s*Hormone[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)'
        ],
        'PRL(ng/mL)': [
            r'PRL[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)',
            r'Prolactin[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)'
        ],
        'Follicle No. (Total)': [
            r'(?:Antral\s*Follicle\s*Count|AFC|Follicle\s*No\.?\s*\(?Total\)?|Total\s*Follicles|Follicles)[^\d\n\r]*?[:\-\|\s]+([0-9]+)'
        ],
        'BMI': [
            r'BMI[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)',
            r'Body\s*Mass\s*Index[^\d\n\r]*?[:\-\|\s]+([0-9]+\.?[0-9]*)'
        ],
        'Cycle length(days)': [
            r'Cycle\s*length[^\d\n\r]*?[:\-\|\s]+([0-9]+)',
            r'Menstrual\s*Cycle[^\d\n\r]*?[:\-\|\s]+([0-9]+)'
        ]
    }

    extracted = {
        'AMH(ng/mL)': 2.8,
        'TSH (mIU/L)': 2.1,
        'FSH(mIU/mL)': 6.2,
        'LH(mIU/mL)': 5.4,
        'PRL(ng/mL)': 16.0,
        'Follicle No. (Total)': 14.0,
        'BMI': 22.5,
        'Cycle length(days)': 29.0
    }

    for param, regex_list in patterns.items():
        for reg in regex_list:
            match = re.search(reg, text, re.IGNORECASE)
            if match:
                try:
                    extracted[param] = float(match.group(1))
                    break
                except ValueError:
                    pass

    return patient_name, extracted, text