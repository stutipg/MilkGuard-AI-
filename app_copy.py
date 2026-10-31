import io
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import qrcode
import streamlit as st
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as PDFImage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "milkguard_samples.csv"
FEATURE_FILE = BASE_DIR / "milkguard_image_features.csv"
MODEL_FILE = BASE_DIR / "milkguard_image_model.joblib"
TRAINING_COLUMNS = [
    "image_width", "image_height", "mean_red", "mean_green", "mean_blue",
    "mean_brightness", "grayscale_std", "mean_saturation", "edge_variation"
]
PARAMETERS = [
    ("Fat", "%"), ("Water content", "%"), ("Starch", "Detected / not detected"),
    ("Protein", "%"), ("SNF", "%"), ("pH", "pH units"), ("Lactose", "%"),
    ("Minerals", "%"), ("Density", "g/mL"), ("Acidity", "% lactic acid"),
    ("Urea", "Detected / not detected"), ("Added sugar", "Detected / not detected")
]

st.set_page_config(page_title="MilkGuard AI", page_icon="🥛", layout="wide")


def load_data(path=DATA_FILE):
    if path.exists():
        try:
            return pd.read_csv(path).fillna("")
        except Exception as exc:
            st.error(f"Could not read {path.name}: {exc}")
    return pd.DataFrame()


def extract_image_features(file_obj):
    """Extract image-pixel features only; these are not milk chemistry measurements."""
    image = Image.open(file_obj).convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)
    pixels = arr.reshape(-1, 3).astype(np.float32)
    rgb = pixels.mean(axis=0)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)
    saturation = float(hsv[:, :, 1].mean())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    # A simple pixel-gradient statistic, not a laboratory or adulteration measurement.
    gx = np.diff(gray.astype(np.float32), axis=1)
    gy = np.diff(gray.astype(np.float32), axis=0)
    edge_variation = float((np.var(gx) + np.var(gy)) / 2.0)
    features = {
        "image_width": int(image.width),
        "image_height": int(image.height),
        "mean_red": round(float(rgb[0]), 3),
        "mean_green": round(float(rgb[1]), 3),
        "mean_blue": round(float(rgb[2]), 3),
        "mean_brightness": round(brightness, 3),
        "grayscale_std": round(contrast, 3),
        "mean_saturation": round(saturation, 3),
        "edge_variation": round(edge_variation, 3),
    }
    if brightness < 65:
        lighting = "Very dark image; retake in brighter, even light"
    elif brightness > 235:
        lighting = "Possibly overexposed; reduce glare"
    else:
        lighting = "Brightness in a usable image range"
    if contrast < 8:
        contrast_note = "Low visible contrast"
    else:
        contrast_note = "Visible tonal variation"
    return image, features, lighting, contrast_note


def generate_demo_readings(sample_id):
    """Create clearly labelled demonstration values; these are not measured milk results."""
    import hashlib
    seed = int(hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return {
        "Fat": round(float(rng.uniform(3.0, 4.5)), 2),
        "Water content": round(float(rng.uniform(0.0, 5.0)), 2),
        "Protein": round(float(rng.uniform(2.8, 3.6)), 2),
        "SNF": round(float(rng.uniform(8.0, 9.5)), 2),
        "pH": round(float(rng.uniform(6.4, 6.8)), 2),
        "Lactose": round(float(rng.uniform(4.5, 5.2)), 2),
        "Minerals": round(float(rng.uniform(0.6, 0.8)), 2),
        "Density": round(float(rng.uniform(1.028, 1.034)), 3),
        "Acidity": round(float(rng.uniform(0.13, 0.18)), 3),
        "Starch": "Simulated: Not detected",
        "Urea": "Simulated: Not detected",
        "Added sugar": "Simulated: Not detected",
    }


def qr_image(sample_id):
    payload = json.dumps({"project": "MilkGuard AI", "sample_id": sample_id,
                          "note": "Identifier only; not a test result."})
    qr = qrcode.QRCode(box_size=7, border=3)
    qr.add_data(payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def pdf_report(record, photo_bytes=None, qr_bytes=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("MilkGuard AI – Sample Analysis Report", styles["Title"]),
        Spacer(1, 8),
        Paragraph("Prototype report. Photo-derived values describe image pixels only; they do not determine milk composition or confirm adulteration.", styles["BodyText"]),
        Spacer(1, 10),
        Paragraph(f"<b>Sample ID:</b> {record.get('Sample ID', '')}", styles["BodyText"]),
        Paragraph(f"<b>Date/time:</b> {record.get('Date/time', '')}", styles["BodyText"]),
        Paragraph(f"<b>Source:</b> {record.get('Source', '')}", styles["BodyText"]),
        Paragraph("Photo-derived image features", styles["Heading2"]),
    ]
    rows = [["Image feature", "Value"]]
    for key, value in record.items():
        if key.startswith("Image feature - "):
            rows.append([key.replace("Image feature - ", ""), str(value)])
    if len(rows) == 1:
        rows.append(["No image features saved", ""])
    table = Table(rows, colWidths=[270, 230])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dcefe9")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [table, Spacer(1, 12), Paragraph("Milk composition and adulterant parameters", styles["Heading2"])]
    rows2 = [["Parameter", "Result", "Status"]]
    for name, unit in PARAMETERS:
        manual_value = record.get(f"Manual - {name}", "")
        source = record.get("Reading source", "Not tested")
        if str(manual_value).strip() and source != "Not tested":
            rows2.append([name, f"{manual_value} {unit}", f"Entered as: {source}"])
        else:
            rows2.append([name, "Not entered", "Requires suitable validated test"])
    table2 = Table(rows2, colWidths=[135, 165, 200], repeatRows=1)
    table2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f4e8d7")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [table2, Spacer(1, 10)]
    if photo_bytes:
        try:
            img = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
            img.thumbnail((500, 300))
            pbuf = io.BytesIO()
            img.save(pbuf, format="PNG")
            pbuf.seek(0)
            story += [Paragraph("Uploaded photograph", styles["Heading2"]),
                      PDFImage(pbuf, width=250, height=150), Spacer(1, 8)]
        except Exception:
            pass
    if qr_bytes:
        story += [Paragraph("Sample tracking QR code", styles["Heading2"]),
                  PDFImage(io.BytesIO(qr_bytes), width=95, height=95)]
    story += [Spacer(1, 10), Paragraph(
        "An ordinary RGB photograph cannot reliably quantify fat, water, protein, SNF, lactose, minerals, pH, density, acidity, starch, urea or added sugar. Use validated laboratory methods or calibrated instruments. This report is not food-safety certification.",
        styles["Italic"])]
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


def get_saved_model():
    if MODEL_FILE.exists():
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            return None
    return None


st.title("🥛 MilkGuard AI")
st.subheader("Smartphone image analysis, sample records and experimental AI")
st.warning("IMPORTANT: Image features describe the photograph, not milk chemistry. An AI classification is shown only after a model is trained using independently verified, labelled samples.")

tab1, tab2, tab3, tab4 = st.tabs(["📷 Register & analyse", "📊 Dashboard", "🔎 Track sample", "🧠 AI model"])

with tab1:
    st.header("Register a milk sample")
    with st.form("sample_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            sample_id = st.text_input("Sample ID", value=f"MG-{datetime.now():%Y%m%d-%H%M%S}")
            source = st.selectbox("Sample source", ["Dairy farm", "Milk collection centre", "Retail shop", "Household", "Other"])
        with col2:
            operator = st.text_input("Collected/tested by (optional)")
            notes = st.text_area("Notes (optional)")
        st.subheader("Automatic demo readings")
        st.info("No manual typing required. The app will generate example values for the dashboard. These are SIMULATED demo values, not measurements from the photograph or a milk-testing instrument.")
        uploaded = st.file_uploader("Upload milk photograph", type=["jpg", "jpeg", "png"])
        submit = st.form_submit_button("Analyse photo and save sample", use_container_width=True)

    if submit:
        if not sample_id.strip():
            st.error("Please enter a sample ID.")
        elif uploaded is None:
            st.error("Please upload a photograph.")
        else:
            old = load_data()
            if not old.empty and sample_id.strip() in old["Sample ID"].astype(str).tolist():
                st.error("Sample ID already exists. Please use a unique ID.")
            else:
                photo_bytes = uploaded.getvalue()
                try:
                    image, features, lighting, contrast_note = extract_image_features(io.BytesIO(photo_bytes))
                    demo_readings = generate_demo_readings(sample_id.strip())
                    record = {
                        "Sample ID": sample_id.strip(),
                        "Date/time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Source": source,
                        "Operator": operator,
                        "Notes": notes,
                        "Photo filename": uploaded.name,
                        "Reading source": "SIMULATED DEMO VALUES - NOT TEST RESULTS",
                        "Data type": "SIMULATED - NOT MEASURED",
                        "Manual - Fat": demo_readings["Fat"],
                        "Manual - Water content": demo_readings["Water content"],
                        "Manual - Protein": demo_readings["Protein"],
                        "Manual - SNF": demo_readings["SNF"],
                        "Manual - pH": demo_readings["pH"],
                        "Manual - Lactose": demo_readings["Lactose"],
                        "Manual - Minerals": demo_readings["Minerals"],
                        "Manual - Density": demo_readings["Density"],
                        "Manual - Acidity": demo_readings["Acidity"],
                        "Manual - Starch": demo_readings["Starch"],
                        "Manual - Urea": demo_readings["Urea"],
                        "Manual - Added sugar": demo_readings["Added sugar"],
                        "Image lighting note": lighting,
                        "Image contrast note": contrast_note,
                    }
                    for key, value in features.items():
                        record[f"Image feature - {key}"] = value
                    combined = pd.concat([old, pd.DataFrame([record])], ignore_index=True)
                    combined.to_csv(DATA_FILE, index=False)
                    feature_record = {**features, "sample_id": sample_id.strip(), "label": ""}
                    feature_df = load_data(FEATURE_FILE)
                    pd.concat([feature_df, pd.DataFrame([feature_record])], ignore_index=True).to_csv(FEATURE_FILE, index=False)
                    st.session_state["latest_record"] = record
                    st.session_state["latest_photo"] = photo_bytes
                    st.success("Sample and image features saved to CSV.")
                except Exception as exc:
                    st.error(f"Could not analyse this image: {exc}")

    if "latest_record" in st.session_state:
        rec = st.session_state["latest_record"]
        st.divider()
        st.header("Latest photo analysis")
        st.image(st.session_state["latest_photo"], caption="Uploaded milk photograph", width=350)
        st.warning("The milk-parameter values saved for this sample are SIMULATED DEMO VALUES. They were generated automatically and were not measured from the photo or a testing device.")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Brightness", rec.get("Image feature - mean_brightness", "N/A"))
        metric_cols[1].metric("Contrast", rec.get("Image feature - grayscale_std", "N/A"))
        metric_cols[2].metric("Image size", f"{rec.get('Image feature - image_width', 'N/A')} × {rec.get('Image feature - image_height', 'N/A')}")
        st.write("**Lighting check:**", rec.get("Image lighting note", "N/A"))
        st.write("**Contrast check:**", rec.get("Image contrast note", "N/A"))
        rgb_df = pd.DataFrame({"Mean pixel value": [rec.get("Image feature - mean_red", 0), rec.get("Image feature - mean_green", 0), rec.get("Image feature - mean_blue", 0)]}, index=["Red", "Green", "Blue"])
        st.subheader("RGB image features")
        st.bar_chart(rgb_df)
        model = get_saved_model()
        if model is not None:
            feature_row = pd.DataFrame([{key: rec.get(f"Image feature - {key}", 0) for key in TRAINING_COLUMNS}])
            prediction = model.predict(feature_row)[0]
            probabilities = model.predict_proba(feature_row)[0]
            confidence = float(np.max(probabilities)) * 100
            st.warning(f"Experimental model output: {prediction} (model confidence: {confidence:.1f}%). This is not a validated food-safety result.")
        else:
            st.info("AI classifier not available yet: add independently tested, labelled training samples in the AI model tab before training.")
        qr_bytes = qr_image(rec["Sample ID"])
        st.image(qr_bytes, caption="Sample tracking QR code", width=150)
        st.download_button("Download PDF report", data=pdf_report(rec, st.session_state["latest_photo"], qr_bytes), file_name=f"MilkGuard_{rec['Sample ID']}.pdf", mime="application/pdf")
        st.download_button("Download QR code", data=qr_bytes, file_name=f"MilkGuard_QR_{rec['Sample ID']}.png", mime="image/png")

with tab2:
    st.header("Dashboard")
    df = load_data()
    if df.empty:
        st.info("No samples recorded yet. Register a sample first.")
    else:
        a, b, c = st.columns(3)
        a.metric("Total samples", len(df))
        b.metric("Sample sources", df["Source"].nunique())
        c.metric("Latest sample", str(df.iloc[-1]["Sample ID"]))
        st.subheader("Samples by source")
        st.bar_chart(df["Source"].value_counts())
        st.subheader("Samples recorded over time")
        dates = pd.to_datetime(df["Date/time"], errors="coerce").dt.date
        st.line_chart(dates.value_counts().sort_index())
        st.warning("Dashboard milk-composition columns contain simulated demo values unless a record is explicitly sourced from a real instrument or laboratory. Do not use these values for food-safety decisions.")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download all records (CSV)", data=df.to_csv(index=False).encode("utf-8"), file_name="milkguard_all_samples.csv", mime="text/csv")

with tab3:
    st.header("Track a saved sample")
    df = load_data()
    if df.empty:
        st.info("No saved samples yet.")
    else:
        chosen = st.selectbox("Choose Sample ID", df["Sample ID"].astype(str).tolist())
        result = df[df["Sample ID"].astype(str) == chosen].iloc[-1].to_dict()
        st.json({key: str(value) for key, value in result.items()})
        qr_bytes = qr_image(chosen)
        st.image(qr_bytes, caption="Sample ID QR code", width=160)
        st.download_button("Download tracking QR", data=qr_bytes, file_name=f"MilkGuard_QR_{chosen}.png", mime="image/png")
        st.download_button("Download PDF report", data=pdf_report(result, qr_bytes=qr_bytes), file_name=f"MilkGuard_{chosen}.pdf", mime="application/pdf")

with tab4:
    st.header("AI model training")
    st.write("The model needs photographs paired with labels established by a suitable independent test. Do not label samples by appearance or by adding water/starch without a controlled reference method.")
    st.info("Expected training CSV columns: " + ", ".join(TRAINING_COLUMNS + ["label"]))
    st.download_button("Download blank training CSV template", data=pd.DataFrame(columns=TRAINING_COLUMNS + ["label"]).to_csv(index=False).encode("utf-8"), file_name="milkguard_training_template.csv", mime="text/csv")
    training_upload = st.file_uploader("Upload labelled training CSV", type=["csv"], key="training_csv")
    if training_upload is not None:
        try:
            training_df = pd.read_csv(training_upload)
            missing = [col for col in TRAINING_COLUMNS + ["label"] if col not in training_df.columns]
            if missing:
                st.error("Missing columns: " + ", ".join(missing))
            else:
                training_df = training_df[TRAINING_COLUMNS + ["label"]].dropna()
                training_df["label"] = training_df["label"].astype(str)
                st.write("Training rows:", len(training_df))
                st.write("Samples per label:")
                st.dataframe(training_df["label"].value_counts().rename("count"))
                if st.button("Train experimental image classifier"):
                    counts = training_df["label"].value_counts()
                    if len(counts) < 2 or counts.min() < 10:
                        st.error("Need at least 2 labels and at least 10 independently verified samples per label. More data is strongly recommended.")
                    else:
                        X = training_df[TRAINING_COLUMNS].apply(pd.to_numeric, errors="coerce")
                        valid = X.notna().all(axis=1)
                        X = X.loc[valid]
                        y = training_df.loc[valid, "label"]
                        if y.nunique() < 2 or y.value_counts().min() < 10:
                            st.error("Not enough valid numeric rows for each label after cleaning.")
                        else:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
                            clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
                            clf.fit(X_train, y_train)
                            predictions = clf.predict(X_test)
                            accuracy = accuracy_score(y_test, predictions)
                            joblib.dump(clf, MODEL_FILE)
                            st.success(f"Experimental model trained and saved. Holdout accuracy: {accuracy:.1%}")
                            st.text(classification_report(y_test, predictions, zero_division=0))
                            st.warning("A holdout score from a small dataset is not proof of real-world reliability. Validate on new samples, controlled lighting and independent laboratory results before making detection claims.")
        except Exception as exc:
            st.error(f"Could not read training CSV: {exc}")
    if MODEL_FILE.exists():
        st.success("A local experimental model file exists: milkguard_image_model.joblib")
        st.caption("Delete this model file if you want to reset the experimental model. Do not use it for safety decisions without independent validation.")
    else:
        st.info("No trained model exists yet. Until you have a verified labelled dataset, this app will show image features only, not an AI adulteration prediction.")

st.divider()
st.caption("MilkGuard AI | Educational prototype | Image features are not laboratory measurements or food-safety certification.")
