"""
Infant Discomfort Classification — Production Streamlit App
============================================================
Three classifiers:
  • Audio-only   : Tuned XGBoost (handcrafted features)
  • Image-only   : VGG19-based Keras CNN
  • Multimodal   : Early Fusion Dual-VGG19 Keras model
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Infant Discomfort Classifier",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ───────────────────────────────────────────────────────────────────
from utils.metrics_data import (
    get_audio_metrics,
    get_image_metrics,
    get_multimodal_metrics,
    get_model_comparison,
)
from utils.ui_helpers import (
    render_classification_report_table,
    render_probability_bars,
    section_header,
    model_info_expander,
    render_comparison_table,
    fusion_comparison_card,
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---- Fonts ---- */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0d1b2a 0%, #1b2d45 100%);
    }
    [data-testid="stSidebar"] * { color: #e8f4f8 !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; padding: 4px 0; }

    /* ---- Hero banner ---- */
    .hero-banner {
        background: linear-gradient(135deg, #1a3a5c 0%, #0e2340 60%, #102535 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    }
    .hero-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #7ecfe0;
        margin: 0 0 0.4rem 0;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #a8c8dc;
        margin: 0;
    }

    /* ---- Model summary cards ---- */
    .model-card {
        background: #f0f8ff;
        border-left: 5px solid #2980b9;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .model-card.best { border-left-color: #27ae60; background: #f0fff4; }
    .model-card h4 { margin: 0 0 0.3rem 0; font-size: 1rem; color: #1a3a5c; }
    .model-card p  { margin: 0; font-size: 0.85rem; color: #555; }

    /* ---- Prediction result ---- */
    .pred-box {
        background: linear-gradient(90deg, #1a3a5c, #27ae60);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: white;
        font-family: 'Space Mono', monospace;
        font-size: 1.4rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(39,174,96,0.3);
    }

    /* ---- Section dividers ---- */
    hr { border-color: #dce8f0; }

    /* ── Metric card tweaks ── */
    [data-testid="stMetric"] {
        background: #f7fafd;
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        border: 1px solid #d6eaf8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar navigation ─────────────────────────────────────────────────────────
PAGES = [
    "🏠 Overview",
    "🔊 Audio Classifier",
    "🖼️ Image Classifier",
    "🎛️ Multimodal Classifier",
    "📊 Metrics & Reports",
    "⚙️ Preprocessing Details",
    "🏗️ Architecture Notes",
]

with st.sidebar:
    st.markdown("## 👶 InfantDx")
    st.caption("Discomfort Classification Suite")
    st.divider()
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("Models")
    st.markdown("🔊 **Audio** — Tuned XGBoost")
    st.markdown("🖼️ **Image** — VGG19 Classifier")
    st.markdown("🎛️ **Multimodal** — Early Fusion")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown(
        """
        <div class="hero-banner">
          <p class="hero-title">👶 Infant Discomfort Classifier</p>
          <p class="hero-subtitle">
            AI-powered system to classify infant discomfort states from audio, image,
            or both modalities. Three independently trained classifiers cover audio-only,
            image-only, and multimodal (early fusion) prediction.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🏆 Best Models — At a Glance")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """<div class="model-card best">
            <h4>🔊 Audio Classifier</h4>
            <p>Tuned XGBoost · Handcrafted features</p>
            </div>""",
            unsafe_allow_html=True,
        )
        st.metric("Test Accuracy", "86.17%")
        st.metric("Macro F1", "0.8696")

    with col2:
        st.markdown(
            """<div class="model-card best">
            <h4>🖼️ Image Classifier</h4>
            <p>VGG19-based CNN · Transfer learning</p>
            </div>""",
            unsafe_allow_html=True,
        )
        st.metric("Test Accuracy", "99.29%")
        st.metric("Test Loss", "0.0395")

    with col3:
        st.markdown(
            """<div class="model-card best">
            <h4>🎛️ Multimodal (Early Fusion)</h4>
            <p>Dual VGG19 · Image + Audio</p>
            </div>""",
            unsafe_allow_html=True,
        )
        st.metric("Test Accuracy", "97.18%")
        st.metric("Macro F1", "0.9689")

    st.divider()
    st.markdown("### 📋 Four Discomfort Classes")
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, label, desc in zip(
        [c1, c2, c3, c4],
        ["😢", "🥶", "😣", "😄"],
        ["Belly Pain", "Cold / Hot", "Discomfort", "Laugh"],
        ["Gastrointestinal distress", "Temperature discomfort", "General discomfort", "Positive state"],
    ):
        with col:
            st.markdown(f"**{icon} {label}**")
            st.caption(desc)

    st.divider()
    st.markdown("### 🚀 How to Use")
    st.markdown(
        """
        1. Select a classifier from the **sidebar**.
        2. Upload the required file(s).
        3. Click **Predict** to run inference.
        4. View the predicted class and confidence scores.

        Visit **Metrics & Reports** for full evaluation details from training notebooks.
        """
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AUDIO CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔊 Audio Classifier":
    section_header("🔊", "Audio Classifier", "Tuned XGBoost on 40 MFCC + extended handcrafted features")

    # Model info
    model_info_expander("Audio Model Details", {
        "Model": "Tuned XGBoost",
        "Saved artifacts": "tuned_xgboost.joblib, scaler.joblib, label_encoder.joblib",
        "Feature extraction": "librosa @ sr=16000, n_mfcc=40",
        "Feature count": "Hundreds (MFCC×40×6 + delta×40×6 + delta2×40×6 + chroma, spectral, mel, tonnetz, pitch...)",
        "Scaling": "StandardScaler (saved, transform only)",
        "Labels (in order)": "belly pain · cold hot · discomfort · laugh",
        "Test Accuracy": "0.8617",
        "Macro F1": "0.8696",
    })

    audio_file = st.file_uploader(
        "Upload an audio file", type=["wav", "mp3", "ogg", "flac", "m4a"]
    )

    if audio_file:
        st.audio(audio_file, format=f"audio/{audio_file.name.split('.')[-1]}")
        audio_file.seek(0)

        if st.button("🔍 Predict Audio", type="primary"):
            with st.spinner("Extracting features and predicting…"):
                try:
                    from utils.audio_utils import predict_audio
                    predicted_label, prob_dict = predict_audio(audio_file)
                    st.markdown(
                        f'<div class="pred-box">🏷️ Predicted: <strong>{predicted_label.upper()}</strong></div>',
                        unsafe_allow_html=True,
                    )
                    render_probability_bars(prob_dict, predicted_label)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.exception(e)
    else:
        st.info("☝️ Upload a WAV, MP3, OGG, FLAC, or M4A file to begin.")

    st.divider()
    st.markdown("### 📊 Audio Model Metrics")
    comparison, per_class, tuned = get_audio_metrics()

    col1, col2, col3 = st.columns(3)
    col1.metric("Tuned XGBoost Accuracy", f"{tuned['accuracy']:.4f}")
    col2.metric("Tuned XGBoost Macro F1", f"{tuned['macro_f1']:.4f}")
    col3.metric("Best CV F1 (XGBoost)", "0.8963 ± 0.0044")

    st.markdown("**Model Comparison (all audio classifiers)**")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    render_classification_report_table(per_class, "Tuned XGBoost — Per-Class Report (Test Set)")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: IMAGE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🖼️ Image Classifier":
    section_header("🖼️", "Image Classifier", "VGG19 backbone with custom dense head · Transfer learning")

    model_info_expander("Image Model Details", {
        "Saved model": "model.keras",
        "Backbone": "VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3))",
        "Head": "Flatten → Dense(512,relu) → Dropout(0.3) → Dense(256,relu) → Dropout(0.3) → Dense(128,relu) → Dropout(0.3) → Dense(4,softmax)",
        "Preprocessing": "mobilenet_v2.preprocess_input (used during training)",
        "Optimizer": "SGD(lr=0.001)",
        "Loss": "categorical_crossentropy",
        "Epochs": "10",
        "Labels (in order)": "Belly Pain · Cold Hot · Discomfort · Laugh",
        "Test Accuracy": "0.9929",
        "Test Loss": "0.0395",
    })

    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if img_file:
        from PIL import Image as PILImage
        image = PILImage.open(img_file)
        col_img, col_pred = st.columns([1, 2])
        with col_img:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        img_file.seek(0)
        if st.button("🔍 Predict Image", type="primary"):
            with st.spinner("Running VGG19 inference…"):
                try:
                    from utils.image_utils import predict_image
                    predicted_label, prob_dict = predict_image(img_file)
                    with col_pred:
                        st.markdown(
                            f'<div class="pred-box">🏷️ Predicted: <strong>{predicted_label.upper()}</strong></div>',
                            unsafe_allow_html=True,
                        )
                        render_probability_bars(prob_dict, predicted_label)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.exception(e)
    else:
        st.info("☝️ Upload a JPG, PNG, or similar image to begin.")

    st.divider()
    st.markdown("### 📊 Image Model Metrics")
    img_m = get_image_metrics()
    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy", f"{img_m['test_accuracy']:.4f}")
    col2.metric("Test Loss", f"{img_m['test_loss']:.6f}")
    col3.metric("Final Val Accuracy", f"{img_m['final_val_accuracy']:.4f}")

    with st.expander("Training Configuration"):
        st.json({
            "backbone": img_m["backbone"],
            "optimizer": img_m["optimizer"],
            "loss": img_m["loss_fn"],
            "epochs": img_m["epochs"],
            "input_size": img_m["input_size"],
        })


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MULTIMODAL CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎛️ Multimodal Classifier":
    section_header(
        "🎛️", "Multimodal Classifier",
        "Early Fusion · Dual VGG19 (image branch + audio mel-spectrogram branch)"
    )

    st.success(
        "✅ **Early Fusion is the best multimodal strategy.**  "
        "It outperforms Late Fusion: Acc 0.9718 vs 0.9577, F1 0.9689 vs 0.9563."
    )

    model_info_expander("Multimodal Model Details", {
        "Saved model": "earlyfusionbest.keras",
        "Strategy": "Early Fusion (best)",
        "Image branch": "VGG19(imagenet, include_top=False) + GlobalAveragePooling2D",
        "Audio branch": "VGG19(imagenet, include_top=False) + GlobalAveragePooling2D on mel-spectrogram",
        "Fusion head": "Concat → Dense(512,relu) → BN → Dropout(0.4) → Dense(256,relu) → Dropout(0.3) → Dense(4,softmax)",
        "Image input key": "image_input",
        "Audio input key": "audio_input",
        "Labels (in order)": "bellypain · coldhot · discomfort · laugh",
        "Test Accuracy": "0.9718",
        "Macro F1": "0.9689",
    })

    col_a, col_b = st.columns(2)
    with col_a:
        img_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if img_file:
            from PIL import Image as PILImage
            st.image(PILImage.open(img_file), caption="Uploaded Image", use_container_width=True)
    with col_b:
        audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg", "flac", "m4a"])
        if audio_file:
            st.audio(audio_file)
            audio_file.seek(0)

    if img_file and audio_file:
        if st.button("🔍 Predict Multimodal", type="primary"):
            with st.spinner("Preprocessing both inputs and running Early Fusion inference…"):
                try:
                    img_file.seek(0)
                    audio_file.seek(0)
                    from utils.multimodal_utils import predict_multimodal
                    predicted_label, prob_dict = predict_multimodal(audio_file, img_file)
                    st.markdown(
                        f'<div class="pred-box">🏷️ Predicted: <strong>{predicted_label.upper()}</strong></div>',
                        unsafe_allow_html=True,
                    )
                    render_probability_bars(prob_dict, predicted_label)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.exception(e)
    else:
        st.info("☝️ Upload **both** an image and an audio file to enable multimodal prediction.")

    st.divider()
    st.markdown("### 📊 Multimodal Metrics")
    early, late = get_multimodal_metrics()
    fusion_comparison_card(early, late)

    st.markdown("---")
    st.markdown("**Early Fusion — Per-Class Report**")
    render_classification_report_table(early["per_class"])
    st.markdown("**Late Fusion — Per-Class Report**")
    render_classification_report_table(late["per_class"])


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: METRICS & REPORTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Metrics & Reports":
    section_header("📊", "Metrics & Reports", "All evaluation results from training notebooks")

    # Cross-model comparison
    st.markdown("### 🔬 Cross-Modality Model Comparison")
    render_comparison_table(get_model_comparison())

    st.divider()

    tabs = st.tabs(["🔊 Audio", "🖼️ Image", "🎛️ Multimodal"])

    # ── Audio tab ────────────────────────────────────────────────────────────
    with tabs[0]:
        comparison, per_class, tuned = get_audio_metrics()

        st.markdown("#### Audio Classifier Comparison")
        c1, c2 = st.columns(2)
        c1.metric("Best Model", tuned["model"])
        c1.metric("Test Accuracy", f"{tuned['accuracy']:.4f}")
        c2.metric("Macro F1", f"{tuned['macro_f1']:.4f}")
        c2.metric("Best CV F1", "0.8963 ± 0.0044")

        st.dataframe(comparison, use_container_width=True, hide_index=True)

        st.markdown("#### Tuned XGBoost — Classification Report (Test Set, n=94)")
        render_classification_report_table(per_class)

        st.markdown("#### Per-Class Highlights")
        with st.expander("View observations"):
            st.markdown(
                """
                - **laugh** achieves perfect precision and recall (1.00 / 1.00) — highly separable class.
                - **discomfort** is the hardest class: F1 = 0.75, recall 0.74.
                - **belly pain** and **cold hot** are well-classified with F1 ≥ 0.82.
                - Overall weighted F1 = 0.86, which closely tracks macro F1 = 0.87.
                """
            )

    # ── Image tab ────────────────────────────────────────────────────────────
    with tabs[1]:
        img_m = get_image_metrics()
        st.markdown("#### VGG19 Image Classifier")

        c1, c2, c3 = st.columns(3)
        c1.metric("Test Accuracy", f"{img_m['test_accuracy']:.4f}")
        c2.metric("Test Loss", f"{img_m['test_loss']:.6f}")
        c3.metric("Final Val Accuracy", f"{img_m['final_val_accuracy']:.4f}")

        with st.expander("Full configuration"):
            st.json({k: str(v) for k, v in img_m.items()})

        st.markdown(
            """
            > **Note:** No per-class breakdown was extracted from the image notebook.
            > The model achieves near-perfect test accuracy (99.29%), suggesting strong
            > separation across all four classes. Final validation accuracy reached 1.0000
            > in the last reported epoch.
            """
        )

    # ── Multimodal tab ───────────────────────────────────────────────────────
    with tabs[2]:
        early, late = get_multimodal_metrics()

        st.markdown("#### Fusion Strategy Comparison")
        fusion_comparison_card(early, late)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Early Fusion — Per-Class Report**")
            render_classification_report_table(early["per_class"])
        with col2:
            st.markdown("**Late Fusion — Per-Class Report**")
            render_classification_report_table(late["per_class"])

        with st.expander("Observations"):
            st.markdown(
                """
                - **Early Fusion** concatenates feature vectors from both VGG19 branches *before*
                  the classification head, allowing joint feature learning.
                - **discomfort** achieves F1=1.00 in Early Fusion — a major improvement over audio alone.
                - **coldhot** recall of 0.88 in both strategies is the shared weak point.
                - Early Fusion beats Late Fusion on every aggregate metric.
                """
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREPROCESSING DETAILS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Preprocessing Details":
    section_header("⚙️", "Preprocessing Details", "Exact pipelines used during training — do not modify")

    tabs = st.tabs(["🔊 Audio", "🖼️ Image", "🎛️ Multimodal Audio", "🎛️ Multimodal Image"])

    with tabs[0]:
        st.markdown("#### Audio-Only Preprocessing")
        st.code(
            """
# 1. Load audio
y, sr = librosa.load(file_path, sr=16000)

# 2. Basic signal stats
duration_sec, signal_mean, signal_std, signal_max
rms_mean, rms_std   (from librosa.feature.rms)
zcr_mean, zcr_std   (from librosa.feature.zero_crossing_rate)
energy_entropy      (custom function, n_short_blocks=10)

# 3. MFCC (n_mfcc=40) → summarize each of 40 coefficients with
#    mean, std, min, max, skew, kurtosis → 40×6 = 240 features
mfcc        = librosa.feature.mfcc(y, sr, n_mfcc=40)
mfcc_delta  = librosa.feature.delta(mfcc)
mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

# 4. Spectral / tonal features (same summarize aggregation)
chroma, spectral_centroid, spectral_bandwidth,
spectral_rolloff, spectral_contrast, spectral_flatness,
mel_spectrogram, tonnetz

# 5. Pitch (pyin)
f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=C2, fmax=C7, sr=sr)
→ f0_mean, f0_std, f0_min, f0_max, voiced_ratio

# 6. Assemble into pandas DataFrame (same column order as training)

# 7. Clean: replace inf/-inf with NaN, then fill NaN with 0

# 8. Scale: scaler.transform(df)  ← never re-fit

# 9. Predict with tuned_xgboost.joblib
# 10. Decode label with label_encoder.joblib
            """,
            language="python",
        )

    with tabs[1]:
        st.markdown("#### Image-Only Preprocessing")
        st.code(
            """
# 1. Open image with PIL
img = Image.open(file).convert("RGB")

# 2. Resize to (224, 224)
img = img.resize((224, 224))

# 3. Convert to numpy float32
arr = np.array(img, dtype=np.float32)

# ⚠️ IMPORTANT: Training used mobilenet_v2.preprocess_input
# even though the backbone is VGG19. Must preserve this exactly.
arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)

# 4. Add batch dimension
arr = np.expand_dims(arr, axis=0)  # shape: (1, 224, 224, 3)

# 5. Predict with model.keras
# 6. Map argmax → ["Belly Pain","Cold Hot","Discomfort","Laugh"]
            """,
            language="python",
        )

    with tabs[2]:
        st.markdown("#### Multimodal — Audio Preprocessing")
        st.code(
            """
# 1. Load at sr=22050, mono=True
y, sr = librosa.load(path, sr=22050, mono=True)

# 2. Fixed duration = 3.0 seconds
n_samples = int(22050 * 3.0)  # = 66150
if len(y) < n_samples:
    y = np.pad(y, (0, n_samples - len(y)))
else:
    y = y[:n_samples]

# 3. Mel spectrogram
mel = librosa.feature.melspectrogram(
    y=y, sr=sr, n_mels=128, hop_length=512
)

# 4. Convert to dB
mel_db = librosa.power_to_db(mel, ref=np.max)

# 5. Min-max normalize to [0, 1]
mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

# 6. Replicate to 3 channels (VGG19 expects 3-channel input)
audio_tensor = np.stack([mel_norm, mel_norm, mel_norm], axis=-1)
# shape: (128, time_steps, 3) where time_steps ≈ 130

# 7. Add batch dim and pass as "audio_input"
            """,
            language="python",
        )

    with tabs[3]:
        st.markdown("#### Multimodal — Image Preprocessing")
        st.code(
            """
# Different from image-only model!
# Uses ImageNet normalization instead of mobilenet_v2.preprocess_input

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# 1. Open with PIL
img = Image.open(file).convert("RGB")

# 2. Resize to (224, 224)
img = img.resize((224, 224))

# 3. Normalize to [0, 1]
arr = np.array(img, dtype=np.float32) / 255.0

# 4. Channel-wise ImageNet normalization
arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
# shape: (224, 224, 3), dtype float32

# 5. Add batch dim and pass as "image_input"
            """,
            language="python",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ARCHITECTURE NOTES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏗️ Architecture Notes":
    section_header("🏗️", "Architecture Notes", "Model design decisions and pipeline structures")

    tabs = st.tabs(["🔊 Audio Pipeline", "🖼️ Image VGG19", "🎛️ Early Fusion Multimodal"])

    with tabs[0]:
        st.markdown(
            """
            ### Audio Classical ML Pipeline

            **Goal:** 4-class infant cry / discomfort recognition from raw audio.

            **Feature engineering:**
            Raw waveform → extensive handcrafted features capturing temporal, spectral,
            rhythmic, and pitch characteristics:

            | Feature Group | Dimensions | Aggregation |
            |---|---|---|
            | MFCC (n=40) | 40 × 6 = 240 | mean/std/min/max/skew/kurt per coeff |
            | MFCC Δ | 240 | same |
            | MFCC ΔΔ | 240 | same |
            | Chroma (12) | 72 | same |
            | Spectral centroid, bandwidth, rolloff | 18 each | same |
            | Spectral contrast (7 bands) | 42 | same |
            | Spectral flatness | 6 | same |
            | Mel spectrogram (128 bands) | 768 | same |
            | Tonnetz (6) | 36 | same |
            | RMS, ZCR, energy entropy | 5 | scalar |
            | Pitch (pyin) | 5 | f0 stats + voiced_ratio |

            **Classifiers compared:**
            - Random Forest (CV F1=0.887), KNN (0.815), Logistic Regression (0.851), **XGBoost (0.896 ★)**

            **Best model:** Tuned XGBoost
            - Artifacts: `tuned_xgboost.joblib`, `scaler.joblib`, `label_encoder.joblib`
            - StandardScaler applied before XGBoost
            """
        )

    with tabs[1]:
        st.markdown(
            """
            ### Image VGG19 Classifier

            **Goal:** 4-class infant discomfort recognition from still face/body images.

            ```
            Input (224, 224, 3)
               ↓
            VGG19(weights='imagenet', include_top=False)
               ↓ feature maps
            Flatten
               ↓
            Dense(512, activation='relu')
            Dropout(0.3)
               ↓
            Dense(256, activation='relu')
            Dropout(0.3)
               ↓
            Dense(128, activation='relu')
            Dropout(0.3)
               ↓
            Dense(4, activation='softmax')
            ```

            **Training:**
            - Optimizer: SGD (lr=0.001, no momentum specified)
            - Loss: categorical_crossentropy
            - Metric: accuracy
            - Epochs: 10
            - Final val accuracy: 1.0000

            **Key note on preprocessing:**
            Although the backbone is VGG19, the notebook applied
            `tf.keras.applications.mobilenet_v2.preprocess_input`.
            This must be preserved at inference time to avoid distribution shift.

            **Saved model:** `model.keras`
            """
        )

    with tabs[2]:
        st.markdown(
            """
            ### Early Fusion Multimodal Model

            **Goal:** Combine visual (face/body) and audio (cry) signals for 4-class infant
            discomfort classification.

            **Strategy: Early Fusion** (concatenate feature vectors before classification head)

            ```
            ┌─────────────────────────────┐   ┌─────────────────────────────────┐
            │  Image Branch               │   │  Audio Branch                   │
            │  Input: (224, 224, 3)       │   │  Input: (128, time_steps, 3)    │
            │  VGG19(imagenet,            │   │  VGG19(imagenet,                │
            │    include_top=False)       │   │    include_top=False)           │
            │  GlobalAveragePooling2D     │   │  GlobalAveragePooling2D         │
            └────────────┬────────────────┘   └───────────────┬─────────────────┘
                         │                                    │
                         └──────────── Concatenate ───────────┘
                                            │
                                      Dense(512, relu)
                                      BatchNormalization
                                      Dropout(0.4)
                                            │
                                      Dense(256, relu)
                                      Dropout(0.3)
                                            │
                                      Dense(4, softmax)
            ```

            **Input keys (functional API):**
            - `image_input` → image branch
            - `audio_input` → mel-spectrogram branch (shape must match VGG19 expectations)

            **Why Early Fusion wins:**
            Joint learning allows the model to discover cross-modal correlations
            (e.g., a high-pitched cry paired with a grimacing face) that Late Fusion
            (which merges independent predictions) cannot exploit.

            | Strategy | Accuracy | Macro F1 |
            |---|---|---|
            | **Early Fusion ★** | **0.9718** | **0.9689** |
            | Late Fusion | 0.9577 | 0.9563 |

            **Saved model:** `earlyfusionbest.keras`
            """
        )
