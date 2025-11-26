# # # # a.py
# # # import streamlit as st
# # # import os
# # # import io
# # # import numpy as np
# # # import librosa
# # # import librosa.display
# # # import matplotlib.pyplot as plt
# # # import noisereduce as nr
# # # from tensorflow.keras.models import load_model
# # # from tensorflow.keras.preprocessing import image
# # # from sklearn.metrics import confusion_matrix
# # # import seaborn as sns
# # # from PIL import Image

# # # st.set_page_config(layout="wide", page_title="Parkinson's Prediction Dashboard")

# # # # ---------------------------
# # # # Paths (adjust if your structure differs)
# # # # ---------------------------
# # # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # # MODEL_DIR = os.path.join(BASE_DIR, "models")      # your models folder
# # # PLOTS_DIR = os.path.join(BASE_DIR, "plots")       # your images folder

# # # IMG_SIZE = (128, 128)
# # # CHUNK_DURATION = 3.0
# # # SR = 22050

# # # # ---------------------------
# # # # Helper functions
# # # # ---------------------------
# # # def audio_to_chunks(file_like, chunk_duration=CHUNK_DURATION, sr=SR):
# # #     # librosa can accept a file-like object if we pass the buffer
# # #     try:
# # #         data, sr = librosa.load(file_like, sr=sr)
# # #         data = nr.reduce_noise(y=data, sr=sr)
# # #         chunk_len = int(chunk_duration * sr)
# # #         chunks = [data[i:i + chunk_len] for i in range(0, len(data), chunk_len) if len(data[i:i + chunk_len]) == chunk_len]
# # #         return chunks, sr
# # #     except Exception as e:
# # #         st.error(f"Error processing audio: {e}")
# # #         return [], sr

# # # def chunk_to_spectrogram_buf(chunk, sr):
# # #     S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, hop_length=512)
# # #     S_db = librosa.power_to_db(S, ref=np.max)
# # #     fig, ax = plt.subplots(figsize=(3,3))
# # #     librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel", ax=ax)
# # #     plt.axis("off")
# # #     buf = io.BytesIO()
# # #     plt.savefig(buf, bbox_inches="tight", pad_inches=0, format="png")
# # #     plt.close(fig)
# # #     buf.seek(0)
# # #     return buf

# # # def preprocess_spectrogram(img_buf):
# # #     try:
# # #         # Keras image loader can read a file-like object
# # #         img = image.load_img(img_buf, target_size=IMG_SIZE)
# # #         arr = image.img_to_array(img) / 255.0
# # #         arr = np.expand_dims(arr, axis=0)
# # #         return arr
# # #     except Exception as e:
# # #         st.error(f"Error preprocessing spectrogram: {e}")
# # #         return None

# # # def generate_confusion_plot(y_true, y_pred, labels=["Healthy","Parkinson's"]):
# # #     cm = confusion_matrix(y_true, y_pred)
# # #     fig, ax = plt.subplots(figsize=(4,4))
# # #     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
# # #     ax.set_xlabel("Predicted")
# # #     ax.set_ylabel("True")
# # #     st.pyplot(fig)

# # # # ---------------------------
# # # # Model loader with caching
# # # # ---------------------------
# # # @st.cache_resource
# # # def load_all_models(model_dir):
# # #     models = {}
# # #     # Map file names (or keys) to expected filenames
# # #     mapping = {
# # #         "bilstm": "best_bilstm.keras",
# # #         "cnn_lstm": "best_cnn_lstm_model.keras",
# # #         "resnet50": "best_resnet50.keras",
# # #         "gru": "best_gru.keras",
# # #         "mobilenetv2": "best_mobilenetv2.keras",   # optional
# # #         "cnn": "best_CNN.keras"                  # optional
# # #     }
# # #     loaded = {}
# # #     errors = {}
# # #     for key, fname in mapping.items():
# # #         path = os.path.join(model_dir, fname)
# # #         if os.path.exists(path):
# # #             try:
# # #                 loaded[key] = load_model(path)
# # #             except Exception as e:
# # #                 errors[key] = str(e)
# # #         else:
# # #             errors[key] = f"file not found: {path}"
# # #     return loaded, errors

# # # # ---------------------------
# # # # UI Sidebar
# # # # ---------------------------
# # # st.sidebar.title("Navigation")
# # # view = st.sidebar.radio("Go to", ["Test & Predict", "Model Metrics", "Performance Graphs"])

# # # # Load models once (cached)
# # # with st.spinner("Loading models..."):
# # #     MODELS, MODEL_ERRORS = load_all_models(MODEL_DIR)

# # # if MODEL_ERRORS:
# # #     # show errors (but not blocking)
# # #     for k, err in MODEL_ERRORS.items():
# # #         st.sidebar.write(f"{k}: {err}")

# # # if view == "Test & Predict":
# # #     st.header("Upload a WAV audio file for prediction")
# # #     uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

# # #     if uploaded_file:
# # #         if "audio_bytes" not in st.session_state:
# # #             st.session_state.audio_bytes = uploaded_file.read()
# # #         # show playback
# # #         st.audio(st.session_state.audio_bytes)

# # #         # process audio from bytes
# # #         chunks, sr = audio_to_chunks(io.BytesIO(st.session_state.audio_bytes))
# # #         st.write(f"Audio split into {len(chunks)} chunk(s) of {CHUNK_DURATION}s")
# # #         if len(chunks) == 0:
# # #             st.warning("No full-length chunks (3s) were found. Try a longer audio or reduce CHUNK_DURATION.")
# # #         else:
# # #             # take first chunk for preview/predictions
# # #             chunk = chunks[0]
# # #             img_buf = chunk_to_spectrogram_buf(chunk, sr)
# # #             st.image(img_buf, caption="Spectrogram (first chunk)", use_column_width=False)

# # #             x = preprocess_spectrogram(io.BytesIO(img_buf.getvalue()))
# # #             if x is not None:
# # #                 results = {}
# # #                 for name, model in MODELS.items():
# # #                     try:
# # #                         # models may return logits or probabilities; adapt if shape differs
# # #                         out = model.predict(x, verbose=0)
# # #                         # attempt common scenarios:
# # #                         score = None
# # #                         if isinstance(out, np.ndarray):
# # #                             # binary single-output
# # #                             if out.shape[-1] == 1:
# # #                                 score = float(out[0][0])
# # #                             else:
# # #                                 # multiclass: take probability of class 1
# # #                                 score = float(out[0][1]) if out.shape[-1] > 1 else float(out[0][0])
# # #                         else:
# # #                             score = float(out)
# # #                         label = "Parkinson's Disease (PD)" if score >= 0.5 else "Healthy Control (HC)"
# # #                         results[name] = {"score": score, "label": label}
# # #                     except Exception as e:
# # #                         results[name] = {"score": None, "label": f"ERROR: {e}"}

# # #                 # Voting summary
# # #                 pd_votes = sum(1 for r in results.values() if r["label"].startswith("Parkinson"))
# # #                 hc_votes = sum(1 for r in results.values() if r["label"].startswith("Healthy"))
# # #                 final_label = "Parkinson's Disease (PD)" if pd_votes > hc_votes else "Healthy Control (HC)"
# # #                 st.subheader("Voting Result")
# # #                 st.write(f"Final: **{final_label}** (PD votes: {pd_votes}, HC votes: {hc_votes})")

# # #                 st.subheader("Model-wise predictions")
# # #                 for k, v in results.items():
# # #                     st.write(f"- **{k}**: {v['label']} (score={v['score']})")

# # #                 # example confusion matrix (for demo) - replace with your real arrays if available
# # #                 if st.button("Show example confusion matrix"):
# # #                     y_true = np.array([0,0,1,1,0,1,0,1,1,0])
# # #                     y_pred = np.array([0,1,1,1,0,1,0,0,1,0])
# # #                     generate_confusion_plot(y_true, y_pred)

# # # elif view == "Model Metrics":
# # #     st.header("Model metrics and tables")
# # #     # you can render tables (hard-coded or loaded)
# # #     st.markdown("**Available models**")
# # #     for m in MODELS.keys():
# # #         st.write(f"- {m}")

# # #     if MODEL_ERRORS:
# # #         st.warning("Some models were not loaded. Check the sidebar for errors.")

# # #     # show first 5 rows of example CSV or feature lists if you have them
# # #     st.markdown("Feature lists (example)")
# # #     total_features = ["voiceID", "meanF0Hz", "maxF0Hz", "minF0Hz", "stdF0Hz", "jitter_local", "shimmer_local", "hnr", "label"]
# # #     st.write(total_features[:10])

# # # elif view == "Performance Graphs":
# # #     st.header("Local performance graphs")
# # #     names = ["CNN-LSTM.png", "BiLSTM.png", "GRU.png", "MobileNetV2.png", "ResNet50.png"]
# # #     cols = st.columns(2)
# # #     for i, fname in enumerate(names):
# # #         path = os.path.join(PLOTS_DIR, fname)
# # #         try:
# # #             if os.path.exists(path):
# # #                 img = Image.open(path)
# # #                 with cols[i % 2]:
# # #                     st.image(img, caption=fname, use_column_width=True)
# # #             else:
# # #                 with cols[i % 2]:
# # #                     st.info(f"{fname} not found in {PLOTS_DIR}")
# # #         except Exception as e:
# # #             st.error(f"Error loading {fname}: {e}")
# # # a.py
# # import streamlit as st
# # import os, io, math
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import noisereduce as nr
# # import librosa
# # import librosa.display
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.preprocessing import image
# # from sklearn.metrics import confusion_matrix
# # import seaborn as sns
# # from PIL import Image

# # # constants (must be defined before functions that use them)
# # IMG_SIZE = (128, 128)
# # CHUNK_DURATION = 3.0
# # SR = 22050

# # def safe_load_audio(file_like, sr=SR):
# #     try:
# #         y, sr_ret = librosa.load(file_like, sr=sr)
# #         return y, sr_ret
# #     except Exception as e:
# #         st.error(f"Error loading audio: {e}")
# #         return None, sr


# # # --- page config
# # st.set_page_config(layout="wide", page_title="Parkinson's Prediction Dashboard")

# # # ---------------------------
# # # Paths (adjust if your structure differs)
# # # ---------------------------
# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # MODEL_DIR = os.path.join(BASE_DIR, "models")      # your models folder
# # PLOTS_DIR = os.path.join(BASE_DIR, "plots")       # your images folder

# # IMG_SIZE = (128, 128)
# # CHUNK_DURATION = 3.0
# # SR = 22050

# # # ---------------------------
# # # Helper functions (feature extraction, spectrogram preprocess, etc.)
# # # ---------------------------

# # def safe_load_audio(file_like, sr=SR):
# #     try:
# #         y, sr_ret = librosa.load(file_like, sr=sr)
# #         return y, sr_ret
# #     except Exception as e:
# #         st.error(f"Error loading audio: {e}")
# #         return None, sr

# # def reduce_noise(y, sr):
# #     try:
# #         return nr.reduce_noise(y=y, sr=sr)
# #     except Exception:
# #         return y

# # def extract_basic_features_from_audio(y, sr):
# #     """
# #     Best-effort extraction for the requested feature set.
# #     Many of the features (jitter, shimmer, HNR) are approximated.
# #     Returns a dict with feature_name -> value
# #     """
# #     features = {}
# #     # safe guards
# #     if y is None or len(y) == 0:
# #         return {k: np.nan for k in FEATURE_COLUMNS}

# #     # 1) pitch (fundamental freq) using librosa.pyin (if available) fallback to yin
# #     f0 = None
# #     try:
# #         # pyin provides voiced F0 with n_frames; use median/mean over voiced frames
# #         f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
# #         # f0 is array with np.nan for unvoiced; filter
# #         voiced_f0 = f0[~np.isnan(f0)]
# #         if voiced_f0.size == 0:
# #             raise Exception("no voiced frames from pyin")
# #     except Exception:
# #         try:
# #             # fallback to yin
# #             f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
# #             voiced_f0 = f0[~np.isnan(f0)]
# #             if voiced_f0.size == 0:
# #                 voiced_f0 = np.array([])
# #         except Exception:
# #             voiced_f0 = np.array([])

# #     # compute mean/max/min/std for F0
# #     if voiced_f0.size > 0:
# #         features["meanF0Hz"] = float(np.mean(voiced_f0))
# #         features["maxF0Hz"] = float(np.max(voiced_f0))
# #         features["minF0Hz"] = float(np.min(voiced_f0))
# #         features["stdF0Hz"] = float(np.std(voiced_f0))
# #     else:
# #         features["meanF0Hz"] = np.nan
# #         features["maxF0Hz"] = np.nan
# #         features["minF0Hz"] = np.nan
# #         features["stdF0Hz"] = np.nan

# #     # 2) Jitter approximations:
# #     # jitter_local: average absolute difference between consecutive voiced F0 divided by mean f0
# #     try:
# #         if voiced_f0.size > 1 and not math.isnan(features["meanF0Hz"]):
# #             jitter_vals = np.abs(np.diff(voiced_f0))
# #             features["jitter_local"] = float(np.mean(jitter_vals) / (features["meanF0Hz"] + 1e-12))
# #             features["jitter_abs"] = float(np.mean(jitter_vals))
# #             # rap, ddp, ppq5 are more complex; approximate with rolling-window normalized diffs
# #             # rap: relative average perturbation (3-sample)
# #             if voiced_f0.size >= 3:
# #                 rap = np.mean(np.abs((voiced_f0[1:-1] - (voiced_f0[:-2] + voiced_f0[2:]) / 2.0)))
# #                 features["jitter_rap"] = float(rap / (features["meanF0Hz"] + 1e-12))
# #             else:
# #                 features["jitter_rap"] = np.nan
# #             # ddp approx (difference of differences)
# #             if voiced_f0.size >= 5:
# #                 ddp = np.mean(np.abs(np.diff(voiced_f0, n=2)))
# #                 features["jitter_ddp"] = float(ddp / (features["meanF0Hz"] + 1e-12))
# #             else:
# #                 features["jitter_ddp"] = np.nan
# #             # ppq5: average absolute difference with average over window 5
# #             if voiced_f0.size >= 5:
# #                 ppq5_vals = []
# #                 for i in range(2, len(voiced_f0)-2):
# #                     ppq5_vals.append(abs(voiced_f0[i] - np.mean(voiced_f0[i-2:i+3])))
# #                 features["jitter_ppq5"] = float(np.mean(ppq5_vals) / (features["meanF0Hz"] + 1e-12))
# #             else:
# #                 features["jitter_ppq5"] = np.nan
# #         else:
# #             features["jitter_local"] = np.nan
# #             features["jitter_abs"] = np.nan
# #             features["jitter_rap"] = np.nan
# #             features["jitter_ddp"] = np.nan
# #             features["jitter_ppq5"] = np.nan
# #     except Exception:
# #         features["jitter_local"] = np.nan
# #         features["jitter_abs"] = np.nan
# #         features["jitter_rap"] = np.nan
# #         features["jitter_ddp"] = np.nan
# #         features["jitter_ppq5"] = np.nan

# #     # 3) Shimmer approximations:
# #     # shimmer_local: amplitude perturbation - use short-time energy or RMS frame-by-frame
# #     try:
# #         hop_length = 512
# #         frame_length = 1024
# #         rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
# #         if rms.size > 1:
# #             amp_diff = np.abs(np.diff(rms))
# #             features["shimmer_local"] = float(np.mean(amp_diff) / (np.mean(rms) + 1e-12))
# #             features["shimmer_db"] = float(20 * np.log10(np.mean(rms) + 1e-12))
# #             # apq3, apq5, dda approximate similar to jitter but on amplitude frames
# #             if rms.size >= 3:
# #                 apq3 = np.mean(np.abs(rms[1:-1] - (rms[:-2] + rms[2:]) / 2.0))
# #                 features["shimmer_apq3"] = float(apq3 / (np.mean(rms) + 1e-12))
# #             else:
# #                 features["shimmer_apq3"] = np.nan
# #             if rms.size >= 5:
# #                 apq5_vals = []
# #                 for i in range(2, len(rms)-2):
# #                     apq5_vals.append(abs(rms[i] - np.mean(rms[i-2:i+3])))
# #                 features["shimmer_apq5"] = float(np.mean(apq5_vals) / (np.mean(rms) + 1e-12))
# #             else:
# #                 features["shimmer_apq5"] = np.nan
# #             if rms.size >= 2:
# #                 dda = np.mean(np.abs(np.diff(rms)))
# #                 features["shimmer_dda"] = float(dda / (np.mean(rms) + 1e-12))
# #             else:
# #                 features["shimmer_dda"] = np.nan
# #         else:
# #             features["shimmer_local"] = np.nan
# #             features["shimmer_db"] = np.nan
# #             features["shimmer_apq3"] = np.nan
# #             features["shimmer_apq5"] = np.nan
# #             features["shimmer_dda"] = np.nan
# #     except Exception:
# #         features["shimmer_local"] = np.nan
# #         features["shimmer_db"] = np.nan
# #         features["shimmer_apq3"] = np.nan
# #         features["shimmer_apq5"] = np.nan
# #         features["shimmer_dda"] = np.nan

# #     # 4) HNR (Harmonics-to-Noise Ratio) approximate: using librosa.effects.harmonic+percussive
# #     try:
# #         harmonic = librosa.effects.harmonic(y)
# #         noise = y - harmonic
# #         hnr_val = 10 * np.log10((np.mean(harmonic ** 2) + 1e-12) / (np.mean(noise ** 2) + 1e-12))
# #         features["hnr"] = float(hnr_val)
# #     except Exception:
# #         features["hnr"] = np.nan

# #     # 5) MFCCs 0..11 (mfcc0..mfcc11)
# #     try:
# #         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12, hop_length=512)
# #         mfcc_means = np.mean(mfccs, axis=1)
# #         for idx in range(12):
# #             features[f"mfcc{idx}"] = float(mfcc_means[idx]) if idx < len(mfcc_means) else np.nan
# #     except Exception:
# #         for idx in range(12):
# #             features[f"mfcc{idx}"] = np.nan

# #     # 6) mel means 0..39 and mel stds 0..39 (mel_mean0..mel_mean39, mel_std0..mel_std39)
# #     try:
# #         mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, hop_length=512)
# #         mel_db = librosa.power_to_db(mel, ref=np.max)
# #         mel_means = np.mean(mel_db, axis=1)
# #         mel_stds = np.std(mel_db, axis=1)
# #         for idx in range(40):
# #             features[f"mel_mean{idx}"] = float(mel_means[idx]) if idx < len(mel_means) else np.nan
# #             features[f"mel_std{idx}"] = float(mel_stds[idx]) if idx < len(mel_stds) else np.nan
# #     except Exception:
# #         for idx in range(40):
# #             features[f"mel_mean{idx}"] = np.nan
# #             features[f"mel_std{idx}"] = np.nan

# #     return features


# # def audio_to_chunks(file_like, chunk_duration=CHUNK_DURATION, sr=SR):
# #     # librosa can accept a file-like object if we pass the buffer
# #     try:
# #         data, sr = librosa.load(file_like, sr=sr)
# #         data = nr.reduce_noise(y=data, sr=sr)
# #         chunk_len = int(chunk_duration * sr)
# #         chunks = [data[i:i + chunk_len] for i in range(0, len(data), chunk_len) if len(data[i:i + chunk_len]) == chunk_len]
# #         return chunks, sr
# #     except Exception as e:
# #         st.error(f"Error processing audio: {e}")
# #         return [], sr

# # def chunk_to_spectrogram_buf(chunk, sr):
# #     S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, hop_length=512)
# #     S_db = librosa.power_to_db(S, ref=np.max)
# #     fig, ax = plt.subplots(figsize=(3,3))
# #     librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel", ax=ax)
# #     plt.axis("off")
# #     buf = io.BytesIO()
# #     plt.savefig(buf, bbox_inches="tight", pad_inches=0, format="png")
# #     plt.close(fig)
# #     buf.seek(0)
# #     return buf

# # def preprocess_spectrogram(img_buf):
# #     try:
# #         img = image.load_img(img_buf, target_size=IMG_SIZE)
# #         img_array = image.img_to_array(img) / 255.0
# #         img_array = np.expand_dims(img_array, axis=0)
# #         return img_array
# #     except Exception as e:
# #         st.error(f"Error preprocessing spectrogram: {e}")
# #         return None

# # def generate_confusion_plot(y_true, y_pred, labels=["Healthy","Parkinson's"]):
# #     cm = confusion_matrix(y_true, y_pred)
# #     fig, ax = plt.subplots(figsize=(4,4))
# #     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
# #     ax.set_xlabel("Predicted")
# #     ax.set_ylabel("True")
# #     st.pyplot(fig)

# # # ---------------------------
# # # Feature column order requested by user
# # # ---------------------------
# # FEATURE_COLUMNS = [
# #     "meanF0Hz","maxF0Hz","minF0Hz","stdF0Hz",
# #     "jitter_local","jitter_abs","jitter_rap","jitter_ddp","jitter_ppq5",
# #     "shimmer_local","shimmer_db","shimmer_apq3","shimmer_apq5","shimmer_dda",
# #     "hnr"
# # ] + [f"mfcc{i}" for i in range(12)] + [f"mel_mean{i}" for i in range(40)] + [f"mel_std{i}" for i in range(40)]

# # # ---------------------------
# # # Model loader with caching
# # # ---------------------------
# # @st.cache_resource
# # def load_all_models(model_dir):
# #     # file mapping - only load files that exist
# #     mapping = {
# #         "bilstm": "best_bilstm.keras",
# #         "cnn_lstm": "best_cnn_lstm_model.keras",
# #         "resnet50": "best_resnet50.keras",
# #         "gru": "best_gru.keras",
# #         "mobilenetv2": "best_mobilenetv2.keras",   # optional
# #         "cnn": "best_CNN.keras"                  # optional
# #     }
# #     loaded = {}
# #     errors = {}
# #     for key, fname in mapping.items():
# #         path = os.path.join(model_dir, fname)
# #         if os.path.exists(path):
# #             try:
# #                 loaded[key] = load_model(path)
# #             except Exception as e:
# #                 errors[key] = str(e)
# #         else:
# #             errors[key] = f"file not found: {path}"
# #     return loaded, errors

# # # ---------------------------
# # # UI Sidebar
# # # ---------------------------
# # st.sidebar.title("Navigation")
# # view = st.sidebar.radio("Go to", ["Test & Predict", "Model Metrics", "Performance Graphs"])

# # # Load models once (cached)
# # with st.spinner("Loading models..."):
# #     MODELS, MODEL_ERRORS = load_all_models(MODEL_DIR)

# # if MODEL_ERRORS:
# #     # present errors in the sidebar for debugging
# #     st.sidebar.markdown("**Model load status**")
# #     for k, err in MODEL_ERRORS.items():
# #         st.sidebar.text(f"{k}: {err}")

# # # prepare static DL model results (user-provided) as a DataFrame for Model Metrics page
# # dl_models_results = [
# #   { "Model": "CNN", "Accuracy": 0.93, "Precision": 0.92, "Recall": 0.95, "F1 Score": 0.93, "Method": "Spectrogram" },
# #   { "Model": "LSTM", "Accuracy": 0.844, "Precision": 0.84, "Recall": 0.85, "F1 Score": 0.844, "Method": "Spectrogram" },
# #   { "Model": "GRU", "Accuracy": 0.796, "Precision": 0.80, "Recall": 0.79, "F1 Score": 0.795, "Method": "Spectrogram" },
# #   { "Model": "BiLSTM", "Accuracy": 0.77, "Precision": 0.82, "Recall": 0.71, "F1 Score": 0.76, "Method": "Spectrogram" },
# #   { "Model": "ResNet50", "Accuracy": 0.76, "Precision": 0.71, "Recall": 0.88, "F1 Score": 0.78, "Method": "Spectrogram" },
# # ]
# # dl_models_df = pd.DataFrame(dl_models_results)

# # # ---------------------------
# # # Main: Test & Predict
# # # ---------------------------
# # if view == "Test & Predict":
# #     st.header("Upload a WAV audio file for prediction")
# #     uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

# #     if uploaded_file:
# #         # store bytes in session so it persists while navigating pages
# #         st.session_state.audio_bytes = uploaded_file.read()
# #         st.audio(st.session_state.audio_bytes)

# #         # Extract features for the whole file (not per-chunk) and display in table
# #         y, sr = safe_load_audio(io.BytesIO(st.session_state.audio_bytes), sr=SR)
# #         if y is not None:
# #             y_nr = reduce_noise(y, sr)
# #             extracted = extract_basic_features_from_audio(y_nr, sr)
# #             # ensure column order and presence
# #             row = {col: extracted.get(col, np.nan) for col in FEATURE_COLUMNS}
# #             features_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
# #             st.subheader("Extracted features (first row)")
# #             st.dataframe(features_df.T.rename(columns={0: "Value"}))  # transpose to show feature -> value
# #             # store in session to show on Model Metrics later
# #             st.session_state.extracted_features_df = features_df

# #         # Do chunk/prediction logic (use first chunk) but do NOT display spectrogram
# #         chunks, sr_chunks = audio_to_chunks(io.BytesIO(st.session_state.audio_bytes))
# #         st.write(f"Audio split into {len(chunks)} chunk(s) of {CHUNK_DURATION}s (used for model predictions)")
# #         if len(chunks) == 0:
# #             st.warning("No full-length chunks (3s) were found. Try a longer audio or reduce CHUNK_DURATION.")
# #         else:
# #             # create spectrogram buffer internally (for model input) but do not show it
# #             first_chunk = chunks[0]
# #             img_buf = chunk_to_spectrogram_buf(first_chunk, sr_chunks)
# #             x = preprocess_spectrogram(io.BytesIO(img_buf.getvalue()))
# #             if x is not None:
# #                 results = {}
# #                 for name, model in MODELS.items():
# #                     try:
# #                         out = model.predict(x, verbose=0)
# #                         score = None
# #                         if isinstance(out, np.ndarray):
# #                             if out.shape[-1] == 1:
# #                                 score = float(out[0][0])
# #                             else:
# #                                 score = float(out[0][1]) if out.shape[-1] > 1 else float(out[0][0])
# #                         else:
# #                             score = float(out)
# #                         label = "Parkinson's Disease (PD)" if (score is not None and score >= 0.5) else "Healthy Control (HC)"
# #                         results[name] = {"score": score, "label": label}
# #                     except Exception as e:
# #                         results[name] = {"score": None, "label": f"ERROR: {e}"}

# #                 # Voting summary
# #                 pd_votes = sum(1 for r in results.values() if r["label"].startswith("Parkinson"))
# #                 hc_votes = sum(1 for r in results.values() if r["label"].startswith("Healthy"))
# #                 final_label = "Parkinson's Disease (PD)" if pd_votes > hc_votes else "Healthy Control (HC)"
# #                 st.subheader("Voting Result")
# #                 st.write(f"Final: **{final_label}** (PD votes: {pd_votes}, HC votes: {hc_votes})")

# #                 # show model-wise predictions in a neat table
# #                 preds_df = pd.DataFrame([
# #                     {"Model": k, "Prediction": v["label"], "Score": (v["score"] if v["score"] is not None else np.nan)}
# #                     for k, v in results.items()
# #                 ])
# #                 st.subheader("Individual model predictions")
# #                 st.dataframe(preds_df.set_index("Model"))

# # elif view == "Model Metrics":
# #     st.header("Model Metrics & Extracted Features")
# #     # top: DL models table (user-specified)
# #     st.subheader("Deep Learning Models Results")
# #     st.dataframe(dl_models_df)

# #     st.markdown("---")
# #     # show the extracted features if present in session, else show placeholder
# #     st.subheader("Extracted Features (from last uploaded audio)")
# #     if "extracted_features_df" in st.session_state:
# #         # show as a wide table (single-row), display columns horizontally
# #         # We'll present as a single-row table with the requested column ordering.
# #         ef = st.session_state.extracted_features_df.copy()
# #         # round numeric columns for display
# #         ef_disp = ef.round(6)
# #         st.dataframe(ef_disp)
# #     else:
# #         st.info("No extracted features found in session. Upload an audio file on the 'Test & Predict' page to extract features.")

# #     st.markdown("---")
# #     st.subheader("Available models")
# #     for m in MODELS.keys():
# #         st.write(f"- {m}")

# #     if MODEL_ERRORS:
# #         st.warning("Some models were not loaded. Check the sidebar for errors.")

# # elif view == "Performance Graphs":
# #     st.header("Local performance graphs")
# #     names = ["CNN-LSTM.png", "BiLSTM.png", "GRU.png", "MobileNetV2.png", "ResNet50.png"]
# #     cols = st.columns(2)
# #     for i, fname in enumerate(names):
# #         path = os.path.join(PLOTS_DIR, fname)
# #         try:
# #             if os.path.exists(path):
# #                 img = Image.open(path)
# #                 with cols[i % 2]:
# #                     st.image(img, caption=fname, use_column_width=True)
# #             else:
# #                 with cols[i % 2]:
# #                     st.info(f"{fname} not found in {PLOTS_DIR}")
# #         except Exception as e:
# #             st.error(f"Error loading {fname}: {e}")
# import streamlit as st
# import os, io, math
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import noisereduce as nr
# import librosa
# import librosa.display
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# from PIL import Image

# IMG_SIZE = (128, 128)
# CHUNK_DURATION = 3.0
# SR = 22050

# st.set_page_config(layout="wide", page_title="Parkinson's Prediction Dashboard")

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, "models")
# PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# def safe_load_audio(file_like, sr=SR):
#     try:
#         y, sr_ret = librosa.load(file_like, sr=sr)
#         return y, sr_ret
#     except:
#         return None, sr

# def reduce_noise(y, sr):
#     try:
#         return nr.reduce_noise(y=y, sr=sr)
#     except:
#         return y

# def extract_basic_features_from_audio(y, sr):
#     features = {}
#     if y is None or len(y) == 0:
#         return {k: np.nan for k in FEATURE_COLUMNS}

#     try:
#         f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
#         voiced_f0 = f0[~np.isnan(f0)]
#     except:
#         voiced_f0 = np.array([])

#     if voiced_f0.size > 0:
#         features["meanF0Hz"] = float(np.mean(voiced_f0))
#         features["maxF0Hz"] = float(np.max(voiced_f0))
#         features["minF0Hz"] = float(np.min(voiced_f0))
#         features["stdF0Hz"] = float(np.std(voiced_f0))
#     else:
#         for k in ["meanF0Hz","maxF0Hz","minF0Hz","stdF0Hz"]:
#             features[k] = np.nan

#     try:
#         harmonic = librosa.effects.harmonic(y)
#         noise = y - harmonic
#         features["hnr"] = float(10 * np.log10((np.mean(harmonic ** 2) + 1e-12)/(np.mean(noise ** 2) + 1e-12)))
#     except:
#         features["hnr"] = np.nan

#     try:
#         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
#         mfcc_means = np.mean(mfccs, axis=1)
#         for i in range(12):
#             features[f"mfcc{i}"] = float(mfcc_means[i])
#     except:
#         for i in range(12):
#             features[f"mfcc{i}"] = np.nan

#     try:
#         mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
#         mel_db = librosa.power_to_db(mel, ref=np.max)
#         mel_means = np.mean(mel_db, axis=1)
#         mel_stds = np.std(mel_db, axis=1)
#         for i in range(40):
#             features[f"mel_mean{i}"] = float(mel_means[i])
#             features[f"mel_std{i}"] = float(mel_stds[i])
#     except:
#         for i in range(40):
#             features[f"mel_mean{i}"] = np.nan
#             features[f"mel_std{i}"] = np.nan

#     return features

# def audio_to_chunks(file_like, chunk_duration=CHUNK_DURATION, sr=SR):
#     y, sr = librosa.load(file_like, sr=sr)
#     y = reduce_noise(y, sr)
#     chunk_len = int(chunk_duration * sr)
#     chunks = [y[i:i + chunk_len] for i in range(0, len(y), chunk_len) if len(y[i:i + chunk_len]) == chunk_len]
#     return chunks, sr

# def chunk_to_spectrogram_buf(chunk, sr):
#     S = librosa.feature.melspectrogram(y=chunk, sr=sr)
#     S_db = librosa.power_to_db(S, ref=np.max)
#     fig, ax = plt.subplots(figsize=(3,3))
#     librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None, ax=ax)
#     plt.axis("off")
#     buf = io.BytesIO()
#     plt.savefig(buf, bbox_inches="tight", pad_inches=0, format="png")
#     plt.close(fig)
#     buf.seek(0)
#     return buf

# def preprocess_spectrogram(img_buf):
#     img = image.load_img(img_buf, target_size=IMG_SIZE)
#     img_array = image.img_to_array(img) / 255.0
#     return np.expand_dims(img_array, axis=0)

# FEATURE_COLUMNS = ["meanF0Hz","maxF0Hz","minF0Hz","stdF0Hz","hnr"] \
#                   + [f"mfcc{i}" for i in range(12)] \
#                   + [f"mel_mean{i}" for i in range(40)] \
#                   + [f"mel_std{i}" for i in range(40)]

# @st.cache_resource
# def load_all_models(model_dir):
#     mapping = {
#         "bilstm": "best_bilstm.keras",
#         "cnn_lstm": "best_cnn_lstm_model.keras",
#         "resnet50": "best_resnet50.keras",
#         "gru": "best_gru.keras",
#         "mobilenetv2": "best_mobilenetv2.keras",
#         "cnn": "best_CNN.keras"
#     }
#     loaded, errors = {}, {}
#     for key, fname in mapping.items():
#         path = os.path.join(model_dir, fname)
#         if os.path.exists(path):
#             try:
#                 loaded[key] = load_model(path)
#             except Exception as e:
#                 errors[key] = str(e)
#         else:
#             errors[key] = f"Missing: {fname}"
#     return loaded, errors

# st.sidebar.title("Navigation")
# view = st.sidebar.radio("Go to", ["Test & Predict", "Model Metrics", "Performance Graphs"])

# MODELS, _ = load_all_models(MODEL_DIR)

# if view == "Test & Predict":
#     st.header("Upload a WAV audio file for prediction")
#     up = st.file_uploader("Choose a file", type=["wav"])

#     if up:
#         audio_bytes = up.read()
#         st.audio(audio_bytes)

#         y, sr = safe_load_audio(io.BytesIO(audio_bytes))
#         if y is not None:
#             extracted = extract_basic_features_from_audio(y, sr)
#             df_feats = pd.DataFrame([{c: extracted.get(c, np.nan) for c in FEATURE_COLUMNS}])
#             st.subheader("Extracted Features")
#             st.dataframe(df_feats.T.rename(columns={0: "Value"}))

#         chunks, sr = audio_to_chunks(io.BytesIO(audio_bytes))
#         if len(chunks) == 0:
#             st.warning("No 3-second chunk found.")
#         else:
#             first_chunk = chunks[0]
#             img_buf = chunk_to_spectrogram_buf(first_chunk, sr)
#             x = preprocess_spectrogram(io.BytesIO(img_buf.getvalue()))

#             results = {}
#             for name, model in MODELS.items():
#                 pred = model.predict(x, verbose=0)
#                 score = float(pred[0][1]) if pred.shape[-1] > 1 else float(pred[0][0])
#                 label = "Parkinson's Disease" if score >= 0.5 else "Healthy Control"
#                 results[name] = {"Score": score, "Prediction": label}

#             st.subheader("Model Predictions")
#             st.table(pd.DataFrame(results).T)

#             votes = [v["Prediction"] for v in results.values()]
#             final = max(set(votes), key=votes.count)
#             st.success(f"🧠 Final Decision: **{final}**")

# if view == "Model Metrics":
#     st.header("Model Classification Report")
#     metrics = [
#         {"Model":"CNN","Accuracy":0.93,"Precision":0.92,"Recall":0.95,"F1 Score":0.93},
#         {"Model":"LSTM","Accuracy":0.844,"Precision":0.84,"Recall":0.85,"F1 Score":0.844},
#         {"Model":"GRU","Accuracy":0.796,"Precision":0.80,"Recall":0.79,"F1 Score":0.795},
#         {"Model":"BiLSTM","Accuracy":0.77,"Precision":0.82,"Recall":0.71,"F1 Score":0.76},
#         {"Model":"ResNet50","Accuracy":0.76,"Precision":0.71,"Recall":0.88,"F1 Score":0.78},
#     ]
#     st.dataframe(pd.DataFrame(metrics))

# if view == "Performance Graphs":
#     st.header("Uploaded Performance Graphs")
#     img_files = [f for f in os.listdir(PLOTS_DIR) if f.lower().endswith(("png","jpg","jpeg"))]

#     for file in img_files:
#         st.image(os.path.join(PLOTS_DIR, file), caption=file)
# # --- Continue from Performance Graphs page ---
# if view == "Performance Graphs":
#     st.header("📊 Model Performance Graphs")
#     st.write("Below are the uploaded loss/accuracy and confusion matrix graphs for each model.")

#     if not os.path.exists(PLOTS_DIR):
#         st.error("❌ 'plots' folder not found. Please create a folder named **plots/** and place your images inside.")
#     else:
#         img_files = [f for f in os.listdir(PLOTS_DIR) if f.lower().endswith(("png", "jpg", "jpeg"))]

#         if len(img_files) == 0:
#             st.warning("⚠ No images found inside the `plots/` directory.")
#         else:
#             cols = st.columns(2)
#             for idx, file in enumerate(img_files):
#                 img_path = os.path.join(PLOTS_DIR, file)
#                 try:
#                     img = Image.open(img_path)
#                     with cols[idx % 2]:
#                         st.image(img, caption=file, use_container_width=True)
#                 except:
#                     st.error(f"Cannot open image: {file}")

# # -----------------------------------------------------------
# # Footer message
# # -----------------------------------------------------------
# st.markdown("""
# <hr style="border: 1px solid #d9d9d9;">
# <div style="text-align:center; color: gray;">
#     Parkinson's Voice Diagnostic Platform • Powered by Deep Learning 🔬
# </div>
# """, unsafe_allow_html=True)


# # # # a.py
# # # import streamlit as st
# # # import os
# # # import io
# # # import numpy as np
# # # import librosa
# # # import librosa.display
# # # import matplotlib.pyplot as plt
# # # import noisereduce as nr
# # # from tensorflow.keras.models import load_model
# # # from tensorflow.keras.preprocessing import image
# # # from sklearn.metrics import confusion_matrix
# # # import seaborn as sns
# # # from PIL import Image

# # # st.set_page_config(layout="wide", page_title="Parkinson's Prediction Dashboard")

# # # # ---------------------------
# # # # Paths (adjust if your structure differs)
# # # # ---------------------------
# # # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # # MODEL_DIR = os.path.join(BASE_DIR, "models")      # your models folder
# # # PLOTS_DIR = os.path.join(BASE_DIR, "plots")       # your images folder

# # # IMG_SIZE = (128, 128)
# # # CHUNK_DURATION = 3.0
# # # SR = 22050

# # # # ---------------------------
# # # # Helper functions
# # # # ---------------------------
# # # def audio_to_chunks(file_like, chunk_duration=CHUNK_DURATION, sr=SR):
# # #     # librosa can accept a file-like object if we pass the buffer
# # #     try:
# # #         data, sr = librosa.load(file_like, sr=sr)
# # #         data = nr.reduce_noise(y=data, sr=sr)
# # #         chunk_len = int(chunk_duration * sr)
# # #         chunks = [data[i:i + chunk_len] for i in range(0, len(data), chunk_len) if len(data[i:i + chunk_len]) == chunk_len]
# # #         return chunks, sr
# # #     except Exception as e:
# # #         st.error(f"Error processing audio: {e}")
# # #         return [], sr

# # # def chunk_to_spectrogram_buf(chunk, sr):
# # #     S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, hop_length=512)
# # #     S_db = librosa.power_to_db(S, ref=np.max)
# # #     fig, ax = plt.subplots(figsize=(3,3))
# # #     librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel", ax=ax)
# # #     plt.axis("off")
# # #     buf = io.BytesIO()
# # #     plt.savefig(buf, bbox_inches="tight", pad_inches=0, format="png")
# # #     plt.close(fig)
# # #     buf.seek(0)
# # #     return buf

# # # def preprocess_spectrogram(img_buf):
# # #     try:
# # #         # Keras image loader can read a file-like object
# # #         img = image.load_img(img_buf, target_size=IMG_SIZE)
# # #         arr = image.img_to_array(img) / 255.0
# # #         arr = np.expand_dims(arr, axis=0)
# # #         return arr
# # #     except Exception as e:
# # #         st.error(f"Error preprocessing spectrogram: {e}")
# # #         return None

# # # def generate_confusion_plot(y_true, y_pred, labels=["Healthy","Parkinson's"]):
# # #     cm = confusion_matrix(y_true, y_pred)
# # #     fig, ax = plt.subplots(figsize=(4,4))
# # #     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
# # #     ax.set_xlabel("Predicted")
# # #     ax.set_ylabel("True")
# # #     st.pyplot(fig)

# # # # ---------------------------
# # # # Model loader with caching
# # # # ---------------------------
# # # @st.cache_resource
# # # def load_all_models(model_dir):
# # #     models = {}
# # #     # Map file names (or keys) to expected filenames
# # #     mapping = {
# # #         "bilstm": "best_bilstm.keras",
# # #         "cnn_lstm": "best_cnn_lstm_model.keras",
# # #         "resnet50": "best_resnet50.keras",
# # #         "gru": "best_gru.keras",
# # #         "mobilenetv2": "best_mobilenetv2.keras",   # optional
# # #         "cnn": "best_CNN.keras"                  # optional
# # #     }
# # #     loaded = {}
# # #     errors = {}
# # #     for key, fname in mapping.items():
# # #         path = os.path.join(model_dir, fname)
# # #         if os.path.exists(path):
# # #             try:
# # #                 loaded[key] = load_model(path)
# # #             except Exception as e:
# # #                 errors[key] = str(e)
# # #         else:
# # #             errors[key] = f"file not found: {path}"
# # #     return loaded, errors

# # # # ---------------------------
# # # # UI Sidebar
# # # # ---------------------------
# # # st.sidebar.title("Navigation")
# # # view = st.sidebar.radio("Go to", ["Test & Predict", "Model Metrics", "Performance Graphs"])

# # # # Load models once (cached)
# # # with st.spinner("Loading models..."):
# # #     MODELS, MODEL_ERRORS = load_all_models(MODEL_DIR)

# # # if MODEL_ERRORS:
# # #     # show errors (but not blocking)
# # #     for k, err in MODEL_ERRORS.items():
# # #         st.sidebar.write(f"{k}: {err}")

# # # if view == "Test & Predict":
# # #     st.header("Upload a WAV audio file for prediction")
# # #     uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

# # #     if uploaded_file:
# # #         if "audio_bytes" not in st.session_state:
# # #             st.session_state.audio_bytes = uploaded_file.read()
# # #         # show playback
# # #         st.audio(st.session_state.audio_bytes)

# # #         # process audio from bytes
# # #         chunks, sr = audio_to_chunks(io.BytesIO(st.session_state.audio_bytes))
# # #         st.write(f"Audio split into {len(chunks)} chunk(s) of {CHUNK_DURATION}s")
# # #         if len(chunks) == 0:
# # #             st.warning("No full-length chunks (3s) were found. Try a longer audio or reduce CHUNK_DURATION.")
# # #         else:
# # #             # take first chunk for preview/predictions
# # #             chunk = chunks[0]
# # #             img_buf = chunk_to_spectrogram_buf(chunk, sr)
# # #             st.image(img_buf, caption="Spectrogram (first chunk)", use_column_width=False)

# # #             x = preprocess_spectrogram(io.BytesIO(img_buf.getvalue()))
# # #             if x is not None:
# # #                 results = {}
# # #                 for name, model in MODELS.items():
# # #                     try:
# # #                         # models may return logits or probabilities; adapt if shape differs
# # #                         out = model.predict(x, verbose=0)
# # #                         # attempt common scenarios:
# # #                         score = None
# # #                         if isinstance(out, np.ndarray):
# # #                             # binary single-output
# # #                             if out.shape[-1] == 1:
# # #                                 score = float(out[0][0])
# # #                             else:
# # #                                 # multiclass: take probability of class 1
# # #                                 score = float(out[0][1]) if out.shape[-1] > 1 else float(out[0][0])
# # #                         else:
# # #                             score = float(out)
# # #                         label = "Parkinson's Disease (PD)" if score >= 0.5 else "Healthy Control (HC)"
# # #                         results[name] = {"score": score, "label": label}
# # #                     except Exception as e:
# # #                         results[name] = {"score": None, "label": f"ERROR: {e}"}

# # #                 # Voting summary
# # #                 pd_votes = sum(1 for r in results.values() if r["label"].startswith("Parkinson"))
# # #                 hc_votes = sum(1 for r in results.values() if r["label"].startswith("Healthy"))
# # #                 final_label = "Parkinson's Disease (PD)" if pd_votes > hc_votes else "Healthy Control (HC)"
# # #                 st.subheader("Voting Result")
# # #                 st.write(f"Final: **{final_label}** (PD votes: {pd_votes}, HC votes: {hc_votes})")

# # #                 st.subheader("Model-wise predictions")
# # #                 for k, v in results.items():
# # #                     st.write(f"- **{k}**: {v['label']} (score={v['score']})")

# # #                 # example confusion matrix (for demo) - replace with your real arrays if available
# # #                 if st.button("Show example confusion matrix"):
# # #                     y_true = np.array([0,0,1,1,0,1,0,1,1,0])
# # #                     y_pred = np.array([0,1,1,1,0,1,0,0,1,0])
# # #                     generate_confusion_plot(y_true, y_pred)

# # # elif view == "Model Metrics":
# # #     st.header("Model metrics and tables")
# # #     # you can render tables (hard-coded or loaded)
# # #     st.markdown("**Available models**")
# # #     for m in MODELS.keys():
# # #         st.write(f"- {m}")

# # #     if MODEL_ERRORS:
# # #         st.warning("Some models were not loaded. Check the sidebar for errors.")

# # #     # show first 5 rows of example CSV or feature lists if you have them
# # #     st.markdown("Feature lists (example)")
# # #     total_features = ["voiceID", "meanF0Hz", "maxF0Hz", "minF0Hz", "stdF0Hz", "jitter_local", "shimmer_local", "hnr", "label"]
# # #     st.write(total_features[:10])

# # # elif view == "Performance Graphs":
# # #     st.header("Local performance graphs")
# # #     names = ["CNN-LSTM.png", "BiLSTM.png", "GRU.png", "MobileNetV2.png", "ResNet50.png"]
# # #     cols = st.columns(2)
# # #     for i, fname in enumerate(names):
# # #         path = os.path.join(PLOTS_DIR, fname)
# # #         try:
# # #             if os.path.exists(path):
# # #                 img = Image.open(path)
# # #                 with cols[i % 2]:
# # #                     st.image(img, caption=fname, use_column_width=True)
# # #             else:
# # #                 with cols[i % 2]:
# # #                     st.info(f"{fname} not found in {PLOTS_DIR}")
# # #         except Exception as e:
# # #             st.error(f"Error loading {fname}: {e}")
# # # a.py
# # import streamlit as st
# # import os, io, math
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import noisereduce as nr
# # import librosa
# # import librosa.display
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.preprocessing import image
# # from sklearn.metrics import confusion_matrix
# # import seaborn as sns
# # from PIL import Image

# # # constants (must be defined before functions that use them)
# # IMG_SIZE = (128, 128)
# # CHUNK_DURATION = 3.0
# # SR = 22050

# # def safe_load_audio(file_like, sr=SR):
# #     try:
# #         y, sr_ret = librosa.load(file_like, sr=sr)
# #         return y, sr_ret
# #     except Exception as e:
# #         st.error(f"Error loading audio: {e}")
# #         return None, sr


# # # --- page config
# # st.set_page_config(layout="wide", page_title="Parkinson's Prediction Dashboard")

# # # ---------------------------
# # # Paths (adjust if your structure differs)
# # # ---------------------------
# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # MODEL_DIR = os.path.join(BASE_DIR, "models")      # your models folder
# # PLOTS_DIR = os.path.join(BASE_DIR, "plots")       # your images folder

# # IMG_SIZE = (128, 128)
# # CHUNK_DURATION = 3.0
# # SR = 22050

# # # ---------------------------
# # # Helper functions (feature extraction, spectrogram preprocess, etc.)
# # # ---------------------------

# # def safe_load_audio(file_like, sr=SR):
# #     try:
# #         y, sr_ret = librosa.load(file_like, sr=sr)
# #         return y, sr_ret
# #     except Exception as e:
# #         st.error(f"Error loading audio: {e}")
# #         return None, sr

# # def reduce_noise(y, sr):
# #     try:
# #         return nr.reduce_noise(y=y, sr=sr)
# #     except Exception:
# #         return y

# # def extract_basic_features_from_audio(y, sr):
# #     """
# #     Best-effort extraction for the requested feature set.
# #     Many of the features (jitter, shimmer, HNR) are approximated.
# #     Returns a dict with feature_name -> value
# #     """
# #     features = {}
# #     # safe guards
# #     if y is None or len(y) == 0:
# #         return {k: np.nan for k in FEATURE_COLUMNS}

# #     # 1) pitch (fundamental freq) using librosa.pyin (if available) fallback to yin
# #     f0 = None
# #     try:
# #         # pyin provides voiced F0 with n_frames; use median/mean over voiced frames
# #         f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
# #         # f0 is array with np.nan for unvoiced; filter
# #         voiced_f0 = f0[~np.isnan(f0)]
# #         if voiced_f0.size == 0:
# #             raise Exception("no voiced frames from pyin")
# #     except Exception:
# #         try:
# #             # fallback to yin
# #             f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
# #             voiced_f0 = f0[~np.isnan(f0)]
# #             if voiced_f0.size == 0:
# #                 voiced_f0 = np.array([])
# #         except Exception:
# #             voiced_f0 = np.array([])

# #     # compute mean/max/min/std for F0
# #     if voiced_f0.size > 0:
# #         features["meanF0Hz"] = float(np.mean(voiced_f0))
# #         features["maxF0Hz"] = float(np.max(voiced_f0))
# #         features["minF0Hz"] = float(np.min(voiced_f0))
# #         features["stdF0Hz"] = float(np.std(voiced_f0))
# #     else:
# #         features["meanF0Hz"] = np.nan
# #         features["maxF0Hz"] = np.nan
# #         features["minF0Hz"] = np.nan
# #         features["stdF0Hz"] = np.nan

# #     # 2) Jitter approximations:
# #     # jitter_local: average absolute difference between consecutive voiced F0 divided by mean f0
# #     try:
# #         if voiced_f0.size > 1 and not math.isnan(features["meanF0Hz"]):
# #             jitter_vals = np.abs(np.diff(voiced_f0))
# #             features["jitter_local"] = float(np.mean(jitter_vals) / (features["meanF0Hz"] + 1e-12))
# #             features["jitter_abs"] = float(np.mean(jitter_vals))
# #             # rap, ddp, ppq5 are more complex; approximate with rolling-window normalized diffs
# #             # rap: relative average perturbation (3-sample)
# #             if voiced_f0.size >= 3:
# #                 rap = np.mean(np.abs((voiced_f0[1:-1] - (voiced_f0[:-2] + voiced_f0[2:]) / 2.0)))
# #                 features["jitter_rap"] = float(rap / (features["meanF0Hz"] + 1e-12))
# #             else:
# #                 features["jitter_rap"] = np.nan
# #             # ddp approx (difference of differences)
# #             if voiced_f0.size >= 5:
# #                 ddp = np.mean(np.abs(np.diff(voiced_f0, n=2)))
# #                 features["jitter_ddp"] = float(ddp / (features["meanF0Hz"] + 1e-12))
# #             else:
# #                 features["jitter_ddp"] = np.nan
# #             # ppq5: average absolute difference with average over window 5
# #             if voiced_f0.size >= 5:
# #                 ppq5_vals = []
# #                 for i in range(2, len(voiced_f0)-2):
# #                     ppq5_vals.append(abs(voiced_f0[i] - np.mean(voiced_f0[i-2:i+3])))
# #                 features["jitter_ppq5"] = float(np.mean(ppq5_vals) / (features["meanF0Hz"] + 1e-12))
# #             else:
# #                 features["jitter_ppq5"] = np.nan
# #         else:
# #             features["jitter_local"] = np.nan
# #             features["jitter_abs"] = np.nan
# #             features["jitter_rap"] = np.nan
# #             features["jitter_ddp"] = np.nan
# #             features["jitter_ppq5"] = np.nan
# #     except Exception:
# #         features["jitter_local"] = np.nan
# #         features["jitter_abs"] = np.nan
# #         features["jitter_rap"] = np.nan
# #         features["jitter_ddp"] = np.nan
# #         features["jitter_ppq5"] = np.nan

# #     # 3) Shimmer approximations:
# #     # shimmer_local: amplitude perturbation - use short-time energy or RMS frame-by-frame
# #     try:
# #         hop_length = 512
# #         frame_length = 1024
# #         rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
# #         if rms.size > 1:
# #             amp_diff = np.abs(np.diff(rms))
# #             features["shimmer_local"] = float(np.mean(amp_diff) / (np.mean(rms) + 1e-12))
# #             features["shimmer_db"] = float(20 * np.log10(np.mean(rms) + 1e-12))
# #             # apq3, apq5, dda approximate similar to jitter but on amplitude frames
# #             if rms.size >= 3:
# #                 apq3 = np.mean(np.abs(rms[1:-1] - (rms[:-2] + rms[2:]) / 2.0))
# #                 features["shimmer_apq3"] = float(apq3 / (np.mean(rms) + 1e-12))
# #             else:
# #                 features["shimmer_apq3"] = np.nan
# #             if rms.size >= 5:
# #                 apq5_vals = []
# #                 for i in range(2, len(rms)-2):
# #                     apq5_vals.append(abs(rms[i] - np.mean(rms[i-2:i+3])))
# #                 features["shimmer_apq5"] = float(np.mean(apq5_vals) / (np.mean(rms) + 1e-12))
# #             else:
# #                 features["shimmer_apq5"] = np.nan
# #             if rms.size >= 2:
# #                 dda = np.mean(np.abs(np.diff(rms)))
# #                 features["shimmer_dda"] = float(dda / (np.mean(rms) + 1e-12))
# #             else:
# #                 features["shimmer_dda"] = np.nan
# #         else:
# #             features["shimmer_local"] = np.nan
# #             features["shimmer_db"] = np.nan
# #             features["shimmer_apq3"] = np.nan
# #             features["shimmer_apq5"] = np.nan
# #             features["shimmer_dda"] = np.nan
# #     except Exception:
# #         features["shimmer_local"] = np.nan
# #         features["shimmer_db"] = np.nan
# #         features["shimmer_apq3"] = np.nan
# #         features["shimmer_apq5"] = np.nan
# #         features["shimmer_dda"] = np.nan

# #     # 4) HNR (Harmonics-to-Noise Ratio) approximate: using librosa.effects.harmonic+percussive
# #     try:
# #         harmonic = librosa.effects.harmonic(y)
# #         noise = y - harmonic
# #         hnr_val = 10 * np.log10((np.mean(harmonic ** 2) + 1e-12) / (np.mean(noise ** 2) + 1e-12))
# #         features["hnr"] = float(hnr_val)
# #     except Exception:
# #         features["hnr"] = np.nan

# #     # 5) MFCCs 0..11 (mfcc0..mfcc11)
# #     try:
# #         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12, hop_length=512)
# #         mfcc_means = np.mean(mfccs, axis=1)
# #         for idx in range(12):
# #             features[f"mfcc{idx}"] = float(mfcc_means[idx]) if idx < len(mfcc_means) else np.nan
# #     except Exception:
# #         for idx in range(12):
# #             features[f"mfcc{idx}"] = np.nan

# #     # 6) mel means 0..39 and mel stds 0..39 (mel_mean0..mel_mean39, mel_std0..mel_std39)
# #     try:
# #         mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, hop_length=512)
# #         mel_db = librosa.power_to_db(mel, ref=np.max)
# #         mel_means = np.mean(mel_db, axis=1)
# #         mel_stds = np.std(mel_db, axis=1)
# #         for idx in range(40):
# #             features[f"mel_mean{idx}"] = float(mel_means[idx]) if idx < len(mel_means) else np.nan
# #             features[f"mel_std{idx}"] = float(mel_stds[idx]) if idx < len(mel_stds) else np.nan
# #     except Exception:
# #         for idx in range(40):
# #             features[f"mel_mean{idx}"] = np.nan
# #             features[f"mel_std{idx}"] = np.nan

# #     return features


# # def audio_to_chunks(file_like, chunk_duration=CHUNK_DURATION, sr=SR):
# #     # librosa can accept a file-like object if we pass the buffer
# #     try:
# #         data, sr = librosa.load(file_like, sr=sr)
# #         data = nr.reduce_noise(y=data, sr=sr)
# #         chunk_len = int(chunk_duration * sr)
# #         chunks = [data[i:i + chunk_len] for i in range(0, len(data), chunk_len) if len(data[i:i + chunk_len]) == chunk_len]
# #         return chunks, sr
# #     except Exception as e:
# #         st.error(f"Error processing audio: {e}")
# #         return [], sr

# # def chunk_to_spectrogram_buf(chunk, sr):
# #     S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, hop_length=512)
# #     S_db = librosa.power_to_db(S, ref=np.max)
# #     fig, ax = plt.subplots(figsize=(3,3))
# #     librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel", ax=ax)
# #     plt.axis("off")
# #     buf = io.BytesIO()
# #     plt.savefig(buf, bbox_inches="tight", pad_inches=0, format="png")
# #     plt.close(fig)
# #     buf.seek(0)
# #     return buf

# # def preprocess_spectrogram(img_buf):
# #     try:
# #         img = image.load_img(img_buf, target_size=IMG_SIZE)
# #         img_array = image.img_to_array(img) / 255.0
# #         img_array = np.expand_dims(img_array, axis=0)
# #         return img_array
# #     except Exception as e:
# #         st.error(f"Error preprocessing spectrogram: {e}")
# #         return None

# # def generate_confusion_plot(y_true, y_pred, labels=["Healthy","Parkinson's"]):
# #     cm = confusion_matrix(y_true, y_pred)
# #     fig, ax = plt.subplots(figsize=(4,4))
# #     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
# #     ax.set_xlabel("Predicted")
# #     ax.set_ylabel("True")
# #     st.pyplot(fig)

# # # ---------------------------
# # # Feature column order requested by user
# # # ---------------------------
# # FEATURE_COLUMNS = [
# #     "meanF0Hz","maxF0Hz","minF0Hz","stdF0Hz",
# #     "jitter_local","jitter_abs","jitter_rap","jitter_ddp","jitter_ppq5",
# #     "shimmer_local","shimmer_db","shimmer_apq3","shimmer_apq5","shimmer_dda",
# #     "hnr"
# # ] + [f"mfcc{i}" for i in range(12)] + [f"mel_mean{i}" for i in range(40)] + [f"mel_std{i}" for i in range(40)]

# # # ---------------------------
# # # Model loader with caching
# # # ---------------------------
# # @st.cache_resource
# # def load_all_models(model_dir):
# #     # file mapping - only load files that exist
# #     mapping = {
# #         "bilstm": "best_bilstm.keras",
# #         "cnn_lstm": "best_cnn_lstm_model.keras",
# #         "resnet50": "best_resnet50.keras",
# #         "gru": "best_gru.keras",
# #         "mobilenetv2": "best_mobilenetv2.keras",   # optional
# #         "cnn": "best_CNN.keras"                  # optional
# #     }
# #     loaded = {}
# #     errors = {}
# #     for key, fname in mapping.items():
# #         path = os.path.join(model_dir, fname)
# #         if os.path.exists(path):
# #             try:
# #                 loaded[key] = load_model(path)
# #             except Exception as e:
# #                 errors[key] = str(e)
# #         else:
# #             errors[key] = f"file not found: {path}"
# #     return loaded, errors

# # # ---------------------------
# # # UI Sidebar
# # # ---------------------------
# # st.sidebar.title("Navigation")
# # view = st.sidebar.radio("Go to", ["Test & Predict", "Model Metrics", "Performance Graphs"])

# # # Load models once (cached)
# # with st.spinner("Loading models..."):
# #     MODELS, MODEL_ERRORS = load_all_models(MODEL_DIR)

# # if MODEL_ERRORS:
# #     # present errors in the sidebar for debugging
# #     st.sidebar.markdown("**Model load status**")
# #     for k, err in MODEL_ERRORS.items():
# #         st.sidebar.text(f"{k}: {err}")

# # # prepare static DL model results (user-provided) as a DataFrame for Model Metrics page
# # dl_models_results = [
# #   { "Model": "CNN", "Accuracy": 0.93, "Precision": 0.92, "Recall": 0.95, "F1 Score": 0.93, "Method": "Spectrogram" },
# #   { "Model": "LSTM", "Accuracy": 0.844, "Precision": 0.84, "Recall": 0.85, "F1 Score": 0.844, "Method": "Spectrogram" },
# #   { "Model": "GRU", "Accuracy": 0.796, "Precision": 0.80, "Recall": 0.79, "F1 Score": 0.795, "Method": "Spectrogram" },
# #   { "Model": "BiLSTM", "Accuracy": 0.77, "Precision": 0.82, "Recall": 0.71, "F1 Score": 0.76, "Method": "Spectrogram" },
# #   { "Model": "ResNet50", "Accuracy": 0.76, "Precision": 0.71, "Recall": 0.88, "F1 Score": 0.78, "Method": "Spectrogram" },
# # ]
# # dl_models_df = pd.DataFrame(dl_models_results)

# # # ---------------------------
# # # Main: Test & Predict
# # # ---------------------------
# # if view == "Test & Predict":
# #     st.header("Upload a WAV audio file for prediction")
# #     uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

# #     if uploaded_file:
# #         # store bytes in session so it persists while navigating pages
# #         st.session_state.audio_bytes = uploaded_file.read()
# #         st.audio(st.session_state.audio_bytes)

# #         # Extract features for the whole file (not per-chunk) and display in table
# #         y, sr = safe_load_audio(io.BytesIO(st.session_state.audio_bytes), sr=SR)
# #         if y is not None:
# #             y_nr = reduce_noise(y, sr)
# #             extracted = extract_basic_features_from_audio(y_nr, sr)
# #             # ensure column order and presence
# #             row = {col: extracted.get(col, np.nan) for col in FEATURE_COLUMNS}
# #             features_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
# #             st.subheader("Extracted features (first row)")
# #             st.dataframe(features_df.T.rename(columns={0: "Value"}))  # transpose to show feature -> value
# #             # store in session to show on Model Metrics later
# #             st.session_state.extracted_features_df = features_df

# #         # Do chunk/prediction logic (use first chunk) but do NOT display spectrogram
# #         chunks, sr_chunks = audio_to_chunks(io.BytesIO(st.session_state.audio_bytes))
# #         st.write(f"Audio split into {len(chunks)} chunk(s) of {CHUNK_DURATION}s (used for model predictions)")
# #         if len(chunks) == 0:
# #             st.warning("No full-length chunks (3s) were found. Try a longer audio or reduce CHUNK_DURATION.")
# #         else:
# #             # create spectrogram buffer internally (for model input) but do not show it
# #             first_chunk = chunks[0]
# #             img_buf = chunk_to_spectrogram_buf(first_chunk, sr_chunks)
# #             x = preprocess_spectrogram(io.BytesIO(img_buf.getvalue()))
# #             if x is not None:
# #                 results = {}
# #                 for name, model in MODELS.items():
# #                     try:
# #                         out = model.predict(x, verbose=0)
# #                         score = None
# #                         if isinstance(out, np.ndarray):
# #                             if out.shape[-1] == 1:
# #                                 score = float(out[0][0])
# #                             else:
# #                                 score = float(out[0][1]) if out.shape[-1] > 1 else float(out[0][0])
# #                         else:
# #                             score = float(out)
# #                         label = "Parkinson's Disease (PD)" if (score is not None and score >= 0.5) else "Healthy Control (HC)"
# #                         results[name] = {"score": score, "label": label}
# #                     except Exception as e:
# #                         results[name] = {"score": None, "label": f"ERROR: {e}"}

# #                 # Voting summary
# #                 pd_votes = sum(1 for r in results.values() if r["label"].startswith("Parkinson"))
# #                 hc_votes = sum(1 for r in results.values() if r["label"].startswith("Healthy"))
# #                 final_label = "Parkinson's Disease (PD)" if pd_votes > hc_votes else "Healthy Control (HC)"
# #                 st.subheader("Voting Result")
# #                 st.write(f"Final: **{final_label}** (PD votes: {pd_votes}, HC votes: {hc_votes})")

# #                 # show model-wise predictions in a neat table
# #                 preds_df = pd.DataFrame([
# #                     {"Model": k, "Prediction": v["label"], "Score": (v["score"] if v["score"] is not None else np.nan)}
# #                     for k, v in results.items()
# #                 ])
# #                 st.subheader("Individual model predictions")
# #                 st.dataframe(preds_df.set_index("Model"))

# # elif view == "Model Metrics":
# #     st.header("Model Metrics & Extracted Features")
# #     # top: DL models table (user-specified)
# #     st.subheader("Deep Learning Models Results")
# #     st.dataframe(dl_models_df)

# #     st.markdown("---")
# #     # show the extracted features if present in session, else show placeholder
# #     st.subheader("Extracted Features (from last uploaded audio)")
# #     if "extracted_features_df" in st.session_state:
# #         # show as a wide table (single-row), display columns horizontally
# #         # We'll present as a single-row table with the requested column ordering.
# #         ef = st.session_state.extracted_features_df.copy()
# #         # round numeric columns for display
# #         ef_disp = ef.round(6)
# #         st.dataframe(ef_disp)
# #     else:
# #         st.info("No extracted features found in session. Upload an audio file on the 'Test & Predict' page to extract features.")

# #     st.markdown("---")
# #     st.subheader("Available models")
# #     for m in MODELS.keys():
# #         st.write(f"- {m}")

# #     if MODEL_ERRORS:
# #         st.warning("Some models were not loaded. Check the sidebar for errors.")

# # elif view == "Performance Graphs":
# #     st.header("Local performance graphs")
# #     names = ["CNN-LSTM.png", "BiLSTM.png", "GRU.png", "MobileNetV2.png", "ResNet50.png"]
# #     cols = st.columns(2)
# #     for i, fname in enumerate(names):
# #         path = os.path.join(PLOTS_DIR, fname)
# #         try:
# #             if os.path.exists(path):
# #                 img = Image.open(path)
# #                 with cols[i % 2]:
# #                     st.image(img, caption=fname, use_column_width=True)
# #             else:
# #                 with cols[i % 2]:
# #                     st.info(f"{fname} not found in {PLOTS_DIR}")
# #         except Exception as e:
# #             st.error(f"Error loading {fname}: {e}")
# # import streamlit as st
# # import os, io, math
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import noisereduce as nr
# # import librosa
# # import librosa.display
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.preprocessing import image
# # from sklearn.metrics import confusion_matrix
# # import seaborn as sns
# # from PIL import Image

# # IMG_SIZE = (128, 128)
# # CHUNK_DURATION = 3.0
# # SR = 22050

# # st.set_page_config(layout="wide", page_title="Parkinson's Prediction Dashboard")

# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # MODEL_DIR = os.path.join(BASE_DIR, "models")
# # PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# # def safe_load_audio(file_like, sr=SR):
# #     try:
# #         y, sr_ret = librosa.load(file_like, sr=sr)
# #         return y, sr_ret
# #     except:
# #         return None, sr

# # def reduce_noise(y, sr):
# #     try:
# #         return nr.reduce_noise(y=y, sr=sr)
# #     except:
# #         return y

# # def extract_basic_features_from_audio(y, sr):
# #     features = {}
# #     if y is None or len(y) == 0:
# #         return {k: np.nan for k in FEATURE_COLUMNS}

# #     try:
# #         f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
# #         voiced_f0 = f0[~np.isnan(f0)]
# #     except:
# #         voiced_f0 = np.array([])

# #     if voiced_f0.size > 0:
# #         features["meanF0Hz"] = float(np.mean(voiced_f0))
# #         features["maxF0Hz"] = float(np.max(voiced_f0))
# #         features["minF0Hz"] = float(np.min(voiced_f0))
# #         features["stdF0Hz"] = float(np.std(voiced_f0))
# #     else:
# #         for k in ["meanF0Hz","maxF0Hz","minF0Hz","stdF0Hz"]:
# #             features[k] = np.nan

# #     try:
# #         harmonic = librosa.effects.harmonic(y)
# #         noise = y - harmonic
# #         features["hnr"] = float(10 * np.log10((np.mean(harmonic ** 2) + 1e-12)/(np.mean(noise ** 2) + 1e-12)))
# #     except:
# #         features["hnr"] = np.nan

# #     try:
# #         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
# #         mfcc_means = np.mean(mfccs, axis=1)
# #         for i in range(12):
# #             features[f"mfcc{i}"] = float(mfcc_means[i])
# #     except:
# #         for i in range(12):
# #             features[f"mfcc{i}"] = np.nan

# #     try:
# #         mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
# #         mel_db = librosa.power_to_db(mel, ref=np.max)
# #         mel_means = np.mean(mel_db, axis=1)
# #         mel_stds = np.std(mel_db, axis=1)
# #         for i in range(40):
# #             features[f"mel_mean{i}"] = float(mel_means[i])
# #             features[f"mel_std{i}"] = float(mel_stds[i])
# #     except:
# #         for i in range(40):
# #             features[f"mel_mean{i}"] = np.nan
# #             features[f"mel_std{i}"] = np.nan

# #     return features

# # def audio_to_chunks(file_like, chunk_duration=CHUNK_DURATION, sr=SR):
# #     y, sr = librosa.load(file_like, sr=sr)
# #     y = reduce_noise(y, sr)
# #     chunk_len = int(chunk_duration * sr)
# #     chunks = [y[i:i + chunk_len] for i in range(0, len(y), chunk_len) if len(y[i:i + chunk_len]) == chunk_len]
# #     return chunks, sr

# # def chunk_to_spectrogram_buf(chunk, sr):
# #     S = librosa.feature.melspectrogram(y=chunk, sr=sr)
# #     S_db = librosa.power_to_db(S, ref=np.max)
# #     fig, ax = plt.subplots(figsize=(3,3))
# #     librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None, ax=ax)
# #     plt.axis("off")
# #     buf = io.BytesIO()
# #     plt.savefig(buf, bbox_inches="tight", pad_inches=0, format="png")
# #     plt.close(fig)
# #     buf.seek(0)
# #     return buf

# # def preprocess_spectrogram(img_buf):
# #     img = image.load_img(img_buf, target_size=IMG_SIZE)
# #     img_array = image.img_to_array(img) / 255.0
# #     return np.expand_dims(img_array, axis=0)

# # FEATURE_COLUMNS = ["meanF0Hz","maxF0Hz","minF0Hz","stdF0Hz","hnr"] \
# #                   + [f"mfcc{i}" for i in range(12)] \
# #                   + [f"mel_mean{i}" for i in range(40)] \
# #                   + [f"mel_std{i}" for i in range(40)]

# # @st.cache_resource
# # def load_all_models(model_dir):
# #     mapping = {
# #         "bilstm": "best_bilstm.keras",
# #         "cnn_lstm": "best_cnn_lstm_model.keras",
# #         "resnet50": "best_resnet50.keras",
# #         "gru": "best_gru.keras",
# #         "mobilenetv2": "best_mobilenetv2.keras",
# #         "cnn": "best_CNN.keras"
# #     }
# #     loaded, errors = {}, {}
# #     for key, fname in mapping.items():
# #         path = os.path.join(model_dir, fname)
# #         if os.path.exists(path):
# #             try:
# #                 loaded[key] = load_model(path)
# #             except Exception as e:
# #                 errors[key] = str(e)
# #         else:
# #             errors[key] = f"Missing: {fname}"
# #     return loaded, errors

# # st.sidebar.title("Navigation")
# # view = st.sidebar.radio("Go to", ["Test & Predict", "Model Metrics", "Performance Graphs"])

# # MODELS, _ = load_all_models(MODEL_DIR)

# # if view == "Test & Predict":
# #     st.header("Upload a WAV audio file for prediction")
# #     up = st.file_uploader("Choose a file", type=["wav"])

# #     if up:
# #         audio_bytes = up.read()
# #         st.audio(audio_bytes)

# #         y, sr = safe_load_audio(io.BytesIO(audio_bytes))
# #         if y is not None:
# #             extracted = extract_basic_features_from_audio(y, sr)
# #             df_feats = pd.DataFrame([{c: extracted.get(c, np.nan) for c in FEATURE_COLUMNS}])
# #             st.subheader("Extracted Features")
# #             st.dataframe(df_feats.T.rename(columns={0: "Value"}))

# #         chunks, sr = audio_to_chunks(io.BytesIO(audio_bytes))
# #         if len(chunks) == 0:
# #             st.warning("No 3-second chunk found.")
# #         else:
# #             first_chunk = chunks[0]
# #             img_buf = chunk_to_spectrogram_buf(first_chunk, sr)
# #             x = preprocess_spectrogram(io.BytesIO(img_buf.getvalue()))

# #             results = {}
# #             for name, model in MODELS.items():
# #                 pred = model.predict(x, verbose=0)
# #                 score = float(pred[0][1]) if pred.shape[-1] > 1 else float(pred[0][0])
# #                 label = "Parkinson's Disease" if score >= 0.5 else "Healthy Control"
# #                 results[name] = {"Score": score, "Prediction": label}

# #             st.subheader("Model Predictions")
# #             st.table(pd.DataFrame(results).T)

# #             votes = [v["Prediction"] for v in results.values()]
# #             final = max(set(votes), key=votes.count)
# #             st.success(f"🧠 Final Decision: **{final}**")

# # if view == "Model Metrics":
# #     st.header("Model Classification Report")
# #     metrics = [
# #         {"Model":"CNN","Accuracy":0.93,"Precision":0.92,"Recall":0.95,"F1 Score":0.93},
# #         {"Model":"LSTM","Accuracy":0.844,"Precision":0.84,"Recall":0.85,"F1 Score":0.844},
# #         {"Model":"GRU","Accuracy":0.796,"Precision":0.80,"Recall":0.79,"F1 Score":0.795},
# #         {"Model":"BiLSTM","Accuracy":0.77,"Precision":0.82,"Recall":0.71,"F1 Score":0.76},
# #         {"Model":"ResNet50","Accuracy":0.76,"Precision":0.71,"Recall":0.88,"F1 Score":0.78},
# #     ]
# #     st.dataframe(pd.DataFrame(metrics))

# # if view == "Performance Graphs":
# #     st.header("Uploaded Performance Graphs")
# #     img_files = [f for f in os.listdir(PLOTS_DIR) if f.lower().endswith(("png","jpg","jpeg"))]

# #     for file in img_files:
# #         st.image(os.path.join(PLOTS_DIR, file), caption=file)
# # # --- Continue from Performance Graphs page ---
# # if view == "Performance Graphs":
# #     st.header("📊 Model Performance Graphs")
# #     st.write("Below are the uploaded loss/accuracy and confusion matrix graphs for each model.")

# #     if not os.path.exists(PLOTS_DIR):
# #         st.error("❌ 'plots' folder not found. Please create a folder named **plots/** and place your images inside.")
# #     else:
# #         img_files = [f for f in os.listdir(PLOTS_DIR) if f.lower().endswith(("png", "jpg", "jpeg"))]

# #         if len(img_files) == 0:
# #             st.warning("⚠ No images found inside the `plots/` directory.")
# #         else:
# #             cols = st.columns(2)
# #             for idx, file in enumerate(img_files):
# #                 img_path = os.path.join(PLOTS_DIR, file)
# #                 try:
# #                     img = Image.open(img_path)
# #                     with cols[idx % 2]:
# #                         st.image(img, caption=file, use_container_width=True)
# #                 except:
# #                     st.error(f"Cannot open image: {file}")

# # # -----------------------------------------------------------
# # # Footer message
# # # -----------------------------------------------------------
# # st.markdown("""
# # <hr style="border: 1px solid #d9d9d9;">
# # <div style="text-align:center; color: gray;">
# #     Parkinson's Voice Diagnostic Platform • Powered by Deep Learning 🔬
# # </div>
# # """, unsafe_allow_html=True)
# import streamlit as st
# import os, io, math
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import noisereduce as nr
# import librosa
# import librosa.display
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# from PIL import Image

# # -------------------------
# # Config & constants
# # -------------------------
# IMG_SIZE = (128, 128)
# CHUNK_DURATION = 3.0
# SR = 22050

# st.set_page_config(layout="wide", page_title="Parkinson's Prediction Dashboard")

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, "models")
# PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# # small CSS for bigger header + nicer metrics table
# st.markdown(
#     """
#     <style>
#     .big-metrics-header {
#         font-size: 32px;
#         font-weight: 800;
#         margin-bottom: 0.5rem;
#     }
#     .metrics-table-wrapper table {
#         font-size: 16px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # -------------------------
# # Helper functions
# # -------------------------
# def safe_load_audio(file_like, sr=SR):
#     try:
#         y, sr_ret = librosa.load(file_like, sr=sr)
#         return y, sr_ret
#     except Exception:
#         return None, sr

# def reduce_noise(y, sr):
#     try:
#         return nr.reduce_noise(y=y, sr=sr)
#     except Exception:
#         return y

# def extract_basic_features_from_audio(y, sr):
#     features = {}
#     if y is None or len(y) == 0:
#         return {k: np.nan for k in FEATURE_COLUMNS}

#     # pitch
#     try:
#         f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
#         voiced_f0 = f0[~np.isnan(f0)]
#     except Exception:
#         voiced_f0 = np.array([])

#     if voiced_f0.size > 0:
#         features["meanF0Hz"] = float(np.mean(voiced_f0))
#         features["maxF0Hz"] = float(np.max(voiced_f0))
#         features["minF0Hz"] = float(np.min(voiced_f0))
#         features["stdF0Hz"] = float(np.std(voiced_f0))
#     else:
#         for k in ["meanF0Hz", "maxF0Hz", "minF0Hz", "stdF0Hz"]:
#             features[k] = np.nan

#     # hnr
#     try:
#         harmonic = librosa.effects.harmonic(y)
#         noise = y - harmonic
#         features["hnr"] = float(
#             10 * np.log10((np.mean(harmonic**2) + 1e-12) / (np.mean(noise**2) + 1e-12))
#         )
#     except Exception:
#         features["hnr"] = np.nan

#     # MFCCs
#     try:
#         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
#         mfcc_means = np.mean(mfccs, axis=1)
#         for i in range(12):
#             features[f"mfcc{i}"] = float(mfcc_means[i])
#     except Exception:
#         for i in range(12):
#             features[f"mfcc{i}"] = np.nan

#     # Mel means/stds
#     try:
#         mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
#         mel_db = librosa.power_to_db(mel, ref=np.max)
#         mel_means = np.mean(mel_db, axis=1)
#         mel_stds = np.std(mel_db, axis=1)
#         for i in range(40):
#             features[f"mel_mean{i}"] = float(mel_means[i])
#             features[f"mel_std{i}"] = float(mel_stds[i])
#     except Exception:
#         for i in range(40):
#             features[f"mel_mean{i}"] = np.nan
#             features[f"mel_std{i}"] = np.nan

#     return features

# def audio_to_chunks(file_like, chunk_duration=CHUNK_DURATION, sr=SR):
#     y, sr = librosa.load(file_like, sr=sr)
#     y = reduce_noise(y, sr)
#     chunk_len = int(chunk_duration * sr)
#     chunks = [
#         y[i : i + chunk_len]
#         for i in range(0, len(y), chunk_len)
#         if len(y[i : i + chunk_len]) == chunk_len
#     ]
#     return chunks, sr

# def chunk_to_spectrogram_buf(chunk, sr):
#     S = librosa.feature.melspectrogram(y=chunk, sr=sr)
#     S_db = librosa.power_to_db(S, ref=np.max)
#     fig, ax = plt.subplots(figsize=(3, 3))
#     librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None, ax=ax)
#     plt.axis("off")
#     buf = io.BytesIO()
#     plt.savefig(buf, bbox_inches="tight", pad_inches=0, format="png")
#     plt.close(fig)
#     buf.seek(0)
#     return buf

# def preprocess_spectrogram(img_buf):
#     img = image.load_img(img_buf, target_size=IMG_SIZE)
#     img_array = image.img_to_array(img) / 255.0
#     return np.expand_dims(img_array, axis=0)

# # -------------------------
# # Columns and feature lists
# # -------------------------
# FEATURE_COLUMNS = (
#     ["meanF0Hz", "maxF0Hz", "minF0Hz", "stdF0Hz", "hnr"]
#     + [f"mfcc{i}" for i in range(12)]
#     + [f"mel_mean{i}" for i in range(40)]
#     + [f"mel_std{i}" for i in range(40)]
# )

# # -------------------------
# # Load models (cached)
# # -------------------------
# @st.cache_resource
# def load_all_models(model_dir):
#     mapping = {
#         "bilstm": "best_bilstm.keras",
#         "cnn_lstm": "best_cnn_lstm_model.keras",
#         "resnet50": "best_resnet50.keras",
#         "gru": "best_gru.keras",
#         "mobilenetv2": "best_mobilenetv2.keras",
#         "cnn": "best_CNN.keras",
#     }
#     loaded, errors = {}, {}
#     for key, fname in mapping.items():
#         path = os.path.join(model_dir, fname)
#         if os.path.exists(path):
#             try:
#                 loaded[key] = load_model(path)
#             except Exception as e:
#                 errors[key] = str(e)
#         else:
#             errors[key] = f"Missing: {fname}"
#     return loaded, errors

# # -------------------------
# # UI: Sidebar navigation
# # -------------------------
# st.sidebar.title("Navigation")
# view = st.sidebar.radio("Go to", ["Test & Predict", "Model Metrics", "Performance Graphs"])

# MODELS, MODEL_ERRORS = load_all_models(MODEL_DIR)

# if MODEL_ERRORS:
#     with st.sidebar.expander("Model load status (click to view)"):
#         for k, err in MODEL_ERRORS.items():
#             st.write(f"- {k}: {err}")

# # -------------------------
# # Test & Predict page
# # -------------------------
# if view == "Test & Predict":
#     st.header("Upload a WAV audio file for prediction")

#     uploaded = st.file_uploader("Choose a .wav file", type=["wav"])

#     if uploaded:
#         audio_bytes = uploaded.read()
#         st.audio(audio_bytes)

#         # extract features
#         y, sr = safe_load_audio(io.BytesIO(audio_bytes))
#         if y is not None:
#             extracted = extract_basic_features_from_audio(y, sr)

#             key_features = {
#                 k: extracted.get(k, np.nan)
#                 for k in ["meanF0Hz", "maxF0Hz", "minF0Hz", "stdF0Hz", "hnr"]
#             }
#             kdf = pd.DataFrame([key_features]).T.rename(columns={0: "Value"})
#             st.subheader("Key extracted features")
#             st.dataframe(kdf.round(4), use_container_width=True)

#             with st.expander("Show full feature list (compact)"):
#                 full = {c: extracted.get(c, np.nan) for c in FEATURE_COLUMNS}
#                 items = list(full.items())
#                 half = math.ceil(len(items) / 2)
#                 col1_items = items[:half]
#                 col2_items = items[half:]
#                 c1, c2 = st.columns(2)
#                 with c1:
#                     for k, v in col1_items:
#                         st.write(f"**{k}**: {np.nan if pd.isna(v) else round(v, 6)}")
#                 with c2:
#                     for k, v in col2_items:
#                         st.write(f"**{k}**: {np.nan if pd.isna(v) else round(v, 6)}")

#         # predictions
#         chunks, sr_chunks = audio_to_chunks(io.BytesIO(audio_bytes))
#         if len(chunks) == 0:
#             st.warning("No full 3-second chunk found. Try a longer upload.")
#         else:
#             first_chunk = chunks[0]
#             img_buf = chunk_to_spectrogram_buf(first_chunk, sr_chunks)
#             x = preprocess_spectrogram(io.BytesIO(img_buf.getvalue()))
#             results = {}
#             for name, model in MODELS.items():
#                 try:
#                     pred = model.predict(x, verbose=0)
#                     score = float(pred[0][1]) if pred.shape[-1] > 1 else float(pred[0][0])
#                     label = "Parkinson's Disease" if score >= 0.5 else "Healthy Control"
#                     results[name] = {"Score": round(score, 4), "Prediction": label}
#                 except Exception as e:
#                     results[name] = {"Score": None, "Prediction": f"ERROR: {e}"}

#             st.subheader("Model Predictions")
#             preds_df = pd.DataFrame(results).T
#             preds_df.index.name = "Model"
#             st.table(preds_df)

#             votes = [v["Prediction"] for v in results.values()]
#             final = max(set(votes), key=votes.count)
#             st.success(f"🧠 Final Decision: **{final}**")

# # -------------------------
# # Model Metrics page
# # -------------------------
# elif view == "Model Metrics":
#     # bigger header via HTML
#     st.markdown(
#         "<div class='big-metrics-header'>Model Classification Report</div>",
#         unsafe_allow_html=True,
#     )

#     metrics = [
#         {"Model": "CNN", "Accuracy": 0.93, "Precision": 0.92, "Recall": 0.95, "F1 Score": 0.93},
#         {"Model": "LSTM", "Accuracy": 0.844, "Precision": 0.84, "Recall": 0.85, "F1 Score": 0.844},
#         {"Model": "GRU", "Accuracy": 0.796, "Precision": 0.80, "Recall": 0.79, "F1 Score": 0.795},
#         {"Model": "BiLSTM", "Accuracy": 0.77, "Precision": 0.82, "Recall": 0.71, "F1 Score": 0.76},
#         {"Model": "ResNet50", "Accuracy": 0.76, "Precision": 0.71, "Recall": 0.88, "F1 Score": 0.78},
#     ]
#     df_metrics = pd.DataFrame(metrics).set_index("Model").round(3)

#     st.markdown("**Deep learning model comparison** — rounded to 3 decimals")

#     # center the table & make it smaller with bigger font via Styler
#     styled = df_metrics.style.set_table_attributes(
#         'class="metrics-table-wrapper" style="width:65%; margin-left:auto; margin-right:auto;"'
#     )
#     st.table(styled)

#     best_model = df_metrics["Accuracy"].idxmax()
#     best_acc = df_metrics["Accuracy"].max()

#     cols = st.columns(3)
#     cols[0].metric("Best model (by Accuracy)", best_model, f"{best_acc:.3f}")
#     cols[1].metric("Models loaded", len(MODELS))
#     cols[2].metric("Errors loading models", len(MODEL_ERRORS))

# # -------------------------
# # Performance Graphs page
# # -------------------------
# elif view == "Performance Graphs":
#     st.header("📊 Model Performance Graphs")
#     st.write("Below are the uploaded model graphs (loss/accuracy/confusion matrices).")

#     if not os.path.exists(PLOTS_DIR):
#         st.error("❌ 'plots' folder not found. Please create a folder named **plots/** and place your images inside.")
#     else:
#         img_files = sorted(
#             [f for f in os.listdir(PLOTS_DIR) if f.lower().endswith(("png", "jpg", "jpeg"))]
#         )
#         if len(img_files) == 0:
#             st.warning("⚠ No images found in the `plots/` directory.")
#         else:
#             cols = st.columns(2)
#             for idx, fname in enumerate(img_files):
#                 p = os.path.join(PLOTS_DIR, fname)
#                 try:
#                     img = Image.open(p)
#                     with cols[idx % 2]:
#                         st.image(img, caption=fname, use_container_width=True)
#                 except Exception:
#                     st.error(f"Cannot open image: {fname}")

# # Footer
# st.markdown("---")
# st.markdown(
#     "<div style='text-align:center;color:gray;'>Parkinson's Voice Diagnostic Platform • Powered by Deep Learning 🔬</div>",
#     unsafe_allow_html=True,
# )

# a.py (updated)
import streamlit as st
import os, io, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import noisereduce as nr
import librosa              # make sure this is in requirements.txt
import librosa.display
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix
import seaborn as sns
from PIL import Image

# ---- Page config ----
st.set_page_config(layout="wide", page_title="Parkinson's Prediction Dashboard")

# Put this near the top of a.py (replace any existing st.markdown style block)
st.markdown(
    """
    <style>
    /* Global typography */
    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", Roboto, Arial, sans-serif !important;
        color: #0f172a;
        font-size: 18px !important;   /* make text larger everywhere */
    }

    /* Page header / section headings */
    .big-metrics-header {
        font-size: 36px !important;
        font-weight: 800;
        margin: 0.25rem 0 0.75rem 0;
        color: #042A48;                /* deep navy */
    }
    h1, .css-1v3fvcr h1 {
        font-size: 34px !important;
        color: #042A48;
    }
    h2, .css-1v3fvcr h2 {
        font-size: 26px !important;
        color: #08314b;
    }

    /* Sidebar font */
    .css-1d391kg, .css-1lcbmhc {
        font-size: 16px !important;
    }

    /* Tighter DataFrame table rows and professional colors */
    .stDataFrame table {
        border-collapse: collapse !important;
        width: 100% !important;
        font-size: 15px !important;
    }
    .stDataFrame thead th {
        background: linear-gradient(90deg, #E6F2F1 0%, #FFFFFF 100%);
        color: #04475e;
        font-weight: 700;
        padding: 6px 10px !important;
        border-bottom: 1px solid rgba(4,71,94,0.08);
    }
    .stDataFrame tbody td {
        padding: 6px 10px !important;
        border-bottom: 1px solid rgba(6, 39, 56, 0.04);
    }
    .stDataFrame tbody tr:nth-child(odd) td {
        background: #fbfdfe; /* very light */
    }
    .stDataFrame tbody tr:hover td {
        background: #f0fbfc !important;
    }

    /* Reduce table container padding */
    .element-container .stDataFrame {
        padding: 6px 8px !important;
    }

    /* Card-like metrics area */
    .metric-card {
        background: linear-gradient(180deg, #ffffff, #fcfeff);
        border: 1px solid rgba(4, 71, 94, 0.06);
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 6px 18px rgba(8, 49, 75, 0.03);
    }

    /* Image card styling */
    .plot-card {
        border: 1px solid rgba(4,71,94,0.06);
        border-radius: 8px;
        padding: 8px;
        background: #ffffff;
    }

    /* Center footer and hide Streamlit default footer */
    footer {visibility: hidden;}
    .app-footer {
        text-align:center;
        color: #6b7280;
        margin-top: 2.0rem;
        font-size: 15px;
    }

    /* Buttons & inputs subtle accent */
    .stButton>button {
        background: linear-gradient(90deg,#0b84a5,#07828f) !important;
        color: white !important;
        font-weight: 600;
        border-radius: 8px !important;
        padding: 8px 12px !important;
    }
    .stFileUploader, .stTextInput {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---- Paths (edit if needed) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ---- Constants ----
IMG_SIZE = (128, 128)
CHUNK_DURATION = 3.0
SR = 22050

# ---- Feature columns order (compact version) ----
FEATURE_COLUMNS = (
    ["meanF0Hz", "maxF0Hz", "minF0Hz", "stdF0Hz", "hnr"]
    + [f"mfcc{i}" for i in range(12)]
    + [f"mel_mean{i}" for i in range(40)]
    + [f"mel_std{i}" for i in range(40)]
)

# ---- Helper functions ----
def safe_load_audio(file_like, sr=SR):
    try:
        y, sr_ret = librosa.load(file_like, sr=sr)
        return y, sr_ret
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None, sr

def reduce_noise(y, sr):
    try:
        return nr.reduce_noise(y=y, sr=sr)
    except Exception:
        return y

def extract_basic_features_from_audio(y, sr):
    features = {}
    if y is None or len(y) == 0:
        return {k: np.nan for k in FEATURE_COLUMNS}

    # pitch / f0
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        voiced_f0 = f0[~np.isnan(f0)]
    except Exception:
        voiced_f0 = np.array([])

    if voiced_f0.size > 0:
        features["meanF0Hz"] = float(np.mean(voiced_f0))
        features["maxF0Hz"] = float(np.max(voiced_f0))
        features["minF0Hz"] = float(np.min(voiced_f0))
        features["stdF0Hz"] = float(np.std(voiced_f0))
    else:
        for k in ["meanF0Hz", "maxF0Hz", "minF0Hz", "stdF0Hz"]:
            features[k] = np.nan

    # HNR
    try:
        harmonic = librosa.effects.harmonic(y)
        noise = y - harmonic
        features["hnr"] = float(10 * np.log10((np.mean(harmonic ** 2) + 1e-12) / (np.mean(noise ** 2) + 1e-12)))
    except Exception:
        features["hnr"] = np.nan

    # MFCCs
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
        mfcc_means = np.mean(mfccs, axis=1)
        for i in range(12):
            features[f"mfcc{i}"] = float(mfcc_means[i])
    except Exception:
        for i in range(12):
            features[f"mfcc{i}"] = np.nan

    # mel means/stds
    try:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_means = np.mean(mel_db, axis=1)
        mel_stds = np.std(mel_db, axis=1)
        for i in range(40):
            features[f"mel_mean{i}"] = float(mel_means[i])
            features[f"mel_std{i}"] = float(mel_stds[i])
    except Exception:
        for i in range(40):
            features[f"mel_mean{i}"] = np.nan
            features[f"mel_std{i}"] = np.nan

    return features

@st.cache_resource
def load_all_models(model_dir):
    mapping = {
        "bilstm": "best_bilstm.keras",
        "cnn_lstm": "best_cnn_lstm_model.keras",
        "resnet50": "best_resnet50.keras",
        "gru": "best_gru.keras",
        "mobilenetv2": "best_mobilenetv2.keras",
        "cnn": "best_CNN.keras",
    }
    loaded, errors = {}, {}
    for key, fname in mapping.items():
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            try:
                loaded[key] = load_model(path)
            except Exception as e:
                errors[key] = str(e)
        else:
            errors[key] = f"Missing: {fname}"
    return loaded, errors

def load_csv_from_repo_or_upload(default_rel_path, uploader_key, label):
    """Try to load CSV from the repository data folder, otherwise show uploader fallback."""
    default_path = os.path.join(DATA_DIR, default_rel_path) if DATA_DIR else default_rel_path
    if os.path.exists(default_path):
        try:
            df = pd.read_csv(default_path)
            return df
        except Exception as e:
            st.warning(f"Failed to read {default_path}: {e}")
    # fallback uploader (single uploader, no extra 'Explore' button)
    up = st.file_uploader(label, type=["csv"], key=uploader_key)
    if up:
        try:
            return pd.read_csv(up)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
    return None

# ---- Sidebar + pages ----
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Test & Predict", "Model Metrics & Tables", "Performance Graphs"])

MODELS, MODEL_ERRORS = load_all_models(MODEL_DIR)

if MODEL_ERRORS:
    with st.sidebar.expander("Model load status"):
        for k, err in MODEL_ERRORS.items():
            st.write(f"- {k}: {err}")

# ---- Test & Predict ----
if page == "Test & Predict":
    st.header("Upload a WAV audio file for prediction")
    uploaded = st.file_uploader("Choose a .wav file", type=["wav"])
    if uploaded:
        audio_bytes = uploaded.read()
        st.audio(audio_bytes)
        y, sr = safe_load_audio(io.BytesIO(audio_bytes))
        if y is not None:
            y_nr = reduce_noise(y, sr)
            extracted = extract_basic_features_from_audio(y_nr, sr)
            # show key features
            key_cols = ["meanF0Hz", "maxF0Hz", "minF0Hz", "stdF0Hz", "hnr"]
            kdf = pd.DataFrame([{c: extracted.get(c, np.nan) for c in key_cols}]).T
            kdf.columns = ["Value"]
            st.subheader("Key extracted features")
            st.dataframe(kdf.round(4), width=520)

            # full features (compact)
            with st.expander("Show full feature set (compact)"):
                items = list({c: extracted.get(c, np.nan) for c in FEATURE_COLUMNS}.items())
                half = math.ceil(len(items) / 2)
                c1_items, c2_items = items[:half], items[half:]
                col1, col2 = st.columns(2)
                with col1:
                    for k, v in c1_items:
                        st.write(f"**{k}**: {np.nan if pd.isna(v) else round(v, 6)}")
                with col2:
                    for k, v in c2_items:
                        st.write(f"**{k}**: {np.nan if pd.isna(v) else round(v, 6)}")
        else:
            st.error("Unable to read audio. Try a different file.")
    else:
        st.info("Upload a WAV to extract features, or go to Model Metrics & Tables for CSVs.")

# ---- Model Metrics & Tables ----
elif page == "Model Metrics & Tables":
    st.markdown("<div class='big-metrics-header'>Model Classification Report</div>", unsafe_allow_html=True)

    metrics = [
        {"Model": "CNN", "Accuracy": 0.93, "Precision": 0.92, "Recall": 0.95, "F1 Score": 0.93},
        {"Model": "LSTM", "Accuracy": 0.844, "Precision": 0.84, "Recall": 0.85, "F1 Score": 0.844},
        {"Model": "GRU", "Accuracy": 0.796, "Precision": 0.80, "Recall": 0.79, "F1 Score": 0.795},
        {"Model": "BiLSTM", "Accuracy": 0.77, "Precision": 0.82, "Recall": 0.71, "F1 Score": 0.76},
        {"Model": "ResNet50", "Accuracy": 0.76, "Precision": 0.71, "Recall": 0.88, "F1 Score": 0.78},
    ]
    df_metrics = pd.DataFrame(metrics).set_index("Model").round(3)
    st.markdown("**Deep learning model comparison** — rounded to 3 decimals")
    st.table(df_metrics)  # neat centered small table

    cols = st.columns(3)
    best_model = df_metrics["Accuracy"].idxmax()
    best_acc = df_metrics["Accuracy"].max()
    cols[0].metric("Best model (Accuracy)", best_model, f"{best_acc:.3f}")
    cols[1].metric("Models loaded", len(MODELS))
    cols[2].metric("Errors loading models", len(MODEL_ERRORS))

    st.markdown("---")
    st.subheader("Feature tables (repo or upload)")

    tab1, tab2 = st.tabs(["1. Root Guided Approach (CSV)", "2. Feature Significance (CSV)"])

    # Tab 1 - load from data folder OR fallback uploader (no extra explore button)
    with tab1:
        st.write("Root guided CSV (from `data/rf_107_root_limited.csv` if present).")
        df1 = load_csv_from_repo_or_upload("rf_107_root_limited.csv", "upload_root", "Upload Root Guided CSV")
        if df1 is not None:
            st.success(f"Loaded CSV — rows: {df1.shape[0]} columns: {df1.shape[1]}")
            with st.expander("Preview (first 20 rows)"):
                st.dataframe(df1.head(20), use_container_width=True)
            st.download_button("Download CSV", df1.to_csv(index=False).encode("utf-8"),
                               file_name="root_guided_approach.csv", mime="text/csv")
        else:
            st.info("CSV not found in repo. Use the uploader above to load one.")

    # Tab 2 - significance CSV
    with tab2:
        st.write("Feature significance CSV (if available in `data/` or upload).")
        df2 = load_csv_from_repo_or_upload("rf_107_root_limited.csv", "upload_signif", "Upload significance CSV")
        # NOTE: change the default path above if you have a different file in data/
        if df2 is not None:
            st.success(f"Loaded CSV — rows: {df2.shape[0]} columns: {df2.shape[1]}")
            with st.expander("Preview (first 20 rows)"):
                st.dataframe(df2.head(20), use_container_width=True)
            st.download_button("Download significance CSV", df2.to_csv(index=False).encode("utf-8"),
                               file_name="feature_significance.csv", mime="text/csv")
        else:
            st.info("CSV not found in repo. Use the uploader above to load one.")
# Performance Graphs (use new container-width and card style)
elif page == "Performance Graphs":
    st.header("Performance Graphs")
    if os.path.exists(PLOTS_DIR):
        imgs = sorted([f for f in os.listdir(PLOTS_DIR) if f.lower().endswith((".png",".jpg",".jpeg"))])
    else:
        imgs = []

    if not imgs:
        st.info(f"No images found in `{PLOTS_DIR}`. Add PNGs there or commit them to the repo.")
    else:
        cols = st.columns(2)
        for i, fname in enumerate(imgs):
            path = os.path.join(PLOTS_DIR, fname)
            with cols[i % 2]:
                st.markdown(f"<div class='plot-card'>", unsafe_allow_html=True)
                try:
                    img = Image.open(path)
                    st.image(img, caption=fname, use_container_width=True)   # <<--- use_container_width
                except Exception as e:
                    st.error(f"Error loading {fname}: {e}")
                st.markdown("</div>", unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown("<div class='app-footer'>Parkinson's Voice Diagnostic Platform • Powered by Deep Learning 🔬</div>", unsafe_allow_html=True)

