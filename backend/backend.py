# # # import os
# # # import io
# # # import base64
# # # import numpy as np
# # # import librosa
# # # import librosa.display
# # # import matplotlib.pyplot as plt
# # # import noisereduce as nr
# # # from flask import Flask, request, jsonify
# # # from tensorflow.keras.models import load_model
# # # from tensorflow.keras.preprocessing import image
# # # from flask_cors import CORS  # <-- Enable CORS


# # # app = Flask(__name__)

# # # UPLOAD_FOLDER = "uploads"
# # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # # # =======================
# # # # Load models
# # # # =======================
# # # # MODEL_PATHS = [
# # # #     "src/assets/models/best_bilstm.keras",
# # # #     "src/assets/models/best_cnn_lstm_model.keras",
# # # #     "src/assets/models/best_gru.keras",
# # # #     "src/assets/models/best_mobilenetv2.keras",
# # # #     "src/assets/models/best_parkinson_model.keras",
# # # #     "src/assets/models/best_resnet50.keras"

# # # # ]

# # # IMG_SIZE = (128, 128)
# # # CHUNK_DURATION = 3.0
# # # SR = 22050


# # # MODELS={}
# # # try:
# # #     MODELS = {
# # #         "bilstm": load_model("src/assets/models/best_bilstm.keras"),
# # #         "lstm": load_model("src/assets/models/best_cnn_lstm_model.keras"),
# # #         "resnet": load_model( "src/assets/models/best_resnet50.keras"),
# # #         "gru":load_model("src/assets/models/best_gru.keras"),
# # #         "mobilenetv2": load_model( "src/assets/models/best_mobilenetv2.keras"),
# # #         "cnn":load_model( "src/assets/models/best_parkinson_model.keras")

# # #     }
# # #     print("✅ Models loaded successfully")
# # # except Exception as e:
# # #     print("❌ Error loading models:", e)


# # # # for path in MODEL_PATHS:
# # # #     model_name = os.path.splitext(os.path.basename(path))[0]
# # # #     models[model_name] = tf.keras.models.load_model(path)

# # # # # =======================
# # # # # Audio preprocessing
# # # # # =======================
# # # # IMG_SIZE = (128, 128)
# # # # CHUNK_DURATION = 3.0
# # # # SR = 22050

# # # # # def audio_to_melspectrogram(file_path):
# # # # #     y, sr = librosa.load(file_path, sr=22050)
# # # # #     S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
# # # # #     S_db = librosa.power_to_db(S, ref=np.max)

# # # # #     # Normalize
# # # # #     S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
# # # # #     S_resized = np.resize(S_norm, IMG_SIZE)

# # # # #     # Generate spectrogram image in memory (base64)
# # # # #     fig, ax = plt.subplots(figsize=(3, 3))
# # # # #     librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None, cmap="magma")
# # # # #     plt.axis("off")
# # # # #     buf = io.BytesIO()
# # # # #     plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
# # # # #     plt.close(fig)
# # # # #     buf.seek(0)
# # # # #     spectrogram_base64 = base64.b64encode(buf.read()).decode("utf-8")

# # # # #     return np.expand_dims(S_resized, axis=-1), spectrogram_base64


# # # # # =======================
# # # # # Prediction endpoint
# # # # # =======================
# # # # @app.route("/predict", methods=["POST"])
# # # # def predict():
# # # #     if "file" not in request.files:
# # # #         return jsonify({"error": "No file uploaded"}), 400
    
# # # #     file = request.files["file"]
# # # #     filename = secure_filename(file.filename)
# # # #     file_path = os.path.join(UPLOAD_FOLDER, filename)
# # # #     file.save(file_path)

# # # #     try:
# # # #         spectrogram, spectrogram_b64 = audio_to_melspectrogram(file_path)
# # # #         spectrogram = np.expand_dims(spectrogram, axis=0)

# # # #         predictions = []
# # # #         for name, model in models.items():
# # # #             probs = model.predict(spectrogram, verbose=0)[0]
# # # #             pred_class = int(np.argmax(probs))
# # # #             label = "Parkinson" if pred_class == 1 else "Healthy"
# # # #             confidence = float(np.max(probs))
# # # #             predictions.append({
# # # #                 "model": name,
# # # #                 "label": label,
# # # #                 "confidence": confidence
# # # #             })

# # # #         return jsonify({
# # # #             "predictions": predictions,
# # # #             "spectrogram": spectrogram_b64
# # # #         })

# # # #     except Exception as e:
# # # #         return jsonify({"error": str(e)}), 500


# # # # if __name__ == "__main__":
# # # #     app.run(debug=True)
# # # def audio_to_chunks(audio_file, chunk_duration=CHUNK_DURATION, sr=SR):
# # #     try:
# # #         y, sr = librosa.load(audio_file, sr=sr)
# # #         y = nr.reduce_noise(y=y, sr=sr)
# # #         chunk_len = int(chunk_duration * sr)
# # #         chunks = [y[i:i+chunk_len] for i in range(0, len(y), chunk_len) if len(y[i:i+chunk_len]) == chunk_len]
# # #         return chunks, sr
# # #     except Exception as e:
# # #         print("❌ Error in audio_to_chunks:", e)
# # #         return [], sr

# # # def chunk_to_spectrogram(chunk, sr):
# # #     try:
# # #         S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, hop_length=512)
# # #         S_db = librosa.power_to_db(S, ref=np.max)
# # #         fig, ax = plt.subplots(figsize=(3,3))
# # #         librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
# # #         plt.axis('off')
# # #         buf = io.BytesIO()
# # #         plt.savefig(buf, bbox_inches='tight', pad_inches=0, format='png')
# # #         plt.close(fig)
# # #         buf.seek(0)
# # #         return buf
# # #     except Exception as e:
# # #         print("❌ Error in chunk_to_spectrogram:", e)
# # #         return None

# # # def preprocess_spectrogram(img_buf):
# # #     try:
# # #         img = image.load_img(img_buf, target_size=IMG_SIZE)
# # #         img_array = image.img_to_array(img) / 255.0
# # #         img_array = np.expand_dims(img_array, axis=0)
# # #         return img_array
# # #     except Exception as e:
# # #         print("❌ Error in preprocess_spectrogram:", e)
# # #         return None

# # # def buf_to_base64(buf):
# # #     return base64.b64encode(buf.read()).decode("utf-8")

# # # # ---------------------------
# # # # Flask API
# # # # ---------------------------
# # # app = Flask(__name__)
# # # CORS(app)  # <-- Allow frontend to call backend

# # # @app.route("/", methods=["POST"])
# # # def predict():
# # #     if "file" not in request.files:
# # #         return jsonify({"error": "No file uploaded"}), 400

# # #     file = request.files["file"]

# # #     # Split audio into chunks
# # #     chunks, sr = audio_to_chunks(file)
# # #     if len(chunks) == 0:
# # #         return jsonify({"error": "Failed to process audio"}), 500

# # #     all_results = []
# # #     for i, chunk in enumerate(chunks):
# # #         buf = chunk_to_spectrogram(chunk, sr)
# # #         if buf is None:
# # #             continue
# # #         img = preprocess_spectrogram(io.BytesIO(buf.getvalue()))
# # #         if img is None:
# # #             continue

# # #         chunk_results = {}
# # #         for model_name, model in MODELS.items():
# # #             try:
# # #                 pred = float(model.predict(img)[0][0])
# # #                 label = "Parkinson's Disease (PD)" if pred >= 0.5 else "Healthy Control (HC)"
# # #                 chunk_results[model_name] = {"score": pred, "label": label}
# # #             except Exception as e:
# # #                 print(f"❌ Error predicting with {model_name}:", e)
# # #                 chunk_results[model_name] = {"score": None, "label": "Error"}

# # #         all_results.append({
# # #             "chunk": i + 1,
# # #             "spectrogram": buf_to_base64(io.BytesIO(buf.getvalue())),
# # #             "predictions": chunk_results
# # #         })

# # #     print("✅ Prediction complete, returning results")
# # #     return jsonify({"chunks": all_results})

# # # if __name__ == "__main__":
# # #     app.run(debug=True)


# # import os
# # import io
# # import base64
# # import numpy as np
# # import librosa
# # import librosa.display
# # import matplotlib.pyplot as plt
# # import noisereduce as nr
# # from flask import Flask, request, jsonify
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.preprocessing import image
# # from flask_cors import CORS

# # from keras.models import load_model

# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # MODEL_PATH = os.path.join(BASE_DIR, "../src/assets/models/best_bilstm.keras")

# # model = load_model(MODEL_PATH)

# # # ---------------------------
# # # Flask setup
# # # ---------------------------
# # app = Flask(__name__)
# # CORS(app)  # allow frontend to call backend

# # UPLOAD_FOLDER = "uploads"
# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # # ---------------------------
# # # Load models
# # # ---------------------------
# # IMG_SIZE = (128, 128)
# # CHUNK_DURATION = 3.0
# # SR = 22050
# # MODEL_DIR = os.path.join(BASE_DIR, "../src/assets/models")

# # MODELS = {}
# # try:
# #     MODELS = {
# #         "bilstm": load_model(os.path.join(MODEL_DIR, "best_bilstm.keras")),
# #         "cnn_lstm": load_model(os.path.join(MODEL_DIR, "best_cnn_lstm_model.keras")),
# #         "resnet50": load_model(os.path.join(MODEL_DIR, "best_resnet50.keras")),
# #         "gru": load_model(os.path.join(MODEL_DIR, "best_gru.keras")),
# #         "mobilenetv2": load_model(os.path.join(MODEL_DIR, "best_mobilenetv2.keras")),
# #         "cnn": load_model(os.path.join(MODEL_DIR, "best_CNN.keras"))  # <-- match your filename
# #     }
# #     print("✅ Models loaded successfully")
# # except Exception as e:
# #     print("❌ Error loading models:", e)


# # # ---------------------------
# # # Helper functions
# # # ---------------------------
# # def audio_to_chunks(audio_file, chunk_duration=CHUNK_DURATION, sr=SR):
# #     try:
# #         y, sr = librosa.load(audio_file, sr=sr)
# #         y = nr.reduce_noise(y=y, sr=sr)
# #         chunk_len = int(chunk_duration * sr)
# #         chunks = [
# #             y[i:i+chunk_len] for i in range(0, len(y), chunk_len)
# #             if len(y[i:i+chunk_len]) == chunk_len
# #         ]
# #         print(f"📂 Audio split into {len(chunks)} chunks")
# #         return chunks, sr
# #     except Exception as e:
# #         print("❌ Error in audio_to_chunks:", e)
# #         return [], sr

# # def chunk_to_spectrogram(chunk, sr):
# #     try:
# #         S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, hop_length=512)
# #         S_db = librosa.power_to_db(S, ref=np.max)
# #         fig, ax = plt.subplots(figsize=(3,3))
# #         librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel")
# #         plt.axis("off")
# #         buf = io.BytesIO()
# #         plt.savefig(buf, bbox_inches="tight", pad_inches=0, format="png")
# #         plt.close(fig)
# #         buf.seek(0)
# #         return buf
# #     except Exception as e:
# #         print("❌ Error in chunk_to_spectrogram:", e)
# #         return None

# # def preprocess_spectrogram(img_buf):
# #     try:
# #         img = image.load_img(img_buf, target_size=IMG_SIZE)
# #         img_array = image.img_to_array(img) / 255.0
# #         img_array = np.expand_dims(img_array, axis=0)
# #         return img_array
# #     except Exception as e:
# #         print("❌ Error in preprocess_spectrogram:", e)
# #         return None

# # def buf_to_base64(buf):
# #     return base64.b64encode(buf.read()).decode("utf-8")

# # # ---------------------------
# # # Routes
# # # ---------------------------
# # @app.route("/", methods=["GET"])
# # def home():
# #     return jsonify({"message": "Backend running ✅"})

# # @app.route("/predict", methods=["POST"])
# # def predict():
# #     if "file" not in request.files:
# #         return jsonify({"error": "No file uploaded"}), 400

# #     file = request.files["file"]
# #     print(f"📂 Received file: {file.filename}")

# #     chunks, sr = audio_to_chunks(file)
# #     if len(chunks) == 0:
# #         return jsonify({"error": "Failed to process audio"}), 500

# #     all_results = []
# #     for i, chunk in enumerate(chunks):
# #         buf = chunk_to_spectrogram(chunk, sr)
# #         if buf is None: continue
# #         img = preprocess_spectrogram(io.BytesIO(buf.getvalue()))
# #         if img is None: continue

# #         chunk_results = {}
# #         for model_name, model in MODELS.items():
# #             try:
# #                 pred = float(model.predict(img, verbose=0)[0][0])
# #                 label = "Parkinson's Disease (PD)" if pred >= 0.5 else "Healthy Control (HC)"
# #                 print(f"🧠 {model_name}: {pred:.4f} → {label}")
# #                 chunk_results[model_name] = {"score": pred, "label": label}
# #             except Exception as e:
# #                 print(f"❌ Error predicting with {model_name}:", e)
# #                 chunk_results[model_name] = {"score": None, "label": "Error"}

# #         all_results.append({
# #             "chunk": i + 1,
# #             "spectrogram": buf_to_base64(io.BytesIO(buf.getvalue())),
# #             "predictions": chunk_results
# #         })

# #     print("✅ Prediction complete, returning results")
# #     return jsonify({"chunks": all_results})

# # if __name__ == "__main__":
# #     app.run(host="0.0.0.0", port=5000, debug=True)
import os
import io
import base64
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import noisereduce as nr
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ---------------------------
# Flask setup
# ---------------------------
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../src/assets/models")

# ---------------------------
# Load models
# ---------------------------
IMG_SIZE = (128, 128)
CHUNK_DURATION = 3.0
SR = 22050

MODELS = {}
try:
    MODELS = {
        "bilstm": load_model(os.path.join(MODEL_DIR, "best_bilstm.keras")),
        "cnn_lstm": load_model(os.path.join(MODEL_DIR, "best_cnn_lstm_model.keras")),
        "resnet50": load_model(os.path.join(MODEL_DIR, "best_resnet50.keras")),
        "gru": load_model(os.path.join(MODEL_DIR, "best_gru.keras")),
        "mobilenetv2": load_model(os.path.join(MODEL_DIR, "best_mobilenetv2.keras")),
        "cnn": load_model(os.path.join(MODEL_DIR, "best_CNN.keras"))
    }
    print("✅ Models loaded successfully")
except Exception as e:
    print("❌ Error loading models:", e)

# ---------------------------
# Helper functions
# ---------------------------
def audio_to_chunks(audio_file, chunk_duration=CHUNK_DURATION, sr=SR):
    try:
        y, sr = librosa.load(audio_file, sr=sr)
        y = nr.reduce_noise(y=y, sr=sr)
        chunk_len = int(chunk_duration * sr)
        chunks = [
            y[i:i + chunk_len] for i in range(0, len(y), chunk_len)
            if len(y[i:i + chunk_len]) == chunk_len
        ]
        print(f"📂 Audio split into {len(chunks)} chunks")
        return chunks, sr
    except Exception as e:
        print("❌ Error in audio_to_chunks:", e)
        return [], sr

def chunk_to_spectrogram(chunk, sr):
    try:
        S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, hop_length=512)
        S_db = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(3, 3))
        librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel")
        plt.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, bbox_inches="tight", pad_inches=0, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        print("❌ Error in chunk_to_spectrogram:", e)
        return None

def preprocess_spectrogram(img_buf):
    try:
        img = image.load_img(img_buf, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print("❌ Error in preprocess_spectrogram:", e)
        return None

def buf_to_base64(buf):
    return base64.b64encode(buf.read()).decode("utf-8")

def generate_confusion_matrix_image(y_true, y_pred, labels=["Healthy", "Parkinson's"]):
    """
    Generates a confusion matrix plot and returns it as a base64-encoded image string.
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"❌ Error generating confusion matrix: {e}")
        return None

# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Backend running ✅"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    print(f"📂 Received file: {file.filename}")

    chunks, sr = audio_to_chunks(file)
    if len(chunks) == 0:
        return jsonify({"error": "Failed to process audio"}), 500

    all_results = []
    # Note: For simplicity, a single chunk is used for prediction feedback
    chunk = chunks[0]
    buf = chunk_to_spectrogram(chunk, sr)
    if buf is not None:
        img = preprocess_spectrogram(io.BytesIO(buf.getvalue()))
        if img is not None:
            chunk_results = {}
            for model_name, model in MODELS.items():
                try:
                    pred = float(model.predict(img, verbose=0)[0][0])
                    label = "Parkinson's Disease (PD)" if pred >= 0.5 else "Healthy Control (HC)"
                    chunk_results[model_name] = {"score": pred, "label": label}
                except Exception as e:
                    print(f"❌ Error predicting with {model_name}:", e)
                    chunk_results[model_name] = {"score": None, "label": "Error"}

            all_results.append({
                "chunk": 1,
                "spectrogram": buf_to_base64(io.BytesIO(buf.getvalue())),
                "predictions": chunk_results
            })

    # --- Generate Confusion Matrix Image (Example) ---
    # In a real scenario, you'd use a test set to generate this.
    # Here, we'll use a sample set of true and predicted labels.
    y_true_example = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_pred_example = np.array([0, 1, 1, 1, 0, 1, 0, 0, 1, 0])
    confusion_matrix_b64 = generate_confusion_matrix_image(y_true_example, y_pred_example)

    print("✅ Prediction complete, returning results")
    return jsonify({
        "chunks": all_results,
        "confusionMatrix": confusion_matrix_b64
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

# import os
# import io
# import base64
# import numpy as np
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import noisereduce as nr
# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from flask_cors import CORS
# from sklearn.metrics import confusion_matrix
# import seaborn as sns

# # ---------------------------
# # Flask setup
# # ---------------------------
# app = Flask(__name__)
# CORS(app)

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, "../src/assets/models")

# # ---------------------------
# # Load models
# # ---------------------------
# IMG_SIZE = (128, 128)
# CHUNK_DURATION = 3.0
# SR = 22050

# MODELS = {}
# try:
#     MODELS = {
#         "bilstm": load_model(os.path.join(MODEL_DIR, "best_bilstm.keras")),
#         "cnn_lstm": load_model(os.path.join(MODEL_DIR, "best_cnn_lstm_model.keras")),
#         "resnet50": load_model(os.path.join(MODEL_DIR, "best_resnet50.keras")),
#         "gru": load_model(os.path.join(MODEL_DIR, "best_gru.keras")),
#         "mobilenetv2": load_model(os.path.join(MODEL_DIR, "best_mobilenetv2.keras")),
#         "cnn": load_model(os.path.join(MODEL_DIR, "best_CNN.keras"))
#     }
#     print("✅ Models loaded successfully")
# except Exception as e:
#     print("❌ Error loading models:", e)

# # ---------------------------
# # Helper functions
# # ---------------------------
# def audio_to_chunks(audio_file, chunk_duration=CHUNK_DURATION, sr=SR):
#     try:
#         # Read the audio file into an in-memory buffer
#         audio_buffer = io.BytesIO(audio_file.read())
        
#         # Use librosa to load from the buffer, not the FileStorage object directly
#         y, sr = librosa.load(audio_buffer, sr=sr)
        
#         y = nr.reduce_noise(y=y, sr=sr)
#         chunk_len = int(chunk_duration * sr)
#         chunks = [
#             y[i:i + chunk_len] for i in range(0, len(y), chunk_len)
#             if len(y[i:i + chunk_len]) == chunk_len
#         ]
#         print(f"📂 Audio split into {len(chunks)} chunks")
#         return chunks, sr
#     except Exception as e:
#         print("❌ Error in audio_to_chunks:", e)
#         return [], sr

# def chunk_to_spectrogram(chunk, sr):
#     try:
#         S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, hop_length=512)
#         S_db = librosa.power_to_db(S, ref=np.max)
#         fig, ax = plt.subplots(figsize=(3, 3))
#         librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel")
#         plt.axis("off")
#         buf = io.BytesIO()
#         plt.savefig(buf, bbox_inches="tight", pad_inches=0, format="png")
#         plt.close(fig)
#         buf.seek(0)
#         return buf
#     except Exception as e:
#         print("❌ Error in chunk_to_spectrogram:", e)
#         return None

# def preprocess_spectrogram(img_buf):
#     try:
#         img = image.load_img(img_buf, target_size=IMG_SIZE)
#         img_array = image.img_to_array(img) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)
#         return img_array
#     except Exception as e:
#         print("❌ Error in preprocess_spectrogram:", e)
#         return None

# def buf_to_base64(buf):
#     return base64.b64encode(buf.read()).decode("utf-8")

# def generate_confusion_matrix_image(y_true, y_pred, labels=["Healthy", "Parkinson's"]):
#     """
#     Generates a confusion matrix plot and returns it as a base64-encoded image string.
#     """
#     try:
#         cm = confusion_matrix(y_true, y_pred)
#         fig, ax = plt.subplots(figsize=(6, 6))
#         sns.heatmap(
#             cm,
#             annot=True,
#             fmt="d",
#             cmap="Blues",
#             xticklabels=labels,
#             yticklabels=labels,
#             ax=ax
#         )
#         ax.set_title("Confusion Matrix")
#         ax.set_xlabel("Predicted Label")
#         ax.set_ylabel("True Label")

#         buf = io.BytesIO()
#         plt.savefig(buf, format="png")
#         plt.close(fig)
#         buf.seek(0)
#         return base64.b64encode(buf.getvalue()).decode('utf-8')
#     except Exception as e:
#         print(f"❌ Error generating confusion matrix: {e}")
#         return None

# # ---------------------------
# # Routes
# # ---------------------------
# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({"message": "Backend running ✅"})

# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]
#     print(f"📂 Received file: {file.filename}")

#     chunks, sr = audio_to_chunks(file)
#     if len(chunks) == 0:
#         return jsonify({"error": "Failed to process audio"}), 500

#     all_results = []
#     chunk = chunks[0]
#     buf = chunk_to_spectrogram(chunk, sr)
#     if buf is not None:
#         img = preprocess_spectrogram(io.BytesIO(buf.getvalue()))
#         if img is not None:
#             chunk_results = {}
#             for model_name, model in MODELS.items():
#                 try:
#                     pred = float(model.predict(img, verbose=0)[0][0])
#                     label = "Parkinson's Disease (PD)" if pred >= 0.5 else "Healthy Control (HC)"
#                     chunk_results[model_name] = {"score": pred, "label": label}
#                 except Exception as e:
#                     print(f"❌ Error predicting with {model_name}:", e)
#                     chunk_results[model_name] = {"score": None, "label": "Error"}

#             all_results.append({
#                 "chunk": 1,
#                 "spectrogram": buf_to_base64(io.BytesIO(buf.getvalue())),
#                 "predictions": chunk_results
#             })

#     y_true_example = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
#     y_pred_example = np.array([0, 1, 1, 1, 0, 1, 0, 0, 1, 0])
#     confusion_matrix_b64 = generate_confusion_matrix_image(y_true_example, y_pred_example)

#     print("✅ Prediction complete, returning results")
#     return jsonify({
#         "chunks": all_results,
#         "confusionMatrix": confusion_matrix_b64
#     })

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)