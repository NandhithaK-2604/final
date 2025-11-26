

import React, { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";

// Import local image assets for graphs
import cnnLstmGraph from "../assets/plots/CNN-LSTM.png";
import bilstmGraph from "../assets/plots/BiLSTM.png";
import gruGraph from "../assets/plots/GRU.png";
import mobilenetGraph from "../assets/plots/MobileNetV2.png";
import resnetGraph from "../assets/plots/ResNet50.png";

// Hard-coded data
const mlModelComparison = [
  { Model: "Random Forest", Method: "PCA", "Test Accuracy": 0.964, "F1 Score": 0.964 },
  { Model: "Support Vector Machine", Method: "PCA", "Test Accuracy": 0.976, "F1 Score": 0.976 },
  { Model: "Logistic Regression", Method: "PCA", "Test Accuracy": 0.976, "F1 Score": 0.976 },
  { Model: "Random Forest", Method: "SELECTKBEST", "Test Accuracy": 0.832, "F1 Score": 0.832 },
  { Model: "Support Vector Machine", Method: "SELECTKBEST", "Test Accuracy": 0.826, "F1 Score": 0.824 },
  { Model: "Logistic Regression", Method: "SELECTKBEST", "Test Accuracy": 0.97, "F1 Score": 0.97 },
];

const dlModelsResults = [
  { Model: "CNN", Accuracy: 0.93, Precision: 0.92, Recall: 0.95, "F1 Score": 0.93, Method: "Spectrogram" },
  { Model: "LSTM", Accuracy: 0.844, Precision: 0.84, Recall: 0.85, "F1 Score": 0.844, Method: "Spectrogram" },
  { Model: "GRU", Accuracy: 0.796, Precision: 0.80, Recall: 0.79, "F1 Score": 0.795, Method: "Spectrogram" },
  { Model: "BiLSTM", Accuracy: 0.77, Precision: 0.82, Recall: 0.71, "F1 Score": 0.76, Method: "Spectrogram" },
  { Model: "ResNet50", Accuracy: 0.76, Precision: 0.71, Recall: 0.88, "F1 Score": 0.78, Method: "Spectrogram" },
];

// Consolidated confusion matrix data
const allConfusionMatrices = {
  "CNN": { TP: 271, TN: 262, FP: 23, FN: 14 },
  "ResNet50": { TP: 252, TN: 183, FP: 102, FN: 33 },
  "MobileNetV2": { TP: 257, TN: 259, FP: 26, FN: 28 },
  "BiLSTM": { TP: 203, TN: 241, FP: 44, FN: 82 },
  "GRU": { TP: 0, TN: 285, FP: 0, FN: 285 },
  "LSTM": { TP: 200, TN: 240, FP: 45, FN: 85 }, // Placeholder data
  "Random Forest": { TP: 88, TN: 85, FP: 15, FN: 12 },
  "Support Vector Machine": { TP: 84, TN: 86, FP: 14, FN: 16 },
};

const totalFeatures = [
  "voiceID", "meanF0Hz", "maxF0Hz", "minF0Hz", "stdF0Hz", "jitter_local", "jitter_abs",
  "jitter_rap", "jitter_ddp", "jitter_ppq5", "shimmer_local", "shimmer_db",
  "shimmer_apq3", "shimmer_apq5", "shimmer_dda", "hnr", "mfcc0", "mfcc1", "mfcc2",
  "mfcc3", "mfcc4", "mfcc5", "mfcc6", "mfcc7", "mfcc8", "mfcc9", "mfcc10", "mfcc11",
  "mel_mean0-39", "mel_std0", "mel_std1-39", "label"
];

const selectedFeatures = [
  "maxF0Hz", "shimmer_apq3", "shimmer_apq5", "shimmer_dda", "hnr", "mfcc8", "mfcc9",
  "mfcc10", "mel_mean4", "mel_mean5", "mel_mean6", "mel_mean7", "mel_mean8",
  "mel_mean9", "mel_mean15", "mel_mean16", "mel_mean17", "mel_mean30",
  "mel_mean31", "mel_mean32", "mel_mean33", "mel_mean34", "mel_mean36",
  "mel_mean37", "mel_mean38", "mel_mean39", "mel_std1", "mel_std29", "mel_std30",
  "mel_std31", "mel_std32", "mel_std33", "mel_std34", "mel_std35", "mel_std36",
  "mel_std37", "mel_std38", "mel_std39"
];

const italianFeaturesData = [
  { "voiceID": "1", "meanF0Hz": 119.5, "jitter_local": 0.007, "shimmer_local": 0.02, "hnr": 18.2, "mfcc0": 4.5, "mel_mean0": 13.1, "label": "PD" },
  { "voiceID": "2", "meanF0Hz": 125.1, "jitter_local": 0.005, "shimmer_local": 0.015, "hnr": 20.1, "mfcc0": 4.1, "mel_mean0": 14.2, "label": "HC" },
  { "voiceID": "3", "meanF0Hz": 115.8, "jitter_local": 0.008, "shimmer_local": 0.023, "hnr": 17.5, "mfcc0": 4.8, "mel_mean0": 12.8, "label": "PD" },
  { "voiceID": "4", "meanF0Hz": 130.4, "jitter_local": 0.004, "shimmer_local": 0.013, "hnr": 21.5, "mfcc0": 3.9, "mel_mean0": 15.5, "label": "HC" },
  { "voiceID": "5", "meanF0Hz": 122.9, "jitter_local": 0.006, "shimmer_local": 0.018, "hnr": 19.0, "mfcc0": 4.3, "mel_mean0": 13.7, "label": "PD" },
];

const ModelGraphCard = ({ title, graphImage }) => (
  <div className="bg-white rounded-xl shadow-lg p-6">
    <h3 className="text-xl font-semibold text-gray-800 mb-4">{title}</h3>
    <img
      src={graphImage}
      alt={`${title} Performance Graphs`}
      className="w-full h-auto rounded-lg"
    />
  </div>
);

// New component for the confusion matrix box
const ConfusionMatrixCard = ({ title, matrix }) => {
  if (!matrix) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6 flex items-center justify-center h-[300px]">
        <p className="text-gray-500">Select a model to view its confusion matrix.</p>
      </div>
    );
  }

  const { TN, FP, FN, TP } = matrix;
  const predictedPD = TP + FP;
  const predictedHC = TN + FN;

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h3 className="text-xl font-bold text-gray-800 mb-4">{title}</h3>
      <div className="grid grid-cols-2 gap-2 text-center">
        <div className="p-4 bg-green-600 text-white rounded-tl-lg">
          <p className="text-lg font-semibold">True Negatives (TN)</p>
          <p className="text-3xl font-bold">{TN}</p>
        </div>
        <div className="p-4 bg-red-600 text-white rounded-tr-lg">
          <p className="text-lg font-semibold">False Positives (FP)</p>
          <p className="text-3xl font-bold">{FP}</p>
        </div>
        <div className="p-4 bg-red-600 text-white rounded-bl-lg">
          <p className="text-lg font-semibold">False Negatives (FN)</p>
          <p className="text-3xl font-bold">{FN}</p>
        </div>
        <div className="p-4 bg-green-600 text-white rounded-br-lg">
          <p className="text-lg font-semibold">True Positives (TP)</p>
          <p className="text-3xl font-bold">{TP}</p>
        </div>
      </div>
      <div className="mt-4 p-4 bg-gray-100 rounded-lg text-center">
        <p className="text-lg font-semibold text-gray-800">Predicted Counts</p>
        <p className="text-gray-600">
          <span className="font-bold text-blue-600">{predictedPD}</span> Predicted as Parkinson's
        </p>
        <p className="text-gray-600">
          <span className="font-bold text-green-600">{predictedHC}</span> Predicted as Healthy
        </p>
      </div>
    </div>
  );
};

function SpectrogramsPage() {
  const [file, setFile] = useState(null);
  const [audioURL, setAudioURL] = useState("");
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [view, setView] = useState("test");
  const [selectedModel, setSelectedModel] = useState("CNN"); // Default selected model

  const [votingResults, setVotingResults] = useState(null);
  const [extractedFeatures, setExtractedFeatures] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files?.[0] || null;
    setFile(selectedFile);
    if (selectedFile) {
      setAudioURL(URL.createObjectURL(selectedFile));
    } else {
      setAudioURL("");
    }
    setResults(null);
    setError(null);
    setVotingResults(null);
    setExtractedFeatures(null);
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file to upload!");
      return;
    }
    setError(null);
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Server Error: " + res.status);
      }
      const data = await res.json();
      setResults(data);

      setExtractedFeatures(data.features || null);

      if (data.chunks && data.chunks.length > 0) {
        const predictions = data.chunks[0].predictions;

        let pdCount = 0;
        let hcCount = 0;
        const models = Object.entries(predictions);

        models.forEach(([, pred]) => {
          if (pred.label === "Parkinson's Disease (PD)") {
            pdCount++;
          } else {
            hcCount++;
          }
        });

        const totalVotes = pdCount + hcCount;
        const pdPercentage = ((pdCount / totalVotes) * 100).toFixed(0);
        const hcPercentage = ((hcCount / totalVotes) * 100).toFixed(0);
        const finalPrediction = pdCount > hcCount ? "Parkinson's Disease (PD)" : "Healthy Control (HC)";

        setVotingResults({
          finalPrediction,
          pdPercentage,
          hcPercentage,
        });

        if (models.length > 0) {
          const topModel = models.sort((a, b) => b[1].score - a[1].score)[0][0];
          setSelectedModel(topModel);
        }
      }
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex bg-gray-100 min-h-screen font-sans">
      {/* Sidebar Navigation */}
      <div className="w-64 bg-gray-900 text-white p-6 space-y-4 shadow-xl">
        <h2 className="text-2xl font-extrabold text-center mb-8 tracking-wider">
          Dashboard 🔬
        </h2>
        <button
          onClick={() => setView("test")}
          className={`w-full py-3 rounded-xl flex items-center justify-center gap-3 transition-all duration-300 font-medium ${
            view === "test"
              ? "bg-blue-600 text-white shadow-md transform scale-105"
              : "bg-gray-700 text-gray-200 hover:bg-blue-500 hover:text-white"
          }`}
        >
          Test & Predict
        </button>
        <button
          onClick={() => setView("metrics")}
          className={`w-full py-3 rounded-xl flex items-center justify-center gap-3 transition-all duration-300 font-medium ${
            view === "metrics"
              ? "bg-purple-600 text-white shadow-md transform scale-105"
              : "bg-gray-700 text-gray-200 hover:bg-purple-500 hover:text-white"
          }`}
        >
          Model Metrics
        </button>
        <button
          onClick={() => setView("graphs")}
          className={`w-full py-3 rounded-xl flex items-center justify-center gap-3 transition-all duration-300 font-medium ${
            view === "graphs"
              ? "bg-teal-600 text-white shadow-md transform scale-105"
              : "bg-gray-700 text-gray-200 hover:bg-teal-500 hover:text-white"
          }`}
        >
          Performance Graphs
        </button>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 p-10 overflow-y-auto bg-gray-100">
        <h1 className="text-3xl font-extrabold text-gray-800 mb-8">
          Parkinson's Prediction Dashboard
        </h1>

        {/* The combined Test & Predict View */}
        {view === "test" && (
          <div className="space-y-8">
            {/* 1. Prediction Controls and Results */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Top Left: Audio Upload and Playback */}
              <div className="bg-white p-6 rounded-xl shadow h-full flex flex-col justify-between">
                <div>
                  <label htmlFor="file-upload" className="block text-gray-700 font-medium mb-2">
                    Select a `.wav` audio file for prediction:
                  </label>
                  <div className="flex items-center gap-4">
                    <input
                      id="file-upload"
                      type="file"
                      accept="audio/wav"
                      onChange={handleFileChange}
                      className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer"
                    />
                    <button
                      onClick={handleUpload}
                      disabled={loading || !file}
                      className={`px-6 py-2 rounded-full font-bold text-white transition-all duration-200 ${
                        loading || !file ? "bg-gray-400 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700 shadow-md"
                      }`}
                    >
                      {loading ? "Processing..." : "Predict"}
                    </button>
                  </div>
                </div>
                {audioURL && (
                  <div className="mt-4">
                    <h2 className="text-lg font-semibold text-gray-800 mb-2">Play Uploaded Audio</h2>
                    <audio controls src={audioURL} className="w-full" />
                  </div>
                )}
                {error && (
                  <div className="mt-4 bg-red-100 p-4 rounded-xl text-red-700">{error}</div>
                )}
              </div>

              {/* Top Right: Voting-Based Prediction Display */}
              {votingResults && (
                <div className="bg-white p-6 rounded-xl shadow-lg h-full flex flex-col justify-between">
                  <h2 className="text-xl font-bold text-gray-800 mb-4">Voting-Based Prediction</h2>
                  <div className="grid grid-cols-2 gap-4 text-center">
                    <div>
                      <h3 className="text-lg font-semibold text-gray-800">Final Prediction</h3>
                      <p className={`text-3xl font-bold ${votingResults.finalPrediction.includes("Parkinson's") ? 'text-red-600' : 'text-green-600'}`}>
                        {votingResults.finalPrediction}
                      </p>
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-gray-800">Vote Breakdown</h3>
                      <p className="text-lg text-gray-600">
                        <span className="font-bold text-blue-600">{votingResults.pdPercentage}%</span> Parkinson's
                      </p>
                      <p className="text-lg text-gray-600">
                        <span className="font-bold text-green-600">{votingResults.hcPercentage}%</span> Healthy
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* A single-column section for the individual predictions table */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="bg-white p-6 rounded-xl shadow overflow-y-auto max-h-[400px]">
                <h2 className="text-xl font-bold text-gray-800 mb-4">Individual Model Prediction Results</h2>
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="bg-gray-100 sticky top-0">
                      <th className="border-b p-3">Model</th>
                      <th className="border-b p-3">Prediction</th>
                      <th className="border-b p-3">probability</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results && results.chunks && results.chunks.length > 0 ? (
                      Object.entries(results.chunks[0].predictions).map(([model, pred]) => (
                        <tr
                          key={model}
                          className={`hover:bg-gray-50 cursor-pointer ${selectedModel === model ? 'bg-blue-100' : ''}`}
                          onClick={() => setSelectedModel(model)}
                        >
                          <td className="border-b p-3 font-semibold">{model}</td>
                          <td className="border-b p-3">{pred.label}</td>
                          <td className="border-b p-3">{pred.score?.toFixed(4) ?? "—"}</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="3" className="p-3 text-center text-gray-500">
                          Prediction results will be displayed here.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>

              {/* Extracted Features from Uploaded File */}
              {extractedFeatures && (
                <div className="bg-white p-6 rounded-xl shadow overflow-y-auto max-h-[400px]">
                  <h2 className="text-xl font-bold text-gray-800 mb-4">Extracted Features</h2>
                  <table className="w-full text-left border-collapse text-sm">
                    <thead>
                      <tr className="bg-gray-100 sticky top-0">
                        <th className="border-b p-3">Feature</th>
                        <th className="border-b p-3">Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(extractedFeatures).map(([feature, value]) => (
                        <tr key={feature} className="hover:bg-gray-50">
                          <td className="border-b p-3 font-semibold">{feature}</td>
                          <td className="border-b p-3">{value?.toFixed(4) ?? "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* 2. Feature Details Section (Lists) */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="bg-white p-6 rounded-xl shadow">
                <h2 className="text-xl font-bold text-gray-800 mb-4">Total Features</h2>
                <ul className="list-disc list-inside columns-2 text-sm text-gray-700">
                  {totalFeatures.map((feature, index) => (
                    <li key={index} className="mb-1">{feature}</li>
                  ))}
                </ul>
              </div>

              <div className="bg-white p-6 rounded-xl shadow">
                <h2 className="text-xl font-bold text-gray-800 mb-4">Selected K Features</h2>
                <ul className="list-disc list-inside columns-2 text-sm text-gray-700">
                  {selectedFeatures.map((feature, index) => (
                    <li key={index} className="mb-1">{feature}</li>
                  ))}
                </ul>
              </div>
            </div>

            {/* 3. Model Metrics Section */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="bg-white p-6 rounded-xl shadow overflow-x-auto">
                <h2 className="text-xl font-bold text-gray-800 mb-4">Machine Learning Models</h2>
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="bg-gray-100">
                      <th className="border-b p-3">Model</th>
                      <th className="border-b p-3">Method</th>
                      <th className="border-b p-3">Test Accuracy</th>
                      <th className="border-b p-3">F1 Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {mlModelComparison.map((row, i) => (
                      <tr key={i} className="hover:bg-gray-50">
                        <td className="border-b p-3 font-semibold">{row.Model}</td>
                        <td className="border-b p-3">{row.Method}</td>
                        <td className="border-b p-3">{row["Test Accuracy"]}</td>
                        <td className="border-b p-3">{row["F1 Score"]}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="bg-white p-6 rounded-xl shadow overflow-x-auto">
                <h2 className="text-xl font-bold text-gray-800 mb-4">Deep Learning Models</h2>
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="bg-gray-100">
                      <th className="border-b p-3">Model</th>
                      <th className="border-b p-3">Accuracy</th>
                      <th className="border-b p-3">Precision</th>
                      <th className="border-b p-3">Recall</th>
                      <th className="border-b p-3">F1 Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dlModelsResults.map((row, i) => (
                      <tr key={i} className="hover:bg-gray-50">
                        <td className="border-b p-3 font-semibold">{row.Model}</td>
                        <td className="border-b p-3">{row.Accuracy}</td>
                        <td className="border-b p-3">{row.Precision}</td>
                        <td className="border-b p-3">{row.Recall}</td>
                        <td className="border-b p-3">{row["F1 Score"]}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* 4. Confusion Matrices and Graphs Section */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Confusion Matrix as a dynamically rendered card */}
              <ConfusionMatrixCard
                title={`Confusion Matrix: ${selectedModel}`}
                matrix={allConfusionMatrices[selectedModel]}
              />
              <div className="bg-white p-6 rounded-xl shadow overflow-x-auto">
                <h2 className="text-xl font-bold text-gray-800 mb-4">First 5 Rows of Italian Audio Features</h2>
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="bg-gray-100">
                      {Object.keys(italianFeaturesData[0]).map((key) => (
                        <th key={key} className="border-b p-3">{key}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {italianFeaturesData.map((row, i) => (
                      <tr key={i} className="hover:bg-gray-50">
                        {Object.values(row).map((val, j) => (
                          <td key={j} className="border-b p-3">{val}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Down: Performance Graphs */}
            <div className="space-y-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">
                Model Performance Graphs
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <ModelGraphCard title="CNN-LSTM Performance" graphImage={cnnLstmGraph} />
                <ModelGraphCard title="BiLSTM Performance" graphImage={bilstmGraph} />
                <ModelGraphCard title="MobileNetV2 Performance" graphImage={mobilenetGraph} />
                <ModelGraphCard title="ResNet50 Performance" graphImage={resnetGraph} />
                <ModelGraphCard title="GRU Performance" graphImage={gruGraph} />
              </div>
            </div>
            
          </div>
        )}

        {/* The other views remain separated for navigation */}
        {view === "metrics" && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-white p-6 rounded-xl shadow overflow-x-auto">
              <h2 className="text-xl font-bold text-gray-800 mb-4">Machine Learning Models</h2>
              <table className="w-full text-left border-collapse text-sm">
                <thead>
                  <tr className="bg-gray-100">
                    <th className="border-b p-3">Model</th>
                    <th className="border-b p-3">Method</th>
                    <th className="border-b p-3">Test Accuracy</th>
                    <th className="border-b p-3">F1 Score</th>
                  </tr>
                </thead>
                <tbody>
                  {mlModelComparison.map((row, i) => (
                    <tr key={i} className="hover:bg-gray-50">
                      <td className="border-b p-3 font-semibold">{row.Model}</td>
                      <td className="border-b p-3">{row.Method}</td>
                      <td className="border-b p-3">{row["Test Accuracy"]}</td>
                      <td className="border-b p-3">{row["F1 Score"]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="bg-white p-6 rounded-xl shadow overflow-x-auto">
              <h2 className="text-xl font-bold text-gray-800 mb-4">Deep Learning Models</h2>
              <table className="w-full text-left border-collapse text-sm">
                <thead>
                  <tr className="bg-gray-100">
                    <th className="border-b p-3">Model</th>
                    <th className="border-b p-3">Accuracy</th>
                    <th className="border-b p-3">Precision</th>
                    <th className="border-b p-3">Recall</th>
                    <th className="border-b p-3">F1 Score</th>
                  </tr>
                </thead>
                <tbody>
                  {dlModelsResults.map((row, i) => (
                    <tr key={i} className="hover:bg-gray-50">
                      <td className="border-b p-3 font-semibold">{row.Model}</td>
                      <td className="border-b p-3">{row.Accuracy}</td>
                      <td className="border-b p-3">{row.Precision}</td>
                      <td className="border-b p-3">{row.Recall}</td>
                      <td className="border-b p-3">{row["F1 Score"]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="bg-white p-6 rounded-xl shadow overflow-x-auto">
              <h2 className="text-xl font-bold text-gray-800 mb-4">Confusion Matrices</h2>
              <table className="w-full text-left border-collapse text-sm">
                <thead>
                  <tr className="bg-gray-100">
                    <th className="border-b p-3">Model</th>
                    <th className="border-b p-3">TP</th>
                    <th className="border-b p-3">TN</th>
                    <th className="border-b p-3">FP</th>
                    <th className="border-b p-3">FN</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(allConfusionMatrices).map(([model, vals]) => (
                    <tr key={model} className="hover:bg-gray-50">
                      <td className="border-b p-3 font-semibold">{model}</td>
                      <td className="border-b p-3">{vals.TP}</td>
                      <td className="border-b p-3">{vals.TN}</td>
                      <td className="border-b p-3">{vals.FP}</td>
                      <td className="border-b p-3">{vals.FN}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="bg-white p-6 rounded-xl shadow">
              <h2 className="text-xl font-bold text-gray-800 mb-4">Feature Lists</h2>
              <div className="grid grid-cols-1 gap-6">
                <div>
                  <h3 className="font-semibold text-lg mb-2">Total Features</h3>
                  <ul className="list-disc list-inside columns-2 text-sm text-gray-700">
                    {totalFeatures.map((feature, index) => (
                      <li key={index} className="mb-1">{feature}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h3 className="font-semibold text-lg mb-2">Selected Features (SELECTKBEST)</h3>
                  <ul className="list-disc list-inside columns-2 text-sm text-gray-700">
                    {selectedFeatures.map((feature, index) => (
                      <li key={index} className="mb-1">{feature}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}

        {view === "graphs" && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-4">
              Model Performance Graphs
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <ModelGraphCard title="CNN-LSTM Performance" graphImage={cnnLstmGraph} />
              <ModelGraphCard title="BiLSTM Performance" graphImage={bilstmGraph} />
              <ModelGraphCard title="MobileNetV2 Performance" graphImage={mobilenetGraph} />
              <ModelGraphCard title="ResNet50 Performance" graphImage={resnetGraph} />
              <ModelGraphCard title="GRU Performance" graphImage={gruGraph} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default SpectrogramsPage;