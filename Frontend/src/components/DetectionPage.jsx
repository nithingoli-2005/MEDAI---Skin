import React, { useState } from "react";

const DetectionPage = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
    setPrediction(null);
    setError(null);

    // Create image preview
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDetect = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const formData = new FormData();
      formData.append("image", selectedFile);

      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Prediction failed");
      }

      const result = await response.json();
      setPrediction(result);
    } catch (err) {
      setError(err.message);
      console.error("Error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-[#0d1b2a] text-white px-4">
      {/* Logo */}
      <img src="/logo.png" alt="MedAI Logo" className="w-40 mb-4" />

      {/* MEDAI Title */}
      <h1 className="text-4xl font-extrabold mb-6">MEDAI</h1>

      {/* Choose Photo Button */}
      <label className="cursor-pointer bg-green-600 hover:bg-green-700 transition px-6 py-2 rounded-md mb-6 text-white font-medium">
        Choose Photo
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
        />
      </label>

      {/* Selected File Info */}
      {selectedFile && (
        <div className="mb-6 w-full max-w-md">
          <p className="text-center mb-4 text-gray-300">
            Selected File: {selectedFile.name}
          </p>
        </div>
      )}

      {/* Image Preview */}
      {imagePreview && (
        <div className="bg-gray-800 p-4 rounded-md mb-6 w-full max-w-md">
          <img
            src={imagePreview}
            alt="Preview"
            className="w-full h-auto rounded-md"
          />
        </div>
      )}

      {/* Detect Button */}
      {selectedFile && (
        <button
          onClick={handleDetect}
          disabled={loading}
          className={`${
            loading
              ? "bg-gray-500 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700"
          } transition px-8 py-3 rounded-md mb-6 text-white font-medium text-lg`}
        >
          {loading ? "Detecting..." : "Detect"}
        </button>
      )}

      {/* Error Message */}
      {error && (
        <div className="bg-red-600 p-4 rounded-md mb-6 w-full max-w-md text-center">
          <p className="text-white">Error: {error}</p>
        </div>
      )}

      {/* Result Div */}
      <div className="bg-gray-700 p-6 rounded-md w-full max-w-md text-center">
        {prediction ? (
          <div className="space-y-4">
            <div>
              <p className="text-gray-300 text-sm">Predicted Disease :</p>
              <p className="text-green-400 font-bold text-lg">
                {prediction.prediction_full}
              </p>
            </div>

            <div>
              <p className="text-base font-semibold">
                Accuracy : {prediction.accuracy} %
              </p>
            </div>
          </div>
        ) : (
          <p className="text-sm">
            {selectedFile && !loading
              ? "Click Detect to analyze the image"
              : "No photo selected yet"}
          </p>
        )}
      </div>
    </div>
  );
};

export default DetectionPage;
