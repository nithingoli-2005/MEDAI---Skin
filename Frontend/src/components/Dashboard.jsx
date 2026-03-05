import React from "react";
import { useNavigate } from "react-router-dom";

const Dashboard = () => {
  const navigate = useNavigate();

  return (
    <div className="h-screen flex flex-col items-center justify-center bg-[#0d1b2a] text-white">
      {/* Logo */}
      <img src="/logo.png" alt="MedAI Logo" className="w-40 mb-6" />

      {/* Title */}
      <h1 className="text-5xl font-extrabold tracking-wider mb-3">MEDAI</h1>

      {/* Subtitle */}
      <p className="text-lg text-gray-300 mb-6 text-center px-4">
        Deep Learning-Driven Skin Disease Detection <br />
        and Classification System
      </p>

      {/* Start Detect Button */}
      <button
        onClick={() => navigate("/detect")} // Navigates to DetectionPage
        className="px-6 py-2 rounded-md bg-green-600 hover:bg-green-700 transition"
      >
        Start Detect
      </button>
    </div>
  );
};

export default Dashboard;
