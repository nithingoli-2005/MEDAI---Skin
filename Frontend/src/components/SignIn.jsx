import React from "react";
import { Link, useNavigate } from "react-router-dom";

const SignIn = () => {
  const navigate = useNavigate();

  const handleLogin = (e) => {
    e.preventDefault();
    // In a real app you would verify credentials first.
    navigate("/dashboard");
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0d1b2a]">
      <div className="bg-white p-8 rounded-md shadow-md w-80">
        <div className="flex flex-col items-center mb-6">
          <img src="/logo.png" alt="MedAI Logo" className="w-16 mb-2" />
          <h1 className="text-2xl font-bold">Sign in</h1>
        </div>

        <label className="block mb-2 font-medium">E-mail</label>
        <input
          type="email"
          placeholder="Please enter your email"
          className="w-full mb-4 px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-teal-600"
        />

        <label className="block mb-2 font-medium">Password</label>
        <input
          type="password"
          placeholder="Please enter your password"
          className="w-full mb-4 px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-teal-600"
        />

        <button
          onClick={handleLogin}
          className="w-full bg-[#0d1b2a] text-white py-2 rounded-md hover:bg-[#152e4d] transition"
        >
          login
        </button>

        <p className="text-sm mt-4 text-center">
          Not a member?{" "}
          <Link to="/signup" className="text-teal-600 hover:underline">
            sign up here
          </Link>
        </p>
      </div>
    </div>
  );
};

export default SignIn;
