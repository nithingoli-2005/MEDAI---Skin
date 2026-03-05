import React from "react";

const SignUp = () => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0d1b2a]">
      <div className="bg-white p-8 rounded-md shadow-md w-80">
        <div className="flex flex-col items-center mb-6">
          <img src="/logo.png" alt="MedAI Logo" className="w-16 mb-2" />
          <h1 className="text-2xl font-bold">Sign up</h1>
        </div>

        <label className="block mb-2 font-medium">E-mail</label>
        <input
          type="email"
          placeholder="Enter your email"
          className="w-full mb-4 px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-teal-600"
        />

        <label className="block mb-2 font-medium">Password</label>
        <input
          type="password"
          placeholder="Create a password"
          className="w-full mb-4 px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-teal-600"
        />

        <label className="block mb-2 font-medium">Confirm Password</label>
        <input
          type="password"
          placeholder="Confirm your password"
          className="w-full mb-4 px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-teal-600"
        />

        <button className="w-full bg-[#0d1b2a] text-white py-2 rounded-md hover:bg-[#152e4d] transition">
          Sign Up
        </button>

        <p className="text-sm mt-4 text-center">
          Already a member?{" "}
          <a href="/signin" className="text-teal-600 hover:underline">
            sign in here
          </a>
        </p>
      </div>
    </div>
  );
};

export default SignUp;
