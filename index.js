import React, { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append("image", file);

    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    setResult(data.character);
  };

  return (
    <div style={{ minHeight: "100vh", background: "#000", color: "#fff", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "2rem" }}>
      <h1 style={{ fontSize: "2rem", color: "#facc15" }}>Star Trek Lookalike</h1>
      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files[0])}
        style={{ marginTop: "1rem" }}
      />
      <button
        onClick={handleUpload}
        style={{
          marginTop: "1rem",
          backgroundColor: "#facc15",
          color: "#000",
          padding: "0.5rem 1.5rem",
          border: "none",
          borderRadius: "8px",
          cursor: "pointer"
        }}
      >
        Predict
      </button>
      {result && (
        <div style={{ textAlign: "center", marginTop: "2rem" }}>
          <p style={{ fontSize: "1.5rem" }}>You look like:</p>
          <p style={{ fontSize: "1.8rem", fontWeight: "bold", color: "#facc15" }}>{result}</p>
          <img
            src={`http://localhost:5000/data/${result}.jpg`}
            alt={result}
            style={{ marginTop: "1rem", width: "200px", borderRadius: "12px", border: "4px solid #facc15" }}
          />
        </div>
      )}
    </div>
  );
}

export default App;
