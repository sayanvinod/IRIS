import { useState } from 'react'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { useMutation } from '@tanstack/react-query'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"

type PredictionResult = {
  Prediction: string;
  Confidence: number;
}

async function analyzePhoto(file: File): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/get_predictions`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  const data = await response.json();
  if (data.error) {
    throw new Error(data.error);
  }
  return data;
}

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mutation = useMutation({
    mutationKey: ['analyze-photo'],
    mutationFn: analyzePhoto,
    onSuccess: (data: PredictionResult) => {
      setResult(data);
      setError(null);
    },
    onError: (error: any) => {
      setResult(null);
      setError(error.message || "Unknown error");
    }
  });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
      if (file) {
        setSelectedFile(file);
        setResult(null);
      }
  };

  const handleUpload = () => {
    if (selectedFile) mutation.mutate(selectedFile);
  }

  return (
    <>
    <header>
      <h1>Myopia Detector</h1>
    </header>
    <input type="file" accept="image/*" onChange={handleFileChange} />
    <button onClick={handleUpload} disabled={!selectedFile || mutation.isPending}>
      {mutation.isPending ? 'Analyzing...' : 'Analyze'}</button>
    <ReactQueryDevtools />

    {error && (
      <div style={{ color: 'red', marginTop: '1em' }}>
        <strong>Error:</strong> {error}
      </div>
    )}

    {/* Image Preview */}
    {selectedFile && (
      <div style={{ marginTop: '1em' }}>
      <h3>Image Preview:</h3>
      <img
        src={URL.createObjectURL(selectedFile)}
        alt='Selected'
        style = {{ maxWidth: '300px', borderRadius: '8px'}}
      />
      </div>
    )}

    {/* Prediction Results */}
    {result && (
      <div style={{ marginTop: '1em' }}>
        <h3>Prediction Result:</h3>
        <p><strong>Prediction:</strong> {result.Prediction}</p>
        <p><strong>Confidence:</strong> {(result.Confidence * 100).toFixed(2)}%</p>
      </div>
    )}
    </>
  )
}

export default App
