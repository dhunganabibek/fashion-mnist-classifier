import React, { useRef, useState, useEffect, useCallback } from "react";

type PredictionResponse = {
  predicted_class: string;
  predicted_class_id: number;
  confidence: number;
  all_probabilities: Record<string, number>;
};

const TARGET_SIZE = 28;
const PREDICT_INTERVAL = 800;

const FASHION_ICONS: Record<string, string> = {
  "T-shirt/top": "👕",
  Trouser: "👖",
  Pullover: "🧥",
  Dress: "👗",
  Coat: "🧥",
  Sandal: "👡",
  Shirt: "👔",
  Sneaker: "👟",
  Bag: "👜",
  "Ankle boot": "👢",
};

const App: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const rafRef = useRef<number | null>(null);
  const lastPredictRef = useRef<number>(0);
  const isPredictingRef = useRef<boolean>(false);

  const [isStreaming, setIsStreaming] = useState(false);
  const [apiUrl, setApiUrl] = useState("http://localhost:8000");
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);

  const predict = useCallback(
    async (blob: Blob) => {
      if (isPredictingRef.current) return;
      isPredictingRef.current = true;

      try {
        const formData = new FormData();
        formData.append("file", blob, "image.png");

        const res = await fetch(`${apiUrl}/predict`, {
          method: "POST",
          body: formData,
        });

        if (res.ok) {
          const data = await res.json();
          setPrediction(data);
        }
      } catch (error) {
        console.error("Prediction failed:", error);
      } finally {
        isPredictingRef.current = false;
      }
    },
    [apiUrl]
  );

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 960 }, height: { ideal: 720 } },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setIsStreaming(true);
      }
    } catch (error) {
      console.error("Camera error:", error);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
  }, []);

  const handleUpload = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (e) => {
        const dataUrl = e.target?.result as string;
        setUploadedImage(dataUrl);

        const img = new Image();
        img.onload = () => {
          const canvas = canvasRef.current;
          if (!canvas) return;

          canvas.width = TARGET_SIZE;
          canvas.height = TARGET_SIZE;
          const ctx = canvas.getContext("2d");
          ctx?.drawImage(img, 0, 0, TARGET_SIZE, TARGET_SIZE);

          canvas.toBlob((blob) => blob && predict(blob), "image/png");
        };
        img.src = dataUrl;
      };
      reader.readAsDataURL(file);
      event.target.value = "";
    },
    [predict]
  );

  const processFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas || !isStreaming) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = TARGET_SIZE;
    canvas.height = TARGET_SIZE;
    ctx.drawImage(video, 0, 0, TARGET_SIZE, TARGET_SIZE);

    const now = performance.now();
    if (now - lastPredictRef.current >= PREDICT_INTERVAL) {
      lastPredictRef.current = now;
      canvas.toBlob((blob) => blob && predict(blob), "image/png");
    }

    rafRef.current = requestAnimationFrame(processFrame);
  }, [isStreaming, predict]);

  useEffect(() => {
    if (isStreaming) {
      rafRef.current = requestAnimationFrame(processFrame);
    }
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [isStreaming, processFrame]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const sortedProbs = Object.entries(prediction?.all_probabilities || {})
    .map(([name, prob]) => ({ name, prob: Number(prob) || 0 }))
    .sort((a, b) => b.prob - a.prob);

  return (
    <div className="h-screen flex flex-col bg-neutral-900 text-white">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 bg-neutral-950 border-b border-neutral-800">
        <h1 className="text-lg font-semibold flex items-center gap-2">
          <span>👕</span> Fashion MNIST
        </h1>
        <div className="flex gap-2">
          <input
            type="text"
            value={apiUrl}
            onChange={(e) => setApiUrl(e.target.value)}
            className="px-3 py-1 rounded bg-neutral-800 border border-neutral-700 text-xs w-48"
            placeholder="API URL"
          />
          <input
            ref={fileInputRef}
            type="file"
            onChange={handleUpload}
            accept="image/*"
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-3 py-1 rounded bg-blue-600 hover:bg-blue-500 text-xs"
          >
            📤 Upload
          </button>
          <button
            onClick={isStreaming ? stopCamera : startCamera}
            className={`px-3 py-1 rounded text-xs ${
              isStreaming
                ? "bg-red-600 hover:bg-red-500"
                : "bg-green-600 hover:bg-green-500"
            }`}
          >
            {isStreaming ? "⏹️ Stop" : "📹 Camera"}
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Video/Upload View */}
        <div className="flex-1 relative bg-black">
          {uploadedImage ? (
            <div className="absolute inset-0 flex items-center justify-center p-8">
              <img
                src={uploadedImage}
                alt="Uploaded"
                className="max-w-full max-h-full object-contain rounded border-2 border-blue-500"
              />
              <button
                onClick={() => setUploadedImage(null)}
                className="absolute top-4 left-4 px-3 py-2 bg-neutral-900/90 hover:bg-neutral-800 rounded text-sm"
              >
                ← Back
              </button>
            </div>
          ) : (
            <>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className={`absolute inset-0 w-full h-full object-cover ${
                  isStreaming ? "" : "hidden"
                }`}
              />
              {!isStreaming && (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-neutral-500">
                  <div className="text-6xl mb-4">📷</div>
                  <div>Start Camera or Upload Image</div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Prediction Panel */}
        <aside className="w-96 bg-neutral-950 border-l border-neutral-800 flex flex-col">
          {/* Model Input Preview */}
          <div className="p-4 border-b border-neutral-800">
            <h2 className="text-xs font-semibold mb-2 uppercase text-neutral-400">
              Model Input ({TARGET_SIZE}×{TARGET_SIZE})
            </h2>
            <canvas
              ref={canvasRef}
              className="border border-purple-700 rounded bg-black mx-auto"
              style={{
                width: TARGET_SIZE * 4,
                height: TARGET_SIZE * 4,
                imageRendering: "pixelated",
              }}
            />
          </div>

          {/* Prediction Results */}
          <div className="flex-1 p-4 overflow-y-auto">
            {prediction ? (
              <>
                {/* Top Prediction */}
                <div className="mb-4 p-3 rounded bg-gradient-to-r from-purple-600 to-indigo-700">
                  <div className="flex items-center gap-3">
                    <div className="text-3xl">
                      {FASHION_ICONS[prediction.predicted_class] || "❓"}
                    </div>
                    <div className="flex-1">
                      <div className="text-sm font-bold">
                        {prediction.predicted_class}
                      </div>
                      <div className="mt-2 h-2 bg-white/30 rounded overflow-hidden">
                        <div
                          className={`h-full ${
                            prediction.confidence > 0.7
                              ? "bg-green-500"
                              : prediction.confidence > 0.4
                              ? "bg-orange-500"
                              : "bg-red-500"
                          }`}
                          style={{
                            width: `${prediction.confidence * 100}%`,
                          }}
                        />
                      </div>
                      <div className="text-xs mt-1 opacity-90">
                        {(prediction.confidence * 100).toFixed(1)}% confidence
                      </div>
                    </div>
                  </div>
                </div>

                {/* All Probabilities */}
                <h3 className="text-xs font-semibold mb-2 uppercase text-neutral-400">
                  All Classes
                </h3>
                <div className="space-y-1">
                  {sortedProbs.map(({ name, prob }, idx) => (
                    <div
                      key={name}
                      className="rounded bg-neutral-900 border border-neutral-800 p-2"
                    >
                      <div className="flex justify-between text-xs mb-1">
                        <span className="truncate">
                          <span className="text-purple-400 font-bold">
                            #{idx + 1}
                          </span>{" "}
                          {name}
                        </span>
                        <span className="font-semibold">
                          {(prob * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-1.5 bg-neutral-800 rounded overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-purple-600 to-indigo-700"
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="text-xs text-neutral-400">
                {isStreaming || uploadedImage
                  ? "Waiting for prediction..."
                  : "Start camera or upload image"}
              </div>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
};

export default App;
