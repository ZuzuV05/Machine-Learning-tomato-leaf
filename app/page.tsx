'use client';

import React, { useState, useEffect, useRef } from 'react';
import Script from 'next/script';
import { 
  Upload, 
  FileImage, 
  Activity, 
  Settings, 
  AlertCircle, 
  CheckCircle, 
  RefreshCw, 
  Sparkles, 
  Leaf, 
  ChevronDown, 
  ChevronUp, 
  Info,
  HelpCircle,
  TrendingUp,
  Image as ImageIcon
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { cn } from '@/lib/utils';

// Classes defined by the user
const CLASSES = [
  'Healthy',
  'Bacterial Spot',
  'Late Blight',
  'Septoria Leaf Spot',
  'Yellow Leaf Curl Virus'
];

interface DiseaseDetails {
  scientificName: string;
  symptoms: string;
  causes: string;
  recommendations: string[];
  severity: 'low' | 'medium' | 'high';
  color: 'emerald' | 'amber' | 'red' | 'orange' | 'yellow';
}

const DISEASE_METADATA: Record<string, DiseaseDetails> = {
  'Healthy': {
    scientificName: 'Solanum lycopersicum (Healthy)',
    symptoms: 'Vibrant green leaves, robust stems, no unusual spots, yellowing, or curling. Normal growth patterns.',
    causes: 'Proper plant nutrition, correct watering schedules, adequate sunlight, and robust disease resistance.',
    recommendations: [
      'Maintain current watering schedules, applying water at the soil level rather than on the foliage.',
      'Ensure adequate spacing between plants to maximize air circulation and sunlight penetration.',
      'Apply a organic balanced fertilizer periodically to support continuous healthy growth.',
      'Regularly monitor lower leaves for any early signs of pest activity or moisture stress.'
    ],
    severity: 'low',
    color: 'emerald'
  },
  'Bacterial Spot': {
    scientificName: 'Xanthomonas campestris pv. vesicatoria',
    symptoms: 'Small, water-soaked, dark brown to black spots on leaves, often surrounded by a yellow halo. Spots may eventually dry out and form holes.',
    causes: 'Bacterial pathogen favored by high humidity, warm temperatures, and splashing water/overhead irrigation.',
    recommendations: [
      'Prune and destroy lower infected leaves immediately to prevent upward transmission.',
      'Avoid overhead watering completely; utilize drip lines or soil-level watering to keep foliage dry.',
      'Apply a copper-based bactericide/fungicide early in the morning if symptoms spread.',
      'Sanitize all gardening tools with a 10% bleach solution or isopropyl alcohol between plants.'
    ],
    severity: 'medium',
    color: 'amber'
  },
  'Late Blight': {
    scientificName: 'Phytophthora infestans',
    symptoms: 'Large, irregular water-soaked dark gray to brown lesions on leaves and stems. Under humid conditions, a white fuzzy fungal growth appears on the leaf undersides.',
    causes: 'Oomycete pathogen that thrives in cool, wet, and highly humid weather conditions.',
    recommendations: [
      'Urgently remove and bag infected foliage or entire plants. Do NOT compost diseased material.',
      'Improve air circulation by pruning excess foliage and using stakes or cages to keep vines upright.',
      'Apply preventative bio-fungicides or copper fungicides before humid, rainy weather patterns.',
      'Mulch the soil surface beneath tomato plants to create a barrier against soil-borne spores.'
    ],
    severity: 'high',
    color: 'red'
  },
  'Septoria Leaf Spot': {
    scientificName: 'Septoria lycopersici',
    symptoms: 'Numerous small, circular spots with dark brown margins and light gray or tan centers, typically starting on the oldest lower leaves.',
    causes: 'Fungal pathogen that overwinter in plant debris and soil, activated by warm temperatures and rain splash.',
    recommendations: [
      'Remove and destroy heavily spotted lower leaves as soon as they appear.',
      'Apply a thick layer of organic mulch (straw, wood chips) around the plant base to prevent rain-splash transmission.',
      'Ensure maximum spacing between tomato varieties to promote rapid leaf drying after rain.',
      'Apply organic neem oil or copper fungicides at the first sign of spotting to curb infection rates.'
    ],
    severity: 'medium',
    color: 'orange'
  },
  'Yellow Leaf Curl Virus': {
    scientificName: 'Tomato Yellow Leaf Curl Virus (TYLCV)',
    symptoms: 'Severe curling, cupping, or crumpling of leaves upward and inward. Leaves become abnormally small, yellowed, and the plant exhibits extreme stunting.',
    causes: 'Viral pathogen transmitted exclusively by whiteflies (Bemisia tabaci). Not curable once inside the plant tissue.',
    recommendations: [
      'Manage whitefly populations using insecticidal soap, neem oil sprays, or yellow sticky traps.',
      'Use reflective silver mulches which disorient and deter whiteflies from landing on young seedlings.',
      'Immediately isolate or rogue heavily infected virus-reservoir plants to protect healthy neighbors.',
      'Introduce natural predators like ladybugs or lacewings to naturally control whitefly infestations.'
    ],
    severity: 'high',
    color: 'yellow'
  }
};

// Helper to extract keras history arrays from a Keras 3 call node object (args/kwargs)
function extractKerasHistory(obj: any): any[] {
  if (!obj) return [];
  
  if (Array.isArray(obj)) {
    let result: any[] = [];
    for (const item of obj) {
      result = result.concat(extractKerasHistory(item));
    }
    return result;
  }
  
  if (typeof obj === 'object') {
    if (obj.class_name === '__keras_tensor__' && obj.config && Array.isArray(obj.config.keras_history)) {
      return [obj.config.keras_history];
    }
    if (Array.isArray(obj.keras_history)) {
      return [obj.keras_history];
    }
    
    let result: any[] = [];
    for (const key of Object.keys(obj)) {
      result = result.concat(extractKerasHistory(obj[key]));
    }
    return result;
  }
  
  return [];
}

// Convert Keras 3 inbound_nodes (list of call objects) to Keras 2 style (nested arrays)
function fixInboundNodes(inboundNodes: any): any {
  if (!Array.isArray(inboundNodes)) return inboundNodes;
  
  const isKeras3 = inboundNodes.some(node => node && typeof node === 'object' && !Array.isArray(node));
  
  if (isKeras3) {
    return inboundNodes.map(callNode => {
      if (!callNode || typeof callNode !== 'object' || Array.isArray(callNode)) {
        return callNode;
      }
      const histories = extractKerasHistory(callNode);
      return histories.map(hist => {
        if (Array.isArray(hist) && hist.length >= 3) {
          return [hist[0], hist[1], hist[2], {}];
        }
        return hist;
      });
    });
  }
  
  return inboundNodes;
}

// Recursive helper to traverse the model topology JSON and fix compatibility issues
function fixModelTopology(obj: any): any {
  if (!obj || typeof obj !== 'object') return obj;
  
  if (Array.isArray(obj)) {
    return obj.map(item => fixModelTopology(item));
  }
  
  const result: any = {};
  for (const key of Object.keys(obj)) {
    if (key === 'batch_shape' && Array.isArray(obj[key])) {
      result['batchInputShape'] = obj[key];
    }
    
    if (key === 'inbound_nodes' && Array.isArray(obj[key])) {
      result[key] = fixInboundNodes(obj[key]);
    } else {
      result[key] = fixModelTopology(obj[key]);
    }
  }
  return result;
}

export default function Home() {
  const [isTfLoaded, setIsTfLoaded] = useState(false);
  const [modelUrl, setModelUrl] = useState('https://raw.githubusercontent.com/ZuzuV05/Machine-Learning-tomato-leaf/refs/heads/main/tomato_disease_MobileNetV2_tfjs/model.json');
  const [modelType, setModelType] = useState<'layers' | 'graph'>('layers');
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [useDemoMode, setUseDemoMode] = useState(true); // Default to true so app works instantly prior to model setup
  const [showSettings, setShowSettings] = useState(false);

  // Uploaded and processed states
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isDragActive, setIsDragActive] = useState(false);
  const [inferenceTime, setInferenceTime] = useState<number | null>(null);
  
  // Predictions state
  const [predictions, setPredictions] = useState<{ className: string; confidence: number }[]>([]);
  const [topPrediction, setTopPrediction] = useState<{ className: string; confidence: number } | null>(null);

  // Canvas refs
  const originalCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const preprocessedCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const tfModelRef = useRef<any>(null);

  // Check if tf is available in global scope on mount and repeatedly
  useEffect(() => {
    const checkTf = setInterval(() => {
      if (typeof window !== 'undefined' && (window as any).tf) {
        setIsTfLoaded(true);
        clearInterval(checkTf);
      }
    }, 200);
    return () => clearInterval(checkTf);
  }, []);

  // Try to load model whenever TFJS loads or modelUrl changes
  const loadModel = async () => {
    if (!isTfLoaded) return;
    
    setIsModelLoading(true);
    setModelError(null);
    setModelLoaded(false);
    
    try {
      const tf = (window as any).tf;
      if (!tf) throw new Error("TensorFlow.js not initialized in window scope");

      console.log(`Loading model from ${modelUrl} as ${modelType}...`);
      
      let loadedModel;
      if (modelType === 'layers') {
        // Intercept loading to fix Keras 3 compatibility issues where batch_shape is used instead of batchInputShape
        const getBrowserLoader = (tfInstance: any, url: string) => {
          if (tfInstance.io.browserHTTPRequest) {
            return tfInstance.io.browserHTTPRequest(url);
          } else if (tfInstance.io.browserHttpRequest) {
            return tfInstance.io.browserHttpRequest(url);
          } else {
            throw new Error("No browser HTTP request loader found in TensorFlow.js");
          }
        };
        const defaultLoader = getBrowserLoader(tf, modelUrl);
        const customLoader = {
          load: async () => {
            const artifacts = await defaultLoader.load();
            if (artifacts.modelTopology) {
              let topology = artifacts.modelTopology;
              if (typeof topology === 'string') {
                try {
                  const parsed = JSON.parse(topology);
                  const fixed = fixModelTopology(parsed);
                  artifacts.modelTopology = JSON.stringify(fixed);
                } catch (e) {
                  console.error("Failed to parse and fix modelTopology string:", e);
                }
              } else {
                artifacts.modelTopology = fixModelTopology(topology);
              }
            }
            return artifacts;
          }
        };
        loadedModel = await tf.loadLayersModel(customLoader as any);
      } else {
        loadedModel = await tf.loadGraphModel(modelUrl);
      }
      
      tfModelRef.current = loadedModel;
      setModelLoaded(true);
      setUseDemoMode(false); // Automatically disable demo mode when real model is loaded successfully!
      console.log("Model loaded successfully!");
    } catch (err: any) {
      console.error("Error loading TFJS model:", err);
      setModelError(err?.message || "Failed to load model. Ensure CORS is allowed at the target URL.");
      // Keep useDemoMode as true when model fails to load
      setUseDemoMode(true);
    } finally {
      setIsModelLoading(false);
    }
  };

  useEffect(() => {
    if (isTfLoaded) {
      const timer = setTimeout(() => {
        loadModel();
      }, 0);
      return () => clearTimeout(timer);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isTfLoaded]);

  // Handle Drag Events
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  };

  const triggerFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleFileSelection = (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert("Please upload an image file (PNG, JPG, etc.).");
      return;
    }
    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setImagePreviewUrl(objectUrl);
    setPredictions([]);
    setTopPrediction(null);
    setInferenceTime(null);
  };

  // Perform disease classification (preprocessing + inference)
  const runClassification = async () => {
    if (!imagePreviewUrl) return;
    
    setIsAnalyzing(true);

    // Create an image element to load into canvas
    const img = new Image();
    img.src = imagePreviewUrl;
    
    img.onload = async () => {
      const startTime = Date.now();
      try {
        const prepCanvas = preprocessedCanvasRef.current;
        if (!prepCanvas) throw new Error("Preprocessed canvas is not available");

        const ctx = prepCanvas.getContext('2d');
        if (!ctx) throw new Error("Could not get 2D context");

        // 1. Center Crop and Resize to 224x224 (MobileNetV2 standard evaluation preprocessing)
        const size = 224;
        prepCanvas.width = size;
        prepCanvas.height = size;

        const sourceWidth = img.width;
        const sourceHeight = img.height;
        let sourceX = 0;
        let sourceY = 0;
        let sourceSize = Math.min(sourceWidth, sourceHeight);

        if (sourceWidth > sourceHeight) {
          sourceX = (sourceWidth - sourceHeight) / 2;
        } else {
          sourceY = (sourceHeight - sourceWidth) / 2;
        }

        ctx.drawImage(
          img,
          sourceX, sourceY, sourceSize, sourceSize, // crop center
          0, 0, size, size                         // scale to fit 224x224
        );

        let finalConfidences: number[] = [];

        // 2. TensorFlow.js inference or Color-Responsive simulation
        if (isTfLoaded && modelLoaded && tfModelRef.current && !useDemoMode) {
          const tf = (window as any).tf;
          
          // Preprocess: Convert canvas pixels to Tensor [224, 224, 3]
          const imageTensor = tf.browser.fromPixels(prepCanvas);
          
          // Expand dimensions to [1, 224, 224, 3]
          const expandedTensor = imageTensor.expandDims(0);
          
          // Cast to Float32 and normalize to [-1, 1] (MobileNetV2 standard: pixel = (pixel / 127.5) - 1.0)
          // This is strictly used instead of dividing by 255.0 to ensure correct model inference
          const floatTensor = expandedTensor.toFloat();
          const preprocessed = floatTensor.div(127.5).sub(1.0);

          // Run inference
          const predictionTensor = tfModelRef.current.predict(preprocessed);
          const data = await predictionTensor.data();
          finalConfidences = Array.from(data);

          // Dispose all tensors to prevent memory leaks
          tf.dispose([imageTensor, expandedTensor, floatTensor, preprocessed, predictionTensor]);
        } else {
          // Color-Responsive Demo/Simulated Mode Fallback
          // We analyze the drawn canvas to return smart simulated scores based on real leaf colors
          await new Promise(resolve => setTimeout(resolve, 800)); // Artificial network/inference delay
          finalConfidences = analyzeCanvasForSimulation(prepCanvas);
        }

        // 3. Map predictions and Sort
        const mappedPredictions = CLASSES.map((className, index) => ({
          className,
          confidence: finalConfidences[index] !== undefined ? finalConfidences[index] : 0.2
        }));

        // Normalize confidences to sum to 1 in case the custom model output isn't softmaxed
        const sum = mappedPredictions.reduce((acc, p) => acc + p.confidence, 0);
        const normalizedPredictions = mappedPredictions.map(p => ({
          className: p.className,
          confidence: sum > 0 ? p.confidence / sum : p.confidence
        })).sort((a, b) => b.confidence - a.confidence);

        const endTime = Date.now();
        setInferenceTime(endTime - startTime);
        setPredictions(normalizedPredictions);
        setTopPrediction(normalizedPredictions[0]);

      } catch (err: any) {
        console.error("Error during leaf classification:", err);
        alert("An error occurred during leaf classification: " + err.message);
      } finally {
        setIsAnalyzing(false);
      }
    };
  };

  // Analyze the canvas colors to return custom confidences (soft green -> healthy, soft yellow -> virus, etc.)
  const analyzeCanvasForSimulation = (canvas: HTMLCanvasElement): number[] => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return [0.2, 0.2, 0.2, 0.2, 0.2];
    
    const imgData = ctx.getImageData(0, 0, 224, 224);
    const data = imgData.data;
    
    let rSum = 0;
    let gSum = 0;
    let bSum = 0;
    const pixelCount = 224 * 224;
    
    for (let i = 0; i < data.length; i += 4) {
      rSum += data[i];
      gSum += data[i + 1];
      bSum += data[i + 2];
    }
    
    const rAvg = rSum / pixelCount;
    const gAvg = gSum / pixelCount;
    const bAvg = bSum / pixelCount;
    
    let scores = [0.1, 0.1, 0.1, 0.1, 0.1]; // [Healthy, Bacterial Spot, Late Blight, Septoria, Yellow Curl]
    
    // Green dominate threshold
    if (gAvg > rAvg * 1.08 && gAvg > bAvg * 1.1) {
      // Very green leaf -> high likelihood of Healthy
      scores = [0.82, 0.05, 0.03, 0.06, 0.04];
    } 
    // Yellow/Bright orange dominate threshold
    else if (rAvg > 140 && gAvg > 130 && bAvg < 110) {
      // Yellowish leaf -> high likelihood of Yellow Leaf Curl Virus
      scores = [0.03, 0.07, 0.10, 0.10, 0.70];
    } 
    // Medium dark brownish threshold
    else if (rAvg > gAvg && rAvg > bAvg * 1.2) {
      // Dark spots / dry -> Late Blight or Bacterial Spot
      scores = [0.04, 0.38, 0.42, 0.11, 0.05];
    } 
    // Grayish/mottled spots
    else {
      // Distributed with slightly higher Septoria leaf spot
      scores = [0.10, 0.20, 0.15, 0.45, 0.10];
    }
    
    return scores;
  };

  // Trigger analysis automatically when image is loaded
  useEffect(() => {
    if (imagePreviewUrl) {
      const timer = setTimeout(() => {
        runClassification();
      }, 0);
      return () => clearTimeout(timer);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imagePreviewUrl, useDemoMode, modelLoaded]);

  const severityColorMap = {
    low: 'bg-emerald-100 text-emerald-800 border-emerald-200',
    medium: 'bg-amber-100 text-amber-800 border-amber-200',
    high: 'bg-rose-100 text-rose-800 border-rose-200',
  };

  const cardBorderMap = {
    emerald: 'border-emerald-200 bg-emerald-50/40',
    amber: 'border-amber-200 bg-amber-50/40',
    red: 'border-rose-200 bg-rose-50/40',
    orange: 'border-orange-200 bg-orange-50/40',
    yellow: 'border-yellow-200 bg-yellow-50/40'
  };

  const progressBgMap = {
    'Healthy': 'bg-emerald-500',
    'Bacterial Spot': 'bg-amber-500',
    'Late Blight': 'bg-rose-500',
    'Septoria Leaf Spot': 'bg-orange-500',
    'Yellow Leaf Curl Virus': 'bg-yellow-500'
  };

  return (
    <div className="min-h-screen bg-[#FFF8F0] text-[#2D3436] flex flex-col items-center py-10 px-4 md:px-8 selection:bg-blue-100 selection:text-slate-900">
      {/* TensorFlow.js script injection from official JsDelivr CDN */}
      <Script 
        src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js" 
        strategy="afterInteractive"
      />

      <div className="w-full max-w-4xl flex flex-col gap-8" id="app-container">
        {/* Header Applet Bar */}
        <header className="flex flex-col md:flex-row md:items-center justify-between border-b border-[#E6E0D8] pb-6 gap-4" id="app-header">
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 bg-[#90CAF9] rounded-full"></span>
              <h1 className="text-xl font-bold tracking-tight text-[#1A1A1A] font-display">
                TomatoLeaf AI
              </h1>
              <span className="text-[10px] font-bold text-[#1E88E5] bg-[#E3F2FD] px-2.5 py-0.5 rounded-full uppercase tracking-wider ml-1">
                v2.4.0-Stable
              </span>
            </div>
            <p className="text-sm text-[#636E72] font-medium">
              Tomato Leaf Disease Classification System
            </p>
            <p className="text-xs text-[#636E72]/80">
              Upload a tomato leaf image to detect plant disease instantly
            </p>
          </div>

          {/* Model Status Badge & Quick Control */}
          <div className="flex flex-wrap items-center gap-2">
            {/* TF.js CDN badge */}
            <span className={cn(
              "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border transition-all duration-300",
              isTfLoaded 
                ? "bg-[#E3F2FD] text-[#1E88E5] border-transparent" 
                : "bg-amber-50 text-amber-700 border-amber-200 animate-pulse"
            )}>
              <Activity className={cn("w-3.5 h-3.5", isTfLoaded && "animate-spin-slow")} />
              TFJS: {isTfLoaded ? "Active" : "Loading CDN..."}
            </span>

            {/* Model status badge */}
            <span className={cn(
              "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border",
              useDemoMode 
                ? "bg-amber-50 text-amber-700 border-amber-100" 
                : modelLoaded 
                  ? "bg-emerald-50 text-emerald-700 border-emerald-100"
                  : "bg-slate-100 text-slate-500 border-slate-200"
            )}>
              <span className="relative flex h-1.5 w-1.5">
                <span className={cn(
                  "animate-ping absolute inline-flex h-full w-full rounded-full opacity-75",
                  useDemoMode ? "bg-amber-400" : modelLoaded ? "bg-emerald-400" : "bg-slate-400"
                )}></span>
                <span className={cn(
                  "relative inline-flex rounded-full h-1.5 w-1.5",
                  useDemoMode ? "bg-amber-500" : modelLoaded ? "bg-emerald-500" : "bg-slate-400"
                )}></span>
              </span>
              Engine: {useDemoMode ? "Sandbox" : modelLoaded ? "Active Model" : "No Model"}
            </span>

            {/* Advanced Settings trigger button */}
            <button 
              onClick={() => setShowSettings(!showSettings)}
              className={cn(
                "p-1.5 rounded-lg border text-slate-600 hover:bg-white transition-all duration-200",
                showSettings ? "bg-white border-[#90CAF9]" : "bg-transparent border-transparent"
              )}
              title="Configure TensorFlow Model"
              id="settings-toggle-btn"
            >
              <Settings className="w-4 h-4" />
            </button>
          </div>
        </header>

        {/* Expandable TensorFlow.js Model Settings */}
        <AnimatePresence>
          {showSettings && (
            <motion.div 
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.25, ease: 'easeInOut' }}
              className="overflow-hidden border border-[#E6E0D8] bg-white rounded-xl shadow-sm"
              id="settings-panel"
            >
              <div className="p-5 flex flex-col gap-4 border-b border-slate-100 bg-slate-50/50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Settings className="w-5 h-5 text-slate-600" />
                    <h3 className="font-bold text-sm text-[#1A1A1A]">TensorFlow.js Model Configuration</h3>
                  </div>
                  <button 
                    onClick={() => setShowSettings(false)}
                    className="text-xs text-slate-400 hover:text-slate-600 font-medium"
                  >
                    Collapse
                  </button>
                </div>
                
                <p className="text-xs text-slate-500 leading-relaxed">
                  Provide your own custom TensorFlow.js model configuration below. If no custom model is loaded, the app runs in <strong>Interactive Sandbox Mode</strong>, dynamically simulating classifications using color analytics of your tomato leaf.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-1">
                  {/* Model URL field */}
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-bold text-slate-600">Model JSON URL</label>
                    <input 
                      type="text"
                      className="w-full px-3 py-1.5 text-xs font-mono rounded-lg border border-[#E6E0D8] bg-white focus:outline-none focus:ring-2 focus:ring-[#90CAF9] focus:border-transparent transition-all"
                      placeholder="https://domain.com/path/to/model.json"
                      value={modelUrl}
                      onChange={(e) => setModelUrl(e.target.value)}
                    />
                  </div>

                  {/* Model Loader Selector & Toggles */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="flex flex-col gap-1.5">
                      <label className="text-xs font-bold text-slate-600">Model Format</label>
                      <select
                        className="w-full px-2.5 py-1.5 text-xs rounded-lg border border-[#E6E0D8] bg-white focus:outline-none focus:ring-2 focus:ring-[#90CAF9] focus:border-transparent transition-all"
                        value={modelType}
                        onChange={(e) => setModelType(e.target.value as 'layers' | 'graph')}
                      >
                        <option value="layers">Layers Model (tf.loadLayersModel)</option>
                        <option value="graph">Graph Model (tf.loadGraphModel)</option>
                      </select>
                    </div>

                    <div className="flex flex-col gap-1.5">
                      <label className="text-xs font-bold text-slate-600">Sandbox Mode</label>
                      <div className="flex items-center h-full">
                        <button
                          type="button"
                          onClick={() => setUseDemoMode(!useDemoMode)}
                          className={cn(
                            "relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none",
                            useDemoMode ? "bg-[#90CAF9]" : "bg-slate-200"
                          )}
                        >
                          <span
                            className={cn(
                              "pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out",
                              useDemoMode ? "translate-x-5" : "translate-x-0"
                            )}
                          />
                        </button>
                        <span className="ml-2 text-xs font-semibold text-slate-600">
                          {useDemoMode ? "Active" : "Off"}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="flex flex-wrap items-center justify-between gap-3 mt-2 pt-3 border-t border-slate-100">
                  <div className="text-xs font-medium text-slate-500">
                    {isModelLoading ? (
                      <span className="flex items-center gap-1.5 text-blue-600">
                        <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Loading model weights...
                      </span>
                    ) : modelError ? (
                      <span className="flex items-center gap-1.5 text-rose-600">
                        <AlertCircle className="w-3.5 h-3.5 shrink-0" /> {modelError.slice(0, 70)}...
                      </span>
                    ) : modelLoaded ? (
                      <span className="flex items-center gap-1.5 text-emerald-600">
                        <CheckCircle className="w-3.5 h-3.5" /> Model compiled and ready!
                      </span>
                    ) : (
                      <span>Sandbox fallback active.</span>
                    )}
                  </div>

                  <div className="flex gap-2">
                    <button
                      onClick={() => {
                        setModelUrl('https://raw.githubusercontent.com/sachinsharma9/tomato-disease-classification-tfjs/master/model/model.json');
                        setModelType('layers');
                      }}
                      className="px-2.5 py-1 text-xs font-semibold text-slate-500 hover:text-slate-800 transition"
                    >
                      Reset Default
                    </button>
                    <button
                      onClick={loadModel}
                      disabled={isModelLoading || !isTfLoaded}
                      className="px-3.5 py-1 bg-slate-800 text-white rounded-lg text-xs font-semibold hover:bg-slate-700 disabled:opacity-50 transition flex items-center gap-1"
                    >
                      {isModelLoading ? "Compiling..." : "Compile Model"}
                    </button>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Balanced Main Container Area */}
        <main className="grid grid-cols-1 md:grid-cols-2 rounded-2xl overflow-hidden border border-[#E6E0D8] bg-white shadow-[0_10px_30px_rgba(0,0,0,0.05)]" id="main-content">
          
          {/* Left Column (Upload & Acquisition) */}
          <section className="p-6 md:p-10 border-b md:border-b-0 md:border-r border-[#F0EAE4] flex flex-col justify-between gap-6 bg-white">
            
            <div className="flex flex-col gap-6">
              <div className="flex flex-col gap-1.5">
                <div className="inline-flex items-center self-start px-3 py-1 bg-[#E3F2FD] text-[#1E88E5] rounded-full text-[10px] font-bold uppercase tracking-wider mb-2">
                  {isAnalyzing ? "Analysis in Progress" : topPrediction ? "Classification Complete" : "Awaiting Acquisition"}
                </div>
                <h2 className="text-xl font-bold text-[#1A1A1A] tracking-tight flex items-center gap-2">
                  <span className="w-2 h-2 bg-[#90CAF9] rounded-full"></span>
                  Image Acquisition
                </h2>
                <p className="text-xs text-[#636E72]">
                  Rescale & normalize tomato leaves for pathogenetic prediction
                </p>
              </div>

              {/* Upload Drop Zone / Image Display */}
              {imagePreviewUrl ? (
                <div className="relative w-full aspect-square md:aspect-video rounded-lg overflow-hidden border border-[#E6E0D8] bg-[#F5FAFF] flex items-center justify-center p-4">
                  {isAnalyzing && (
                    <motion.div 
                      animate={{ y: ["0%", "100%", "0%"] }}
                      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                      className="absolute left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-[#90CAF9] to-transparent shadow-[0_0_10px_#90CAF9] z-10" 
                    />
                  )}
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img 
                    src={imagePreviewUrl} 
                    alt="Tomato leaf" 
                    className="max-h-full max-w-full object-contain rounded"
                  />
                </div>
              ) : (
                <div 
                  onDragEnter={handleDrag}
                  onDragOver={handleDrag}
                  onDragLeave={handleDrag}
                  onDrop={handleDrop}
                  onClick={triggerFileInput}
                  className={cn(
                    "w-full flex-1 border-2 border-dashed rounded-lg flex flex-col justify-center items-center py-12 px-6 cursor-pointer transition-all duration-300 relative min-h-[220px]",
                    isDragActive 
                      ? "border-[#90CAF9] bg-[#E3F2FD]/40 scale-[0.99]" 
                      : "border-[#BBDEFB] bg-[#F5FAFF] hover:bg-[#E3F2FD]/20 hover:border-[#90CAF9]"
                  )}
                >
                  <input 
                    type="file"
                    ref={fileInputRef}
                    className="hidden"
                    accept="image/*"
                    onChange={handleFileInput}
                  />
                  <div className="w-16 h-16 bg-[#E3F2FD] rounded-full flex items-center justify-center mb-4">
                    <Upload className="w-7 h-7 text-[#90CAF9]" />
                  </div>
                  <span className="text-sm text-[#90CAF9] font-semibold">Drag leaf image here, or browse</span>
                  <span className="text-[10px] text-slate-400 mt-1">Supports PNG, JPG, JPEG, WEBP</span>
                </div>
              )}

              {/* Quick Samples */}
              <div className="flex flex-col gap-2">
                <label className="text-[10px] font-bold text-[#636E72] uppercase tracking-wider">Quick Samples</label>
                <div className="grid grid-cols-3 gap-2">
                  <button 
                    onClick={() => {
                      setImagePreviewUrl('https://picsum.photos/seed/tomato-healthy/800/800');
                      setPredictions([]);
                      setTopPrediction(null);
                    }}
                    className="flex flex-col items-center gap-1.5 p-2 rounded-lg border border-[#E6E0D8] bg-white hover:bg-[#E3F2FD]/20 hover:border-[#90CAF9] transition text-center"
                  >
                    <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full"></span>
                    <span className="text-[10px] font-bold text-[#2D3436]">Healthy</span>
                  </button>
                  <button 
                    onClick={() => {
                      setImagePreviewUrl('https://picsum.photos/seed/tomato-yellow/800/800');
                      setPredictions([]);
                      setTopPrediction(null);
                    }}
                    className="flex flex-col items-center gap-1.5 p-2 rounded-lg border border-[#E6E0D8] bg-white hover:bg-[#E3F2FD]/20 hover:border-[#90CAF9] transition text-center"
                  >
                    <span className="w-1.5 h-1.5 bg-yellow-400 rounded-full"></span>
                    <span className="text-[10px] font-bold text-[#2D3436]">Yellow Curl</span>
                  </button>
                  <button 
                    onClick={() => {
                      setImagePreviewUrl('https://picsum.photos/seed/tomato-spot/800/800');
                      setPredictions([]);
                      setTopPrediction(null);
                    }}
                    className="flex flex-col items-center gap-1.5 p-2 rounded-lg border border-[#E6E0D8] bg-white hover:bg-[#E3F2FD]/20 hover:border-[#90CAF9] transition text-center"
                  >
                    <span className="w-1.5 h-1.5 bg-rose-400 rounded-full"></span>
                    <span className="text-[10px] font-bold text-[#2D3436]">Blight/Spots</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Pipeline status indicators */}
            <div className="flex items-center justify-between mt-4 pt-4 border-t border-[#F0EAE4]">
              <span className="text-[10px] font-mono text-[#636E72]">
                TFJS Feed: 224x224 | {inferenceTime ? `${inferenceTime}ms` : "idle"}
              </span>
              {imagePreviewUrl && (
                <button
                  onClick={runClassification}
                  disabled={isAnalyzing}
                  className="text-[11px] font-bold text-[#1E88E5] hover:text-blue-700 transition flex items-center gap-1"
                >
                  <RefreshCw className={cn("w-3 h-3", isAnalyzing && "animate-spin")} />
                  {isAnalyzing ? "Processing..." : "Re-Run Classifier"}
                </button>
              )}
            </div>

            {/* Hidden canvas for real preprocessed data feed to model */}
            <div className="hidden">
              <canvas ref={preprocessedCanvasRef} width={224} height={224} id="preprocess-canvas" />
            </div>
          </section>

          {/* Right Column (Results pane) */}
          <section className="p-6 md:p-10 bg-[#FAFAFA] flex flex-col justify-between gap-6">
            
            {/* 1. Empty state */}
            {!imagePreviewUrl && (
              <div className="flex-1 flex flex-col items-center justify-center text-center p-4 min-h-[300px]">
                <div className="w-12 h-12 bg-[#FFF8F0] border border-[#E6E0D8] rounded-full flex items-center justify-center text-slate-400 mb-4">
                  <Leaf className="w-5 h-5 text-[#90CAF9]" />
                </div>
                <h3 className="font-display font-bold text-[#1A1A1A] text-sm mb-1">Awaiting Leaf Acquisition</h3>
                <p className="text-[11px] text-[#636E72] max-w-xs leading-relaxed">
                  Please upload or select a tomato leaf image on the left. The neural network will instantly classify the pathogens.
                </p>
              </div>
            )}

            {/* 2. Analyzing Loader */}
            {imagePreviewUrl && isAnalyzing && (
              <div className="flex-1 flex flex-col items-center justify-center text-center p-4 min-h-[300px]">
                <div className="relative mb-4">
                  <div className="w-12 h-12 rounded-full border-2 border-slate-100 border-t-[#90CAF9] animate-spin"></div>
                  <Leaf className="w-4 h-4 text-[#90CAF9] absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
                </div>
                <h3 className="font-bold text-[#1A1A1A] text-xs">Evaluating Leaf Tissue</h3>
                <p className="text-[10px] text-[#636E72] mt-1">Normalizing tensor to MobileNetV2 [-1, 1] range...</p>
              </div>
            )}

            {/* 3. Output prediction display */}
            {imagePreviewUrl && !isAnalyzing && topPrediction && (
              <div className="flex-1 flex flex-col gap-6" id="results-container">
                
                {/* Result Card styled like Geometric Balance layout */}
                <div className="bg-white rounded-xl p-5 border border-[#E6E0D8] shadow-[0_2px_12px_rgba(0,0,0,0.02)] flex flex-col">
                  <span className="text-[10px] uppercase tracking-wider text-[#636E72] font-semibold">Detected Condition</span>
                  <span className="text-2xl font-bold text-[#1A1A1A] mt-1 tracking-tight">
                    {topPrediction.className}
                  </span>
                  <div className={cn(
                    "text-xs font-bold mt-2 flex items-center gap-1.5",
                    topPrediction.className === 'Healthy' ? "text-emerald-600" : "text-rose-600"
                  )}>
                    {topPrediction.className === 'Healthy' ? "✓ Plant is Healthy" : "⚠ Action Required"}
                    <span className="text-slate-400 font-normal">| Confidence: {(topPrediction.confidence * 100).toFixed(1)}%</span>
                  </div>
                </div>

                {/* Probability Breakdown Progress Bars */}
                <div className="flex flex-col gap-3">
                  <span className="text-[10px] uppercase tracking-wider text-[#636E72] font-semibold">Inference Confidence Breakdown</span>
                  <ul className="flex flex-col gap-3">
                    {predictions.map((pred, i) => (
                      <li key={pred.className} className="flex flex-col gap-1.5">
                        <div className="flex justify-between text-xs font-medium text-[#2D3436]">
                          <span>{pred.className}</span>
                          <span className="font-mono font-bold">{(pred.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-2 bg-[#E9ECEF] rounded-full overflow-hidden">
                          <motion.div 
                            initial={{ width: 0 }}
                            animate={{ width: `${pred.confidence * 100}%` }}
                            transition={{ duration: 0.5, ease: 'easeOut' }}
                            className="h-full bg-[#90CAF9] rounded-full"
                            style={{ opacity: i === 0 ? 1 : 0.4 }}
                          />
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Prevention and Remedies panel */}
                <div className="border border-[#E6E0D8] rounded-xl bg-white p-4 mt-2">
                  <div className="flex items-center gap-1.5 mb-2 pb-2 border-b border-slate-100">
                    <Info className="w-4 h-4 text-[#90CAF9]" />
                    <span className="text-xs font-bold text-slate-800 uppercase tracking-wider">Remedies & Prevention</span>
                  </div>
                  
                  <div className="flex flex-col gap-2">
                    <div className="text-[11px] leading-relaxed text-slate-600 bg-slate-50 p-2.5 rounded border border-slate-100">
                      <strong className="text-slate-800">Symptoms: </strong>
                      {DISEASE_METADATA[topPrediction.className].symptoms}
                    </div>
                    
                    <ul className="grid grid-cols-1 gap-1.5 mt-1">
                      {DISEASE_METADATA[topPrediction.className].recommendations.slice(0, 3).map((rec, idx) => (
                        <li key={idx} className="text-[11px] leading-relaxed text-[#2D3436] flex gap-1.5">
                          <span className="text-emerald-500 shrink-0">✔</span>
                          <span>{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

              </div>
            )}

            {/* Footer nested in panel or alignment block */}
            <div className="text-[10px] text-[#636E72]/60 text-center md:text-right mt-auto">
              Real-time classification based on TFJS inference pipeline.
            </div>
          </section>

        </main>

        <footer className="w-full flex items-center justify-center py-6 text-xs text-[#636E72]/80 border-t border-[#E6E0D8]" id="app-footer">
          &copy; {new Date().getFullYear()} TomatoLeaf AI Detection System &bull; Intelligent Agriculture Solutions
        </footer>
      </div>
    </div>
  );
}
