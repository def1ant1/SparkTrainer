import React, { useState, useCallback } from 'react';
import { Upload, Film, FolderOpen, CheckCircle, XCircle, Loader, Play } from 'lucide-react';
import { Button } from './components/Button';
import { Card } from './components/Card';
import { Input } from './components/Input';
import { Progress } from './components/Progress';
import { Modal } from './components/Modal';

const DatasetWizard = () => {
  const [step, setStep] = useState(1);
  const [config, setConfig] = useState({
    name: '',
    modality: 'video',
    inputPath: '',
    outputPath: '',
    fps: 1,
    resolution: '224,224',
    extractAudio: true,
    enableTranscription: true,
    whisperModel: 'base',
    enableCaptioning: true,
    captionerBackend: 'blip2',
    enableSceneDetection: true,
  });
  const [files, setFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [integrityCheck, setIntegrityCheck] = useState(null);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFiles = Array.from(e.dataTransfer.files);
    const videoFiles = droppedFiles.filter(file =>
      file.type.startsWith('video/') ||
      /\.(mp4|avi|mov|mkv|webm)$/i.test(file.name)
    );

    setFiles(prev => [...prev, ...videoFiles]);
  }, []);

  const handleFileInput = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(prev => [...prev, ...selectedFiles]);
  };

  const handleFolderSelect = async () => {
    // Use directory picker API if available
    if ('showDirectoryPicker' in window) {
      try {
        const dirHandle = await window.showDirectoryPicker();
        setConfig(prev => ({ ...prev, inputPath: dirHandle.name }));
      } catch (err) {
        console.error('Error selecting directory:', err);
      }
    } else {
      alert('Directory picker not supported. Please enter path manually.');
    }
  };

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const runIntegrityCheck = async () => {
    setProcessing(true);
    setProgress(0);

    try {
      // Simulate integrity check
      const results = files.map((file, i) => {
        const duration = Math.random() * 300 + 10; // 10-310 seconds
        const valid = Math.random() > 0.1; // 90% valid

        return {
          filename: file.name,
          size: (file.size / (1024 * 1024)).toFixed(2) + ' MB',
          duration: duration.toFixed(1) + 's',
          valid: valid,
          issues: valid ? [] : ['Corrupted frames detected', 'Audio sync issue']
        };
      });

      // Simulate progress
      for (let i = 0; i <= 100; i += 10) {
        setProgress(i);
        await new Promise(resolve => setTimeout(resolve, 200));
      }

      setIntegrityCheck(results);
      setStep(3);
    } catch (error) {
      console.error('Integrity check failed:', error);
    } finally {
      setProcessing(false);
    }
  };

  const startProcessing = async () => {
    setProcessing(true);
    setProgress(0);

    try {
      const response = await fetch('/api/datasets/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...config,
          files: files.map(f => f.name)
        })
      });

      const data = await response.json();

      // Simulate processing progress
      const interval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 95) {
            clearInterval(interval);
            return 95;
          }
          return prev + 5;
        });
      }, 1000);

      // In production, poll for actual progress
      setTimeout(() => {
        clearInterval(interval);
        setProgress(100);
        setResult(data);
        setStep(4);
        setProcessing(false);
      }, 10000);

    } catch (error) {
      console.error('Processing failed:', error);
      setProcessing(false);
    }
  };

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Dataset Wizard</h1>
          <p className="text-gray-600 dark:text-gray-400">
            Import and process video datasets with automatic extraction and annotation
          </p>
        </div>
        <div className="flex items-center gap-2">
          {[1, 2, 3, 4].map(s => (
            <div
              key={s}
              className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                s === step
                  ? 'bg-blue-500 text-white'
                  : s < step
                  ? 'bg-green-500 text-white'
                  : 'bg-gray-300 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
              }`}
            >
              {s < step ? <CheckCircle size={16} /> : s}
            </div>
          ))}
        </div>
      </div>

      {/* Step 1: Upload / Select Files */}
      {step === 1 && (
        <Card>
          <div className="p-6 space-y-6">
            <div>
              <h2 className="text-xl font-semibold mb-4">Step 1: Select Video Files</h2>
              <div className="grid grid-cols-2 gap-4 mb-4">
                <Input
                  label="Dataset Name"
                  value={config.name}
                  onChange={(e) => setConfig({ ...config, name: e.target.value })}
                  placeholder="my-video-dataset"
                />
                <div>
                  <label className="block text-sm font-medium mb-2">Modality</label>
                  <select
                    value={config.modality}
                    onChange={(e) => setConfig({ ...config, modality: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800"
                  >
                    <option value="video">Video</option>
                    <option value="image">Image</option>
                    <option value="audio">Audio</option>
                    <option value="multimodal">Multimodal</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Drag & Drop Zone */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
                isDragging
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-300 dark:border-gray-700'
              }`}
            >
              <Upload className="mx-auto mb-4 text-gray-400" size={48} />
              <p className="text-lg font-medium mb-2">Drag and drop video files here</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                Supports MP4, AVI, MOV, MKV, WEBM
              </p>
              <div className="flex items-center justify-center gap-4">
                <Button onClick={() => document.getElementById('fileInput').click()}>
                  <Upload size={16} />
                  Choose Files
                </Button>
                <Button variant="secondary" onClick={handleFolderSelect}>
                  <FolderOpen size={16} />
                  Select Folder
                </Button>
              </div>
              <input
                id="fileInput"
                type="file"
                multiple
                accept="video/*"
                onChange={handleFileInput}
                className="hidden"
              />
            </div>

            {/* File List */}
            {files.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-medium">Selected Files ({files.length})</h3>
                  <span className="text-sm text-gray-600">
                    Total: {formatBytes(files.reduce((sum, f) => sum + f.size, 0))}
                  </span>
                </div>
                <div className="max-h-64 overflow-y-auto space-y-2">
                  {files.map((file, i) => (
                    <div key={i} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                      <div className="flex items-center gap-3">
                        <Film size={20} className="text-blue-500" />
                        <div>
                          <div className="font-medium">{file.name}</div>
                          <div className="text-sm text-gray-600">{formatBytes(file.size)}</div>
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => removeFile(i)}
                      >
                        <XCircle size={16} />
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="flex justify-end">
              <Button
                onClick={() => setStep(2)}
                disabled={files.length === 0 || !config.name}
              >
                Next: Configure Processing
              </Button>
            </div>
          </div>
        </Card>
      )}

      {/* Step 2: Configuration */}
      {step === 2 && (
        <Card>
          <div className="p-6 space-y-6">
            <h2 className="text-xl font-semibold">Step 2: Configure Processing</h2>

            <div className="grid grid-cols-2 gap-6">
              {/* Frame Extraction */}
              <div className="space-y-4">
                <h3 className="font-medium">Frame Extraction</h3>
                <Input
                  label="Frames per Second (FPS)"
                  type="number"
                  value={config.fps}
                  onChange={(e) => setConfig({ ...config, fps: parseInt(e.target.value) })}
                  min="1"
                  max="30"
                />
                <Input
                  label="Resolution (width,height)"
                  value={config.resolution}
                  onChange={(e) => setConfig({ ...config, resolution: e.target.value })}
                  placeholder="224,224"
                />
              </div>

              {/* Audio Processing */}
              <div className="space-y-4">
                <h3 className="font-medium">Audio Processing</h3>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={config.extractAudio}
                    onChange={(e) => setConfig({ ...config, extractAudio: e.target.checked })}
                    className="rounded"
                  />
                  <span>Extract Audio</span>
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={config.enableTranscription}
                    onChange={(e) => setConfig({ ...config, enableTranscription: e.target.checked })}
                    className="rounded"
                    disabled={!config.extractAudio}
                  />
                  <span>Enable Transcription (Whisper)</span>
                </label>
                {config.enableTranscription && (
                  <select
                    value={config.whisperModel}
                    onChange={(e) => setConfig({ ...config, whisperModel: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800"
                  >
                    <option value="tiny">Tiny (fastest)</option>
                    <option value="base">Base</option>
                    <option value="small">Small</option>
                    <option value="medium">Medium</option>
                    <option value="large">Large (best quality)</option>
                  </select>
                )}
              </div>

              {/* Image Captioning */}
              <div className="space-y-4">
                <h3 className="font-medium">Image Captioning</h3>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={config.enableCaptioning}
                    onChange={(e) => setConfig({ ...config, enableCaptioning: e.target.checked })}
                    className="rounded"
                  />
                  <span>Enable Captioning</span>
                </label>
                {config.enableCaptioning && (
                  <select
                    value={config.captionerBackend}
                    onChange={(e) => setConfig({ ...config, captionerBackend: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800"
                  >
                    <option value="blip2">BLIP-2</option>
                    <option value="internvl">InternVL</option>
                    <option value="qwen2vl">Qwen2-VL</option>
                    <option value="florence2">Florence-2</option>
                  </select>
                )}
              </div>

              {/* Scene Detection */}
              <div className="space-y-4">
                <h3 className="font-medium">Advanced Options</h3>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={config.enableSceneDetection}
                    onChange={(e) => setConfig({ ...config, enableSceneDetection: e.target.checked })}
                    className="rounded"
                  />
                  <span>Enable Scene Detection</span>
                </label>
              </div>
            </div>

            <div className="flex justify-between pt-4">
              <Button variant="secondary" onClick={() => setStep(1)}>
                Back
              </Button>
              <Button onClick={runIntegrityCheck} disabled={processing}>
                {processing ? (
                  <>
                    <Loader className="animate-spin" size={16} />
                    Running Checks...
                  </>
                ) : (
                  'Next: Integrity Check'
                )}
              </Button>
            </div>
          </div>
        </Card>
      )}

      {/* Step 3: Integrity Check */}
      {step === 3 && (
        <Card>
          <div className="p-6 space-y-6">
            <h2 className="text-xl font-semibold">Step 3: Integrity Check Results</h2>

            {processing ? (
              <div className="text-center py-12">
                <Loader className="animate-spin mx-auto mb-4" size={48} />
                <p className="text-lg">Checking file integrity...</p>
                <Progress value={progress} className="mt-4" />
              </div>
            ) : integrityCheck ? (
              <div className="space-y-2">
                {integrityCheck.map((result, i) => (
                  <div
                    key={i}
                    className={`p-4 rounded-lg border ${
                      result.valid
                        ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                        : 'border-red-500 bg-red-50 dark:bg-red-900/20'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-3">
                        {result.valid ? (
                          <CheckCircle className="text-green-500 mt-1" size={20} />
                        ) : (
                          <XCircle className="text-red-500 mt-1" size={20} />
                        )}
                        <div>
                          <div className="font-medium">{result.filename}</div>
                          <div className="text-sm text-gray-600 mt-1">
                            {result.size} • {result.duration}
                          </div>
                          {!result.valid && result.issues.length > 0 && (
                            <ul className="text-sm text-red-600 dark:text-red-400 mt-2 list-disc list-inside">
                              {result.issues.map((issue, j) => (
                                <li key={j}>{issue}</li>
                              ))}
                            </ul>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}

                <div className="flex items-center justify-between pt-4 mt-6 border-t">
                  <div className="text-sm">
                    <span className="font-medium">Summary:</span>{' '}
                    <span className="text-green-600">{integrityCheck.filter(r => r.valid).length} valid</span>
                    {' • '}
                    <span className="text-red-600">{integrityCheck.filter(r => !r.valid).length} invalid</span>
                  </div>
                  <div className="flex gap-2">
                    <Button variant="secondary" onClick={() => setStep(2)}>
                      Back
                    </Button>
                    <Button onClick={startProcessing}>
                      <Play size={16} />
                      Start Processing
                    </Button>
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        </Card>
      )}

      {/* Step 4: Processing */}
      {step === 4 && (
        <Card>
          <div className="p-6 space-y-6">
            <h2 className="text-xl font-semibold">Step 4: Processing Dataset</h2>

            {processing ? (
              <div className="text-center py-12">
                <Loader className="animate-spin mx-auto mb-4 text-blue-500" size={64} />
                <p className="text-xl font-medium mb-2">Processing videos...</p>
                <p className="text-gray-600 mb-4">
                  Extracting frames, audio, generating captions and transcriptions
                </p>
                <Progress value={progress} className="mt-4" />
                <p className="text-sm text-gray-600 mt-2">{progress}% complete</p>
              </div>
            ) : result ? (
              <div className="space-y-6">
                <div className="flex items-center justify-center py-8">
                  <CheckCircle className="text-green-500" size={64} />
                </div>
                <div className="text-center">
                  <h3 className="text-2xl font-bold text-green-600 mb-2">Processing Complete!</h3>
                  <p className="text-gray-600">Your dataset has been successfully processed</p>
                </div>

                <div className="grid grid-cols-3 gap-4">
                  <Card>
                    <div className="p-4 text-center">
                      <div className="text-3xl font-bold text-blue-500">
                        {result.totalVideos || files.length}
                      </div>
                      <div className="text-sm text-gray-600 mt-1">Videos Processed</div>
                    </div>
                  </Card>
                  <Card>
                    <div className="p-4 text-center">
                      <div className="text-3xl font-bold text-green-500">
                        {result.totalFrames || files.length * 100}
                      </div>
                      <div className="text-sm text-gray-600 mt-1">Frames Extracted</div>
                    </div>
                  </Card>
                  <Card>
                    <div className="p-4 text-center">
                      <div className="text-3xl font-bold text-purple-500">
                        {result.totalCaptions || files.length * 100}
                      </div>
                      <div className="text-sm text-gray-600 mt-1">Captions Generated</div>
                    </div>
                  </Card>
                </div>

                <div className="flex justify-center gap-4 pt-4">
                  <Button variant="secondary" onClick={() => window.location.href = '/datasets'}>
                    View Datasets
                  </Button>
                  <Button onClick={() => {
                    setStep(1);
                    setFiles([]);
                    setResult(null);
                    setIntegrityCheck(null);
                  }}>
                    Process Another Dataset
                  </Button>
                </div>
              </div>
            ) : null}
          </div>
        </Card>
      )}
    </div>
  );
};

export default DatasetWizard;
