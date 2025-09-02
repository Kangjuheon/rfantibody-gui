import { useState } from "react";
import { Button } from "./components/ui/button";
import { Input } from "./components/ui/input";
import { Label } from "./components/ui/label";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "./components/ui/select";
import {
  Card, CardContent, CardHeader, CardTitle,
} from "./components/ui/card";
import {
  Tooltip, TooltipContent, TooltipProvider, TooltipTrigger,
} from "./components/ui/tooltip";
import { Badge } from "./components/ui/badge";
import { FileUpload } from "./components/FileUpload";
import { Download, Info, Play, Loader2 } from "lucide-react";

export default function App() {
  const [jobName, setJobName] = useState("");
  const [mode, setMode] = useState("Antibody");
  const [frameworkFile, setFrameworkFile] = useState<File | null>(null);
  const [targetFile, setTargetFile] = useState<File | null>(null);
  const [hotspots, setHotspots] = useState("");
  const [hotspotsIsPlaceholder, setHotspotsIsPlaceholder] = useState(true);
  const [rfDiffusionDesigns, setRfDiffusionDesigns] = useState("1");
  const [proteinMPNNDesigns, setProteinMPNNDesigns] = useState("1");
  const [designLoops, setDesignLoops] = useState("");
  const [designLoopsIsPlaceholder, setDesignLoopsIsPlaceholder] = useState(true);
  const [isRunning, setIsRunning] = useState(false);

  // NEW: API 결과/에러 상태
  const [result, setResult] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Default placeholders
  const defaultHotspots = "A21, B14-21";
  const defaultDesignLoops = "H1_, H2.7, H3.5-13, L1.10-15, L2.10-15, L3.10-15";

  const generateDefaultJobName = () => {
    const now = new Date();
    const timestamp = now.toISOString().slice(0, 19).replace(/[:.]/g, "-");
    return `RFantibody_${timestamp}`;
  };

  const getEffectiveJobName = () => jobName.trim() || generateDefaultJobName();
  const getEffectiveHotspots = () => (hotspotsIsPlaceholder ? defaultHotspots : hotspots);
  const getEffectiveDesignLoops = () => (designLoopsIsPlaceholder ? defaultDesignLoops : designLoops);

  const isFormValid = () => {
    const effectiveHotspots = getEffectiveHotspots();
    return (
      frameworkFile !== null &&
      targetFile !== null &&
      effectiveHotspots.trim() !== "" &&
      parseInt(rfDiffusionDesigns) > 0 &&
      parseInt(proteinMPNNDesigns) > 0
    );
  };

  // NEW: API 베이스 (Vite 프록시 기본 /api)
  const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api";

  const handleRunJob = async () => {
    if (!isFormValid()) return;
    setIsRunning(true);
    setResult(null);
    setError(null);

    const effectiveJobName = getEffectiveJobName();

    try {
      const fd = new FormData();
      fd.append("jobName", effectiveJobName);
      fd.append("mode", mode);
      fd.append("hotspots", getEffectiveHotspots());
      fd.append("rfDiffusionDesigns", String(parseInt(rfDiffusionDesigns)));
      fd.append("proteinMPNNDesigns", String(parseInt(proteinMPNNDesigns)));
      fd.append("designLoops", getEffectiveDesignLoops());

      if (frameworkFile) fd.append("frameworkFile", frameworkFile, frameworkFile.name);
      if (targetFile) fd.append("targetFile", targetFile, targetFile.name);

      const resp = await fetch(`${API_BASE}/rfantibody_pipeline`, {
        method: "POST",
        body: fd,
        // Content-Type는 FormData가 자동 설정 (절대 수동 설정하지 말 것)
      });

      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`HTTP ${resp.status} - ${text}`);
      }
      const data = await resp.json();
      setResult(data);
    } catch (err: any) {
      setError(err?.message || "Unknown error");
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <TooltipProvider>
      <div className="min-h-screen bg-background relative">
        {/* Top decorative band */}
        <div className="absolute top-0 left-0 right-0 h-3 bg-gradient-to-r from-blue-500/30 via-purple-600/40 to-blue-500/30 shadow-sm"></div>
        {/* Bottom decorative band */}
        <div className="absolute bottom-0 left-0 right-0 h-3 bg-gradient-to-r from-purple-600/30 via-blue-500/40 to-purple-600/30 shadow-sm"></div>

        <div className="max-w-4xl mx-auto px-6 py-8 relative z-10">
          {/* Header */}
          <div className="mb-8 relative">
            <div className="absolute top-0 right-0">
              <Badge variant="secondary" className="text-xs">v1.0.7</Badge>
            </div>
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              <h1 className="mb-2 text-4xl font-bold">Use RFantibody</h1>
            </div>
          </div>

          {/* Job Name */}
          <Card className="mb-6">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2">
                Job Information
                <Tooltip>
                  <TooltipTrigger><Info className="w-4 h-4 text-muted-foreground" /></TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">
                      Provide a name or description for your job to help you organize and track its results.
                    </p>
                  </TooltipContent>
                </Tooltip>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <Label htmlFor="jobName">Job Name / Description</Label>
                <Input
                  id="jobName"
                  placeholder={`Enter job name (default: ${generateDefaultJobName()})`}
                  value={jobName}
                  onChange={(e) => setJobName(e.target.value)}
                  className="w-full"
                />
                {!jobName.trim() && (
                  <p className="text-xs text-muted-foreground">
                    Will use default name: <span className="font-mono">{generateDefaultJobName()}</span>
                  </p>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Service Configuration */}
          <Card className="mb-6">
            <CardHeader className="pb-3">
              <CardTitle>Service Configuration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Mode */}
              <div className="space-y-2">
                <Label className="flex items-center gap-2">
                  Mode
                  <Tooltip>
                    <TooltipTrigger><Info className="w-4 h-4 text-muted-foreground" /></TooltipTrigger>
                    <TooltipContent><p>Select the mode of the design.</p></TooltipContent>
                  </Tooltip>
                </Label>
                <Select value={mode} onValueChange={setMode}>
                  <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Antibody">Antibody</SelectItem>
                    <SelectItem value="Nanobody">Nanobody</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Input Framework */}
              <div className="space-y-2">
                <Label className="flex items-center gap-2">
                  Input Framework
                  <Tooltip>
                    <TooltipTrigger><Info className="w-4 h-4 text-muted-foreground" /></TooltipTrigger>
                    <TooltipContent><p className="max-w-xs">Input framework template.</p></TooltipContent>
                  </Tooltip>
                </Label>
                <FileUpload
                  onFileSelect={setFrameworkFile}
                  selectedFile={frameworkFile}
                  label="Select Framework Structure"
                />
              </div>

              {/* Input Target */}
              <div className="space-y-2">
                <Label className="flex items-center gap-2">
                  Input Target
                  <Tooltip>
                    <TooltipTrigger><Info className="w-4 h-4 text-muted-foreground" /></TooltipTrigger>
                    <TooltipContent><p>Input target protein for the design.</p></TooltipContent>
                  </Tooltip>
                </Label>
                <FileUpload
                  onFileSelect={setTargetFile}
                  selectedFile={targetFile}
                  label="Select Target Structure"
                />
              </div>

              {/* Hotspots */}
              <div className="space-y-2">
                <Label htmlFor="hotspots" className="flex items-center gap-2">
                  Hotspots
                  <Tooltip>
                    <TooltipTrigger><Info className="w-4 h-4 text-muted-foreground" /></TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-sm">Comma-separated. Example: A21, B14-21</p>
                    </TooltipContent>
                  </Tooltip>
                </Label>
                <Input
                  id="hotspots"
                  placeholder="e.g., A21, B14-21"
                  autoComplete="off"
                  value={hotspotsIsPlaceholder ? defaultHotspots : hotspots}
                  onFocus={() => { if (hotspotsIsPlaceholder) { setHotspots(""); setHotspotsIsPlaceholder(false); } }}
                  onChange={(e) => { setHotspots(e.target.value); setHotspotsIsPlaceholder(false); }}
                  onBlur={(e) => { if (e.target.value.trim() === "") setHotspotsIsPlaceholder(true); }}
                  className={`w-full ${hotspotsIsPlaceholder ? "text-muted-foreground" : ""}`}
                />
              </div>

              {/* Design Loops */}
              <div className="space-y-2">
                <Label htmlFor="designLoops" className="flex items-center gap-2">
                  Design Loops
                  <Tooltip>
                    <TooltipTrigger><Info className="w-4 h-4 text-muted-foreground" /></TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-md">Examples: H1, L2:7, H3:5-13</p>
                    </TooltipContent>
                  </Tooltip>
                </Label>
                <Input
                  id="designLoops"
                  placeholder="e.g., H1, H2:7, H3:5-13, L1:10-15"
                  autoComplete="off"
                  value={designLoopsIsPlaceholder ? defaultDesignLoops : designLoops}
                  onFocus={() => { if (designLoopsIsPlaceholder) { setDesignLoops(""); setDesignLoopsIsPlaceholder(false); } }}
                  onChange={(e) => { setDesignLoops(e.target.value); setDesignLoopsIsPlaceholder(false); }}
                  onBlur={(e) => { if (e.target.value.trim() === "") setDesignLoopsIsPlaceholder(true); }}
                  className={`w-full ${designLoopsIsPlaceholder ? "text-muted-foreground" : ""}`}
                />
              </div>
            </CardContent>
          </Card>

          {/* Design Parameters */}
          <Card className="mb-6">
            <CardHeader className="pb-3"><CardTitle>Design Parameters</CardTitle></CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="rfDiffusion" className="flex items-center gap-2">
                  RFdiffusion Backbone Designs
                  <Tooltip><TooltipTrigger><Info className="w-4 h-4 text-muted-foreground" /></TooltipTrigger>
                    <TooltipContent><p className="max-w-sm">Number of backbones via RFdiffusion.</p></TooltipContent>
                  </Tooltip>
                </Label>
                <Input
                  id="rfDiffusion"
                  type="number"
                  min="1"
                  value={rfDiffusionDesigns}
                  onChange={(e) => setRfDiffusionDesigns(e.target.value)}
                  className="w-full"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="proteinMPNN" className="flex items-center gap-2">
                  ProteinMPNN Designs
                  <Tooltip><TooltipTrigger><Info className="w-4 h-4 text-muted-foreground" /></TooltipTrigger>
                    <TooltipContent><p className="max-w-sm">Sequences per backbone.</p></TooltipContent>
                  </Tooltip>
                </Label>
                <Input
                  id="proteinMPNN"
                  type="number"
                  min="1"
                  value={proteinMPNNDesigns}
                  onChange={(e) => setProteinMPNNDesigns(e.target.value)}
                  className="w-full"
                />
              </div>
            </CardContent>
          </Card>

          {/* Submit */}
          <Card className={`border-2 ${isFormValid() ? "border-primary/30 bg-primary/5" : "border-dashed border-muted-foreground/20"}`}>
            <CardContent className="p-8">
              <div className="text-center space-y-4">
                <div className={`w-16 h-16 mx-auto rounded-full flex items-center justify-center ${isFormValid() ? "bg-primary/20" : "bg-muted"}`}>
                  {isRunning
                    ? <Loader2 className="w-8 h-8 text-primary animate-spin" />
                    : <Play className={`w-8 h-8 ${isFormValid() ? "text-primary" : "text-muted-foreground"}`} />
                  }
                </div>
                <div>
                  <h3 className="mb-2">{isFormValid() ? "Ready to Run Job" : "Complete Configuration"}</h3>
                  <p className="text-muted-foreground max-w-md mx-auto">
                    {isFormValid()
                      ? "All required fields are completed. Click below to start the protein design process."
                      : "Please fill in all required fields to proceed with the job."}
                  </p>
                  {!isFormValid() && (
                    <div className="mt-3 text-sm space-y-1">
                      <p className="text-destructive">Missing required fields:</p>
                      <ul className="text-muted-foreground space-y-1">
                        {!frameworkFile && <li>• Framework Structure</li>}
                        {!targetFile && <li>• Target Structure</li>}
                        {getEffectiveHotspots().trim() === "" && <li>• Hotspots</li>}
                      </ul>
                    </div>
                  )}
                </div>
                <div className="pt-2">
                  <Button size="lg" className="px-8" disabled={!isFormValid() || isRunning} onClick={handleRunJob}>
                    {isRunning ? (<><Loader2 className="w-4 h-4 mr-2 animate-spin" />Submitting...</>)
                              : (<><Play className="w-4 h-4 mr-2" />Run Job</>)}
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* 결과 표시 */}
          {(error || result) && (
            <Card className="mt-6">
              <CardHeader><CardTitle>Pipeline Result</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                {error && (
                  <div className="text-sm text-destructive">
                    Error: {error}
                  </div>
                )}
                {result && (
                  <>
                    {/* 간단 요약 */}
                    <div className="text-sm text-muted-foreground">
                      Status: <span className="font-medium text-foreground">{result.status}</span>
                      {result.job?.jobName ? <> · Job: <span className="font-mono">{result.job.jobName}</span></> : null}
                    </div>

                    {/* 결과 JSON */}
                    <pre className="rounded-md border bg-muted p-3 text-xs overflow-auto max-h-80">
{JSON.stringify(result, null, 2)}
                    </pre>

                    {/* 아티팩트가 URL 리스트로 온 경우 다운로드 버튼 예시 */}
                    {Array.isArray(result.artifacts) && result.artifacts.length > 0 && (
                      <div className="flex flex-wrap gap-2">
                        {result.artifacts.map((url: string, i: number) => (
                          <a key={i} href={url} target="_blank" rel="noreferrer">
                            <Button variant="secondary" className="h-8">
                              <Download className="w-4 h-4 mr-2" />
                              Download {i + 1}
                            </Button>
                          </a>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </CardContent>
            </Card>
          )}

          {/* Footer */}
          <div className="mt-12 flex justify-end">
            <div className="bg-background/80 backdrop-blur-sm px-3 py-2 rounded-lg border border-border/30 shadow-sm">
              <span className="text-xs text-muted-foreground">
                Developed by University of Seoul & KAIST
              </span>
            </div>
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}
