// App.tsx
import { useState } from "react";
import { Button } from "./components/ui/button";
import { Input } from "./components/ui/input";
import { Label } from "./components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./components/ui/select";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "./components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./components/ui/tooltip";
import { Badge } from "./components/ui/badge";
import { FileUpload } from "./components/FileUpload";
import { Download, Info, Play, Loader2 } from "lucide-react";

export default function App() {
  const [jobName, setJobName] = useState("");
  const [mode, setMode] = useState("Antibody");
  const [frameworkFile, setFrameworkFile] = useState<File | null>(null);
  const [targetFile, setTargetFile] = useState<File | null>(null);

  // Hotspots optional (default: none)
  const [hotspots, setHotspots] = useState("");
  const [hotspotsIsPlaceholder, setHotspotsIsPlaceholder] = useState(true);

  const [rfDiffusionDesigns, setRfDiffusionDesigns] = useState("1");
  const [designLoops, setDesignLoops] = useState("");
  const [designLoopsIsPlaceholder, setDesignLoopsIsPlaceholder] = useState(true);
  const [isRunning, setIsRunning] = useState(false);

  // API 결과/에러 상태
  const [result, setResult] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);

  // RFdiffusion 세부 파라미터
  const [rfDiffusionFinalStep, setRfDiffusionFinalStep] = useState("50");
  const [rfDiffusionDeterministic, setRfDiffusionDeterministic] = useState(false);
  const [rfDiffusionDiffuserT, setRfDiffusionDiffuserT] = useState("50");

  // NEW: Num MPNN Sequence 파라미터
  const [proteinMPNNDesigns, setProteinMPNNDesigns] = useState("1");

  // Placeholders
  const defaultHotspots = ""; // none by default
  const defaultDesignLoops = "H1:7,H2:6,H3:5-13";

  const generateDefaultJobName = () => {
    const now = new Date();
    const timestamp = now.toISOString().slice(0, 19).replace(/[:.]/g, "-");
    return `RFantibody_${timestamp}`;
  };

  const getEffectiveJobName = () => jobName.trim() || generateDefaultJobName();
  const getEffectiveHotspots = () =>
    hotspotsIsPlaceholder ? defaultHotspots : hotspots;
  const getEffectiveDesignLoops = () =>
    designLoopsIsPlaceholder ? defaultDesignLoops : designLoops;

  const isFormValid = () => {
    // Hotspots are OPTIONAL
    return (
      frameworkFile !== null &&
      targetFile !== null &&
      parseInt(rfDiffusionDesigns) > 0
    );
  };

  // API 베이스 (Vite 프록시 기본 /api)
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

      // Hotspots: omit when empty to represent "null/none"
      const effHotspots = getEffectiveHotspots().trim();
      if (effHotspots !== "") {
        fd.append("hotspots", effHotspots);
      }

      fd.append("rfDiffusionDesigns", String(parseInt(rfDiffusionDesigns)));
      fd.append("designLoops", getEffectiveDesignLoops());

      // RFdiffusion 세부 파라미터
      fd.append("rfDiffusionFinalStep", String(parseInt(rfDiffusionFinalStep)));
      fd.append("rfDiffusionDeterministic", String(rfDiffusionDeterministic));
      fd.append("rfDiffusionDiffuserT", String(parseInt(rfDiffusionDiffuserT)));

      // NEW: Num MPNN Sequence
      fd.append("proteinMPNNDesigns", String(parseInt(proteinMPNNDesigns)));

      if (frameworkFile)
        fd.append("frameworkFile", frameworkFile, frameworkFile.name);
      if (targetFile)
        fd.append("targetFile", targetFile, targetFile.name);

      const resp = await fetch(`${API_BASE}/rfantibody_pipeline`, {
        method: "POST",
        body: fd,
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
              <Badge variant="secondary" className="text-xs">
                v1.0.9
              </Badge>
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
                  <TooltipTrigger>
                    <Info className="w-4 h-4 text-muted-foreground" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">
                      Provide a name or description for your job to help you
                      organize and track its results. This input is solely for
                      organizational purposes and does not impact the outcome of the job.
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
                    Will use default name:{" "}
                    <span className="font-mono">{generateDefaultJobName()}</span>
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
                    <TooltipTrigger>
                      <Info className="w-4 h-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Select the mode of the design.</p>
                    </TooltipContent>
                  </Tooltip>
                </Label>
                <Select value={mode} onValueChange={setMode}>
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
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
                    <TooltipTrigger>
                      <Info className="w-4 h-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-xs">
                        Input framework template for the design. The expected type
                        (Antibody or Nanobody) depends on the selected 'Mode'.
                      </p>
                    </TooltipContent>
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
                    <TooltipTrigger>
                      <Info className="w-4 h-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Input target protein for the design.</p>
                    </TooltipContent>
                  </Tooltip>
                </Label>
                <FileUpload
                  onFileSelect={setTargetFile}
                  selectedFile={targetFile}
                  label="Select Target Structure"
                />
              </div>

              {/* Hotspots (Optional) */}
              <div className="space-y-2">
                <Label htmlFor="hotspots" className="flex items-center gap-2">
                  Hotspots <span className="text-xs text-muted-foreground">(optional)</span>
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="w-4 h-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-sm">
                        Optional key residues on the target protein to guide binding.
                        Provide a comma-separated list. Each item can be a single
                        residue (e.g., 'A21') or a range (e.g., 'B14-21').
                        Leave empty to design without explicit hotspots. Default is none.
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </Label>
                <Input
                  id="hotspots"
                  placeholder=""
                  autoComplete="off"
                  value={hotspotsIsPlaceholder ? defaultHotspots : hotspots}
                  onFocus={() => {
                    if (hotspotsIsPlaceholder) {
                      setHotspots("");
                      setHotspotsIsPlaceholder(false);
                    }
                  }}
                  onChange={(e) => {
                    setHotspots(e.target.value);
                    setHotspotsIsPlaceholder(false);
                  }}
                  onBlur={(e) => {
                    if (e.target.value.trim() === "") setHotspotsIsPlaceholder(true);
                  }}
                  className={`w-full ${hotspotsIsPlaceholder ? "text-muted-foreground" : ""}`}
                />
              </div>

              {/* Design Loops */}
              <div className="space-y-2">
                <Label htmlFor="designLoops" className="flex items-center gap-2">
                  Design Loops
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="w-4 h-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-md">
                        Specify which CDR loops (H1, H2, H3, L1, L2, L3) to design
                        and optionally constrain their lengths. Each item follows the
                        format: 'LOOP', 'LOOP:LENGTH' or 'LOOP:START-END'. Examples:
                        'H1', 'L2:7', 'H3:5-13'. If omitted, loops remain fixed from
                        the input framework. Default is H1:7,H2:6,H3:5-13.
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </Label>
                <Input
                  id="designLoops"
                  placeholder=""
                  autoComplete="off"
                  value={designLoopsIsPlaceholder ? defaultDesignLoops : designLoops}
                  onFocus={() => {
                    if (designLoopsIsPlaceholder) {
                      setDesignLoops("");
                      setDesignLoopsIsPlaceholder(false);
                    }
                  }}
                  onChange={(e) => {
                    setDesignLoops(e.target.value);
                    setDesignLoopsIsPlaceholder(false);
                  }}
                  onBlur={(e) => {
                    if (e.target.value.trim() === "") setDesignLoopsIsPlaceholder(true);
                  }}
                  className={`w-full ${designLoopsIsPlaceholder ? "text-muted-foreground" : ""}`}
                />
              </div>
            </CardContent>
          </Card>

          {/* Design Parameters */}
          <Card className="mb-6">
            <CardHeader className="pb-3">
              <CardTitle>Design Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* RFdiffusion Backbone Designs */}
              <div className="space-y-2">
                <Label htmlFor="rfDiffusion" className="flex items-center gap-2">
                  RFdiffusion Backbone Designs
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="w-4 h-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-sm">
                        Number of distinct backbone structures to generate using RFdiffusion.
                      </p>
                    </TooltipContent>
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

              {/* rfDiffusion Final Step */}
              <div className="space-y-2">
                <Label htmlFor="rfDiffusionFinalStep" className="flex items-center gap-2">
                  RFdiffusion Final Step
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="w-4 h-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-sm">
                        Number of refinement steps for RFdiffusion. Higher values may
                        improve quality but increase runtime.
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </Label>
                <Input
                  id="rfDiffusionFinalStep"
                  type="number"
                  min="1"
                  value={rfDiffusionFinalStep}
                  onChange={(e) => setRfDiffusionFinalStep(e.target.value)}
                  className="w-full"
                />
              </div>

              {/* rfDiffusion Deterministic */}
              <div className="space-y-2">
                <Label htmlFor="rfDiffusionDeterministic" className="flex items-center gap-2">
                  RFdiffusion Deterministic
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="w-4 h-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-sm">
                        If enabled, results will be deterministic (reproducible).
                        Default is False.
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </Label>
                <input
                  id="rfDiffusionDeterministic"
                  type="checkbox"
                  checked={rfDiffusionDeterministic}
                  onChange={(e) => setRfDiffusionDeterministic(e.target.checked)}
                  className="h-4 w-4"
                />
              </div>

              {/* rfDiffusion Diffuser T */}
              <div className="space-y-2">
                <Label htmlFor="rfDiffusionDiffuserT" className="flex items-center gap-2">
                  RFdiffusion Diffuser T
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="w-4 h-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-sm">
                        Number of diffusion timesteps (T) used by RFdiffusion. Default is 50.
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </Label>
                <Input
                  id="rfDiffusionDiffuserT"
                  type="number"
                  min="1"
                  value={rfDiffusionDiffuserT}
                  onChange={(e) => setRfDiffusionDiffuserT(e.target.value)}
                  className="w-full"
                />
              </div>

              {/* NEW: Num MPNN Sequence */}
              <div className="space-y-2">
                <Label htmlFor="proteinMPNNDesigns" className="flex items-center gap-2">
                  Num MPNN Sequence
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="w-4 h-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-sm">
                        Number of sequences to sample with ProteinMPNN for each backbone.
                        Set higher for more sequence diversity. Minimum 1.
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </Label>
                <Input
                  id="proteinMPNNDesigns"
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
          <Card
            className={`border-2 ${
              isFormValid()
                ? "border-primary/30 bg-primary/5"
                : "border-dashed border-muted-foreground/20"
            }`}
          >
            <CardContent className="p-8">
              <div className="text-center space-y-4">
                <div
                  className={`w-16 h-16 mx-auto rounded-full flex items-center justify-center ${
                    isFormValid() ? "bg-primary/20" : "bg-muted"
                  }`}
                >
                  {isRunning ? (
                    <Loader2 className="w-8 h-8 text-primary animate-spin" />
                  ) : (
                    <Play
                      className={`w-8 h-8 ${
                        isFormValid() ? "text-primary" : "text-muted-foreground"
                      }`}
                    />
                  )}
                </div>
                <div>
                  <h3 className="mb-2">
                    {isFormValid() ? "Ready to Run Job" : "Complete Configuration"}
                  </h3>
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
                      </ul>
                    </div>
                  )}
                </div>
                <div className="pt-2">
                  <Button
                    size="lg"
                    className="px-8"
                    disabled={!isFormValid() || isRunning}
                    onClick={handleRunJob}
                  >
                    {isRunning ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Submitting...
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4 mr-2" />
                        Run Job
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* 결과 표시 */}
          {(error || result) && (
            <Card className="mt-6">
              <CardHeader>
                <CardTitle>Pipeline Result</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {error && <div className="text-sm text-destructive">Error: {error}</div>}
                {result && (
                  <>
                    {/* 간단 요약 */}
                    <div className="text-sm text-muted-foreground">
                      Status:{" "}
                      <span className="font-medium text-foreground">
                        {result.status}
                      </span>
                      {result.job?.jobName ? (
                        <>
                          {" "}· Job:{" "}
                          <span className="font-mono">{result.job.jobName}</span>
                        </>
                      ) : null}
                    </div>

                    {/* 결과 JSON */}
                    <pre className="rounded-md border bg-muted p-3 text-xs overflow-auto max-h-80">
                      {JSON.stringify(result, null, 2)}
                    </pre>

                    {result.links?.download && (
                      <div className="flex flex-wrap gap-2">
                        {result.links.download.jobZip && (
                          <a
                            href={`${API_BASE}${result.links.download.jobZip}`}
                            target="_blank"
                            rel="noreferrer"
                          >
                            <Button variant="secondary" className="h-8">
                              <Download className="w-4 h-4 mr-2" />
                              Download Job ZIP
                            </Button>
                          </a>
                        )}
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
              <span className="text-xs text-muted-foreground">Developed by KandO</span>
            </div>
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}
