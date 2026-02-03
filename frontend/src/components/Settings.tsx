import React, { useState, useEffect } from "react";
import {
  Container,
  Box,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Button,
  Paper,
  Alert,
  CircularProgress,
  Snackbar,
  Tabs,
  Tab,
} from "@mui/material";
import axios from "axios";
import { useNavigate } from "react-router-dom";

/** Single OpenAI model option; always requires user API key. */
const OPENAI_MODEL = {
  value: "openai/gpt-5-nano-2025-08-07",
  label: "gpt-5-nano-2025-08-07 (OpenAI - API key required)",
};

const DEFAULT_PROMPTS = {
  summarize: "Summarize the following text for efficacy and clarity.",
  qa: "Given the following text, answer the question as accurately as possible.",
};

const API_URL = (window as any).APP_CONFIG?.API_URL || 'http://localhost:8001';

function getAuthHeaders(): Record<string, string> {
  const token = localStorage.getItem("token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

const Settings: React.FC = () => {
  const [pdfParser, setPdfParser] = useState<string>("tesseract");
  const [selectedModel, setSelectedModel] = useState<string>("ollama/gemma3:1b");
  const [llmModelOptions, setLlmModelOptions] = useState<{ label: string; value: string }[]>([]);
  const [ensureDefaultLoading, setEnsureDefaultLoading] = useState<boolean>(true);
  const [apiKeys, setApiKeys] = useState<Record<string, string>>({
    openai: "",
    google: "",
  });
  const [prompts, setPrompts] = useState<{ summarize: string; qa: string }>(
    DEFAULT_PROMPTS,
  );
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");
  const [success, setSuccess] = useState<boolean>(false);
  const navigate = useNavigate();

  useEffect(() => {
    const load = async () => {
      setError("");
      setEnsureDefaultLoading(true);
      try {
        await axios.get(`${API_URL}/auth/ollama/ensure-default`, {
          headers: getAuthHeaders(),
          timeout: 300_000,
        });
      } catch (err: any) {
        setError("Failed to prepare default model.");
      } finally {
        setEnsureDefaultLoading(false);
      }

      setLoading(true);
      try {
        const [modelsRes, settingsRes] = await Promise.all([
          axios.get<{ models: string[] }>(`${API_URL}/auth/ollama/models`, {
            headers: getAuthHeaders(),
          }),
          axios.get(`${API_URL}/auth/settings`, { headers: getAuthHeaders() }),
        ]);
        const data = settingsRes.data;
        setPdfParser(data.pdf_parser || "tesseract");
        setSelectedModel(data.model_name || "ollama/gemma3:1b");
        setApiKeys(data.api_keys || {});
        setPrompts({
          summarize: data.prompts?.summarize || DEFAULT_PROMPTS.summarize,
          qa: data.prompts?.qa || DEFAULT_PROMPTS.qa,
        });
        const ollamaOptions = (modelsRes.data.models || []).map((m) => ({
          value: m,
          label: m.replace(/^ollama\//, ""),
        }));
        const savedModel = data.model_name || "ollama/gemma3:1b";
        const allOptions = [OPENAI_MODEL, ...ollamaOptions];
        const hasSaved = allOptions.some((o) => o.value === savedModel);
        if (!hasSaved && savedModel.startsWith("ollama/")) {
          allOptions.push({
            value: savedModel,
            label: savedModel.replace(/^ollama\//, ""),
          });
        }
        setLlmModelOptions(allOptions);
      } catch (err: any) {
        setError("Failed to load settings.");
        setLlmModelOptions([OPENAI_MODEL]);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const handleApiKeyChange = (provider: string, value: string) => {
    setApiKeys((prev) => ({ ...prev, [provider]: value }));
  };

  const handlePromptChange = (field: "summarize" | "qa", value: string) => {
    setPrompts((prev) => ({ ...prev, [field]: value }));
  };

  const handleSave = async () => {
    setError("");
    setLoading(true);
    try {
      await axios.post(`${API_URL}/auth/settings`, {
        pdf_parser: pdfParser,
        model_name: selectedModel,
        api_keys: apiKeys,
        prompts,
      });
      setSuccess(true);
    } catch (err: any) {
      setError("Failed to save settings.");
    } finally {
      setLoading(false);
    }
  };

  const showApiKeyInput = selectedModel === OPENAI_MODEL.value;

  return (
    <Container maxWidth="sm">
      <Box sx={{ mt: 4, mb: 4 }}>
        <Tabs value={0} sx={{ mb: 2 }}>
          <Tab label="Home" onClick={() => navigate("/")} />
          <Tab label="Settings" />
        </Tabs>
        <Typography variant="h4" gutterBottom>
          Settings
        </Typography>
        {(ensureDefaultLoading || loading) && (
          <CircularProgress sx={{ mb: 2 }} />
        )}
        {ensureDefaultLoading && (
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Downloading default model (one time only)…
          </Typography>
        )}
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            LLM Model
          </Typography>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel id="llm-model-label">LLM Model</InputLabel>
            <Select
              labelId="llm-model-label"
              value={selectedModel}
              label="LLM Model"
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={loading || ensureDefaultLoading}
            >
              {llmModelOptions.map((opt) => (
                <MenuItem key={opt.value} value={opt.value}>
                  {opt.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          {showApiKeyInput && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle1">
                API Key for {selectedModel}
              </Typography>
              <TextField
                fullWidth
                type="password"
                label={`API Key for ${selectedModel}`}
                value={apiKeys[selectedModel] || ""}
                onChange={(e) =>
                  handleApiKeyChange(selectedModel, e.target.value)
                }
                sx={{ mt: 1 }}
                disabled={loading}
              />
            </Box>
          )}
        </Paper>
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Prompts
          </Typography>
          <TextField
            fullWidth
            label="Summarize Prompt"
            value={prompts.summarize}
            onChange={(e) => handlePromptChange("summarize", e.target.value)}
            sx={{ mb: 2 }}
            multiline
            minRows={2}
            disabled={loading}
          />
          <TextField
            fullWidth
            label="QA Prompt"
            value={prompts.qa}
            onChange={(e) => handlePromptChange("qa", e.target.value)}
            sx={{ mb: 2 }}
            multiline
            minRows={2}
            disabled={loading}
          />
        </Paper>
        <Button
          variant="contained"
          color="primary"
          onClick={handleSave}
          disabled={loading}
        >
          Save Settings
        </Button>
        <Snackbar
          open={success}
          autoHideDuration={3000}
          onClose={() => setSuccess(false)}
          message="Settings saved successfully"
        />
      </Box>
    </Container>
  );
};

export default Settings;
