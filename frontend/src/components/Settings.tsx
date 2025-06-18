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

const PDF_PARSERS = [
  { label: "Tesseract OCR", value: "tesseract" },
  { label: "Docling", value: "docling" },
];

const LLM_PROVIDERS = [
  { label: "Default - No API Key Required (gpt-4o-mini)", value: "openai/gpt-4o-mini" },
  { label: "GPT-4.1-mini (Add your own key)", value: "openai/gpt-4.1-mini" },
  { label: "Gemini-2.0-flash-lite (Add your own key)", value: "google/gemini-2.0-flash-lite" },
];

const DEFAULT_PROMPTS = {
  summarize: "Summarize the following text for efficacy and clarity.",
  qa: "Given the following text, answer the question as accurately as possible.",
};

const API_URL = (window as any).APP_CONFIG?.API_URL || 'http://localhost:8001';

const Settings: React.FC = () => {
  const [pdfParser, setPdfParser] = useState<string>("tesseract");
  const [selectedModel, setSelectedModel] = useState<string>("openai/gpt-4o-mini");
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
    const fetchSettings = async () => {
      setLoading(true);
      setError("");
      try {
        const response = await axios.get(`${API_URL}/auth/settings`);
        const data = response.data;
        setPdfParser(data.pdf_parser || "tesseract");
        setSelectedModel(data.model_name || "openai/gpt-4o-mini");
        setApiKeys(data.api_keys || {});
        setPrompts({
          summarize: data.prompts?.summarize || DEFAULT_PROMPTS.summarize,
          qa: data.prompts?.qa || DEFAULT_PROMPTS.qa,
        });
      } catch (err: any) {
        setError("Failed to load settings.");
      } finally {
        setLoading(false);
      }
    };
    fetchSettings();
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

  const showApiKeyInput = selectedModel !== "openai/gpt-4o-mini";

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
        {loading && <CircularProgress sx={{ mb: 2 }} />}
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            PDF Parsing Library
          </Typography>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel id="pdf-parser-label">PDF Parser</InputLabel>
            <Select
              labelId="pdf-parser-label"
              value={pdfParser}
              label="PDF Parser"
              onChange={(e) => setPdfParser(e.target.value)}
              disabled={loading}
            >
              {PDF_PARSERS.map((opt) => (
                <MenuItem key={opt.value} value={opt.value}>
                  {opt.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Paper>
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            LLM Model Provider
          </Typography>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel id="llm-model-label">LLM Model</InputLabel>
            <Select
              labelId="llm-model-label"
              value={selectedModel}
              label="LLM Model"
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={loading}
            >
              {LLM_PROVIDERS.map((opt) => (
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
