import React, { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import {
  Container,
  Box,
  Typography,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Paper,
  Alert,
  TextField,
  CircularProgress,
  Checkbox,
  FormGroup,
  FormControlLabel,
} from "@mui/material";
import {
  Delete as DeleteIcon,
  Description as DescriptionIcon,
  CheckCircle,
  HourglassEmpty,
  Error as ErrorIcon,
  RadioButtonUnchecked,
} from "@mui/icons-material";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import { useAuth } from "../contexts/AuthContext";
import ConversationHistory from './ConversationHistory';

interface Document {
  id: number;
  filename: string;
  file_type: string;
  created_at: string;
  summary?: string;
  faqs?: string;
  parsing_status: string;
}

interface DocumentStatus {
  document_id: number;
  parsing_status: string;
  error_message?: string;
  method?: string;
  model_name?: string;
}

interface Message {
  id: number;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: string;
}

interface Conversation {
  id: number;
  type: 'qa' | 'summarize';
  title: string;
  messages: Message[];
  created_at: string;
  updated_at: string;
  meta_data?: {
    document_ids?: number[];
  };
}

const API_URL = (window as any).APP_CONFIG?.API_URL || window.location.origin;

const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const { logout } = useAuth();
  const [documents, setDocuments] = useState<Document[]>([]);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [llmResult, setLlmResult] = useState<string>("");
  const [llmLoading, setLlmLoading] = useState(false);
  const [llmError, setLlmError] = useState("");
  const [llmResponseTimeSeconds, setLlmResponseTimeSeconds] = useState<number | null>(null);
  const [question, setQuestion] = useState("");
  const [statuses, setStatuses] = useState<DocumentStatus[]>([]);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const startParseRequestedRef = useRef<Set<number>>(new Set());
  const [lastUploadTime, setLastUploadTime] = useState<number | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedLlmModel, setSelectedLlmModel] = useState<string>("");

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      "application/pdf": [".pdf"],
      "image/jpeg": [".jpg", ".jpeg"],
      "image/png": [".png"],
    },
    onDrop: handleFileDrop,
  });

  const fetchDocuments = useCallback(async () => {
    try {
      const response = await axios.get(`${API_URL}/documents/`, {
        headers: getAuthHeaders(),
      });
      setDocuments(response.data);
      setError("");
    } catch (err) {
      setError("Failed to fetch documents");
    }
  }, []);

  const fetchStatuses = useCallback(async () => {
    try {
      const response = await axios.get(`${API_URL}/documents/status`, {
        headers: getAuthHeaders(),
      });
      setStatuses(response.data);
    } catch (err) {
      // ignore for now
    }
  }, []);

  const fetchConversations = useCallback(async () => {
    try {
      const response = await axios.get(`${API_URL}/conversations/`, {
        headers: getAuthHeaders(),
      });
      setConversations(response.data.conversations);
    } catch (err) {
      console.error('Failed to fetch conversations:', err);
    }
  }, []);

  const fetchSettings = useCallback(async () => {
    try {
      const response = await axios.get(`${API_URL}/auth/settings`, {
        headers: getAuthHeaders(),
      });
      setSelectedLlmModel(response.data.model_name || "ollama/gemma3:1b");
    } catch (err) {
      // use default if fetch fails
      setSelectedLlmModel("ollama/gemma3:1b");
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
    fetchStatuses();
    fetchConversations();
    fetchSettings();
  }, [fetchDocuments, fetchStatuses, fetchConversations, fetchSettings]);

  // Start parsing for any document that shows "not_started" (e.g. no ParsedDocument row yet)
  useEffect(() => {
    const notStarted = documents.filter((doc) => doc.parsing_status === "not_started");
    if (notStarted.length === 0) return;
    const headers = getAuthHeaders();
    notStarted.forEach((doc) => {
      if (startParseRequestedRef.current.has(doc.id)) return;
      startParseRequestedRef.current.add(doc.id);
      axios
        .post(`${API_URL}/documents/${doc.id}/start-parsing`, {}, { headers })
        .catch(() => {
          startParseRequestedRef.current.delete(doc.id);
        });
    });
  }, [documents]);

  useEffect(() => {
    const shouldPoll = documents.some(
      (doc) =>
        doc.parsing_status === "pending" ||
        doc.parsing_status === "in_progress" ||
        doc.parsing_status === "not_started",
    );
    let stopTimeout: NodeJS.Timeout | null = null;

    if (shouldPoll && !pollingRef.current) {
      pollingRef.current = setInterval(() => {
        fetchDocuments();
        fetchStatuses();
      }, 3000);
      // Set a timeout to stop polling after 2 minutes from last upload
      if (lastUploadTime) {
        stopTimeout = setTimeout(
          () => {
            if (pollingRef.current) {
              clearInterval(pollingRef.current);
              pollingRef.current = null;
            }
          },
          2 * 60 * 1000,
        ); // 2 minutes
      }
    }
    if (!shouldPoll && pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
      if (stopTimeout) {
        clearTimeout(stopTimeout);
      }
    };
  }, [documents, fetchDocuments, fetchStatuses, lastUploadTime]);

  function getAuthHeaders(): Record<string, string> {
    const token = localStorage.getItem("token");
    return token ? { Authorization: `Bearer ${token}` } : {};
  }

  async function handleFileDrop(acceptedFiles: File[]) {
    setUploading(true);
    setError("");
    try {
      for (const file of acceptedFiles) {
        const formData = new FormData();
        formData.append("file", file);
        await axios.post(`${API_URL}/documents/upload`, formData, {
          headers: {
            "Content-Type": "multipart/form-data",
            ...getAuthHeaders(),
          },
        });
      }
      setLastUploadTime(Date.now());
      await fetchDocuments();
    } catch (err) {
      setError("Failed to upload file");
    } finally {
      setUploading(false);
    }
  }

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  // Add helper functions for select all
  const isAllSelected = documents.length > 0 && selectedIds.length === documents.length;
  const isAllPdfsSelected = documents.filter((doc) => doc.file_type.startsWith("application/pdf")).every((doc) => selectedIds.includes(doc.id));
  const isAllImagesSelected = documents.filter((doc) => doc.file_type.startsWith("image/")).every((doc) => selectedIds.includes(doc.id));

  const handleSelectAll = (checked: boolean) => {
    setSelectedIds(checked ? documents.map((doc) => doc.id) : []);
  };
  const handleSelectAllPdfs = (checked: boolean) => {
    const pdfIds = documents.filter((doc) => doc.file_type.startsWith("application/pdf")).map((doc) => doc.id);
    setSelectedIds((prev) => checked ? Array.from(new Set([...prev, ...pdfIds])) : prev.filter((id) => !pdfIds.includes(id)));
  };
  const handleSelectAllImages = (checked: boolean) => {
    const imgIds = documents.filter((doc) => doc.file_type.startsWith("image/")).map((doc) => doc.id);
    setSelectedIds((prev) => checked ? Array.from(new Set([...prev, ...imgIds])) : prev.filter((id) => !imgIds.includes(id)));
  };
  const handleFileCheckbox = (id: number, checked: boolean) => {
    setSelectedIds((prev) => checked ? [...prev, id] : prev.filter((x) => x !== id));
  };

  const handleSummarize = async () => {
    setLlmResult("");
    setLlmError("");
    setLlmResponseTimeSeconds(null);
    setLlmLoading(true);
    try {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
        ...getAuthHeaders(),
      };

      const response = await fetch(`${API_URL}/documents/llm/summarize/stream`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          document_ids: selectedIds,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to summarize");
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("No response body");
      }

      let accumulatedText = "";
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              if (typeof data.text === "string") {
                accumulatedText += data.text;
                setLlmResult(accumulatedText);
              }
              if (typeof data.response_time_seconds === "number") {
                setLlmResponseTimeSeconds(data.response_time_seconds);
              }
            } catch (e) {
              console.error("Failed to parse SSE data:", e);
            }
          }
        }
      }
      // Refetch conversations after summarization
      await fetchConversations();
    } catch (err: any) {
      setLlmError("Failed to summarize.");
    } finally {
      setLlmLoading(false);
    }
  };

  const handleQA = async () => {
    setLlmResult("");
    setLlmError("");
    setLlmResponseTimeSeconds(null);
    setLlmLoading(true);
    try {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
        ...getAuthHeaders(),
      };

      const response = await fetch(`${API_URL}/documents/llm/qa/stream`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          document_ids: selectedIds,
          question,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get answer");
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("No response body");
      }

      let accumulatedText = "";
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              if (typeof data.text === "string") {
                accumulatedText += data.text;
                setLlmResult(accumulatedText);
              }
              if (typeof data.response_time_seconds === "number") {
                setLlmResponseTimeSeconds(data.response_time_seconds);
              }
            } catch (e) {
              console.error("Failed to parse SSE data:", e);
            }
          }
        }
      }
      // Refetch conversations after Q&A
      await fetchConversations();
    } catch (err: any) {
      setLlmError("Failed to get answer.");
    } finally {
      setLlmLoading(false);
    }
  };

  function getStatusInfo(status: string) {
    switch (status) {
      case "done":
        return {
          icon: <CheckCircle sx={{ color: "green", mr: 1 }} />,
          label: "Parsed",
        };
      case "pending":
      case "in_progress":
        return {
          icon: <HourglassEmpty sx={{ color: "goldenrod", mr: 1 }} />,
          label: "Parsing",
        };
      case "error":
        return {
          icon: <ErrorIcon sx={{ color: "red", mr: 1 }} />,
          label: "Error",
        };
      default:
        return {
          icon: <RadioButtonUnchecked sx={{ color: "gray", mr: 1 }} />,
          label: "Not started",
        };
    }
  }

  const handleDelete = async (docId: number) => {
    try {
      await axios.delete(`${API_URL}/documents/${docId}`, {
        headers: getAuthHeaders(),
      });
      await fetchDocuments();
      await fetchStatuses();
    } catch (err) {
      setError("Failed to delete document");
    }
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ mt: 4, mb: 4 }}>
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            mb: 4,
          }}
        >
          <Box>
            <Typography variant="h4" component="h1">
              DocChat
            </Typography>
            {selectedLlmModel && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                LLM: {selectedLlmModel}
              </Typography>
            )}
          </Box>
          <Box>
            <Button
              variant="outlined"
              color="primary"
              onClick={() => navigate("/settings")}
              sx={{ mr: 1 }}
            >
              Settings
            </Button>
            <Button variant="outlined" color="primary" onClick={handleLogout}>
              Logout
            </Button>
          </Box>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <Paper
          {...getRootProps()}
          sx={{
            p: 3,
            mb: 4,
            textAlign: "center",
            cursor: "pointer",
            backgroundColor: isDragActive ? "action.hover" : "background.paper",
            border: "2px dashed",
            borderColor: isDragActive ? "primary.main" : "divider",
          }}
        >
          <input {...getInputProps()} />
          <Typography variant="h6" gutterBottom>
            {isDragActive
              ? "Drop the files here"
              : "Drag and drop files here, or click to select files"}
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Supported formats: PDF, JPEG, PNG
          </Typography>
        </Paper>

        {uploading && (
          <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
            <CircularProgress size={24} sx={{ mr: 2 }} />
            <Typography variant="body1">
              Uploading and parsing files...
            </Typography>
          </Box>
        )}

        <Typography variant="h5" gutterBottom>
          Select Files
        </Typography>
        <FormGroup>
          <FormControlLabel
            control={<Checkbox checked={isAllSelected} onChange={(_, checked) => handleSelectAll(checked)} />}
            label="All Files"
          />
          <FormControlLabel
            control={<Checkbox checked={isAllPdfsSelected} onChange={(_, checked) => handleSelectAllPdfs(checked)} />}
            label="All PDFs"
          />
          <FormControlLabel
            control={<Checkbox checked={isAllImagesSelected} onChange={(_, checked) => handleSelectAllImages(checked)} />}
            label="All Images"
          />
        </FormGroup>
        <Typography variant="subtitle1" sx={{ mt: 2 }}>
          Individual Files
        </Typography>
        <FormGroup>
          {documents.map((doc) => (
            <FormControlLabel
              key={doc.id}
              control={<Checkbox checked={selectedIds.includes(doc.id)} onChange={(_, checked) => handleFileCheckbox(doc.id, checked)} />}
              label={doc.filename}
            />
          ))}
        </FormGroup>

        <Typography variant="h5" gutterBottom>
          Your Documents
        </Typography>

        <List>
          {documents.map((doc) => {
            const statusObj = statuses.find((s) => s.document_id === doc.id);
            // Prefer doc.parsing_status (from list, updated by polling); fallback to statuses
            const status = doc.parsing_status || statusObj?.parsing_status || "not_started";
            const { icon, label } = getStatusInfo(status);
            return (
              <ListItem
                key={doc.id}
                sx={{
                  mb: 1,
                  backgroundColor: "background.paper",
                  borderRadius: 1,
                }}
              >
                <DescriptionIcon sx={{ mr: 2 }} />
                <ListItemText
                  primary={doc.filename}
                  secondary={
                    <span>
                      {icon}
                      {label} &nbsp;| Uploaded on{" "}
                      {new Date(doc.created_at).toLocaleDateString()}
                    </span>
                  }
                />
                <ListItemSecondaryAction>
                  <IconButton
                    edge="end"
                    aria-label="delete"
                    onClick={() => handleDelete(doc.id)}
                  >
                    <DeleteIcon />
                  </IconButton>
                </ListItemSecondaryAction>
              </ListItem>
            );
          })}
        </List>

        <Box sx={{ display: "flex", gap: 2, mb: 2 }}>
          <Button
            variant="contained"
            color="primary"
            disabled={selectedIds.length === 0 || llmLoading}
            onClick={handleSummarize}
          >
            Summarize
          </Button>
          <TextField
            label="Ask a Question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            size="small"
            sx={{ flex: 1 }}
          />
          <Button
            variant="contained"
            color="secondary"
            disabled={selectedIds.length === 0 || !question || llmLoading}
            onClick={handleQA}
          >
            Ask
          </Button>
        </Box>
        {llmLoading && <CircularProgress sx={{ mb: 2 }} />}
        {llmError && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {llmError}
          </Alert>
        )}
        {llmResult && (
          <Paper sx={{ p: 2, mb: 2, backgroundColor: "#f5f5f5" }}>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>
              Result:
            </Typography>
            <Typography variant="body1" sx={{ whiteSpace: "pre-line" }}>
              {llmResult}
            </Typography>
            {llmResponseTimeSeconds != null && (
              <Typography variant="caption" display="block" sx={{ mt: 1, color: "text.secondary" }}>
                Response time: {llmResponseTimeSeconds} s
              </Typography>
            )}
          </Paper>
        )}

        <ConversationHistory conversations={conversations} documents={documents} />
      </Box>
    </Container>
  );
};

export default Dashboard;
