const API_BASE = (import.meta.env.VITE_API_URL as string) || "http://localhost:8000";
const API_URL = `${API_BASE}/api/v1/rag`;
const API_KEY = import.meta.env.VITE_API_KEY as string | undefined;

// Base headers for JSON requests, including the API key when configured.
const jsonHeaders = (): Record<string, string> => {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (API_KEY) headers["X-API-Key"] = API_KEY;
  return headers;
};

export interface RepoInfo {
  repo_id: string;
  display_name: string;
  source: string;
  status: string;
  files_indexed: number;
  chunks_created: number;
  progress?: string | null;
  error?: string | null;
  created_at: string;
  updated_at: string;
}

export const checkHealth = async () => {
  try {
    const response = await fetch(`${API_BASE}/health`);
    return response.ok;
  } catch (e) {
    return false;
  }
};

export const indexRepository = async (repoInput: string) => {
  const response = await fetch(`${API_URL}/index`, {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify({ repo_input: repoInput }),
  });
  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail || "Indexing failed");
  }
  return response.json();
};

const authHeaders = (): Record<string, string> => (API_KEY ? { "X-API-Key": API_KEY } : {});

export const listRepos = async (): Promise<RepoInfo[]> => {
  const response = await fetch(`${API_URL}/repos`, { headers: authHeaders() });
  if (!response.ok) return [];
  return response.json();
};

export const deleteRepo = async (repoId: string) => {
  const response = await fetch(`${API_URL}/repos/${repoId}`, { method: "DELETE", headers: authHeaders() });
  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail || "Delete failed");
  }
  return response.json();
};

export const queryCodebase = async (repoId: string, query: string, history: any[]) => {
  const response = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify({
      repo_id: repoId,
      query,
      conversation_history: history,
    }),
  });
  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail || "Query failed");
  }
  return response.json();
};

export interface StreamCallbacks {
  onSources?: (sources: any[]) => void;
  onToken?: (text: string) => void;
}

// Streams the answer via Server-Sent Events. Resolves when the stream completes;
// rejects on a non-2xx status or an error frame.
export const queryCodebaseStream = async (
  repoId: string,
  query: string,
  history: any[],
  callbacks: StreamCallbacks,
) => {
  const response = await fetch(`${API_URL}/query/stream`, {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify({ repo_id: repoId, query, conversation_history: history }),
  });
  if (!response.ok || !response.body) {
    let detail = "Query failed";
    try { detail = (await response.json()).detail || detail; } catch { /* non-JSON */ }
    throw new Error(detail);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const handleFrame = (frame: string) => {
    const line = frame.split("\n").find(l => l.startsWith("data:"));
    if (!line) return;
    const payload = JSON.parse(line.slice(5).trim());
    if (payload.type === "sources") callbacks.onSources?.(payload.sources || []);
    else if (payload.type === "token") callbacks.onToken?.(payload.text || "");
    else if (payload.type === "error") throw new Error(payload.message || "Generation failed");
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      if (frame.trim()) handleFrame(frame);
    }
  }
  if (buffer.trim()) handleFrame(buffer);
};
