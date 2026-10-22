// Typed client for the FastAPI backend.

export interface Citation {
  id: number;
  source: string;
  section: string;
  score: number;
  text: string;
}

export interface Health {
  status: string;
  llm: string;
  embeddings: string;
  chunks: number;
}

export interface StreamHandlers {
  onStep: (node: string, trace: string[]) => void;
  onToken: (text: string) => void;
  onCitations: (c: Citation[]) => void;
  onDone: (info: { route: string; redacted: string[] }) => void;
}

const BASE = "/api";

export async function getHealth(): Promise<Health> {
  const r = await fetch(`${BASE}/health`);
  return r.json();
}

export async function uploadDoc(file: File): Promise<{ source: string; chunks: number }> {
  const form = new FormData();
  form.append("file", file);
  const r = await fetch(`${BASE}/documents`, { method: "POST", body: form });
  if (!r.ok) throw new Error((await r.json()).detail ?? "Upload failed");
  return r.json();
}

/**
 * POST + Server-Sent Events. The browser's EventSource only supports GET,
 * so we read the response body stream and parse "event:/data:" frames ourselves.
 */
export async function streamChat(question: string, h: StreamHandlers, signal?: AbortSignal) {
  const r = await fetch(`${BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!r.ok || !r.body) throw new Error(`HTTP ${r.status}`);

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop() ?? ""; // keep the incomplete frame for the next chunk
    for (const frame of frames) {
      let event = "message";
      let data = "";
      for (const line of frame.split(/\r?\n/)) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      const payload = JSON.parse(data);
      if (event === "step") h.onStep(payload.node, payload.trace);
      else if (event === "token") h.onToken(payload.text);
      else if (event === "citations") h.onCitations(payload);
      else if (event === "done") h.onDone(payload);
    }
  }
}
