import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The UI calls /api/* and Vite forwards to a backend. All three backends expose the same contract,
// so pick one with BACKEND_URL (default: the Python service):
//   BACKEND_URL=http://localhost:8000 npm run dev   # Python FastAPI + LangGraph
//   BACKEND_URL=http://localhost:8080 npm run dev   # Spring Boot gateway -> Python
//   BACKEND_URL=http://localhost:8081 npm run dev   # Java Spring AI
const target = process.env.BACKEND_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target, rewrite: (p) => p.replace(/^\/api/, "") } },
  },
});
