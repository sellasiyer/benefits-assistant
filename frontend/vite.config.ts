import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev proxy: the UI calls /api/* and Vite forwards to FastAPI, so no CORS in dev.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://localhost:8000", rewrite: (p) => p.replace(/^\/api/, "") } },
  },
});
