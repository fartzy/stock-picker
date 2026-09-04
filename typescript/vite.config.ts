import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// kept in sync with python/stock_picker/api/main.py's DEFAULT_HOST/DEFAULT_PORT
const API_PROXY_TARGET = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": API_PROXY_TARGET,
    },
  },
});
