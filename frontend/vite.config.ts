import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
// Si lo usas, déjalo. Si no, puedes quitarlo sin problema.
// import { componentTagger } from "lovable-tagger";

export default defineConfig(({ mode }) => ({
  server: {
    host: "127.0.0.1",   // evita líos con IPv6
    port: 8001,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        secure: false,
      },
    },
  },
  plugins: [
    react(),
    // mode === "development" && componentTagger(),
  ].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
