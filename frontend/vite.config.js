import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" keeps the built site working from any folder or sub-path
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: { port: 5173 },
});
