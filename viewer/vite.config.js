import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

import { cloudflare } from "@cloudflare/vite-plugin";

// Where the built assets live in production. Set VITE_BASE_PATH at build time
// for a subpath deploy (e.g. GitHub Project Pages at /<repo>/). The Cloudflare
// wrangler deploy serves at root, so its build leaves it empty.
// Dev server always stays at /.
export default defineConfig(({ command }) => ({
  plugins: [react(), cloudflare()],
  base: command === 'build' ? (process.env.VITE_BASE_PATH || '/') : '/',
}))
