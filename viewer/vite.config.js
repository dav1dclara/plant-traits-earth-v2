import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// GitHub Project Pages serve at /<repo>/, so production assets need that base.
// Dev server stays at / .
export default defineConfig(({ command }) => ({
  plugins: [react()],
  base: command === 'build' ? '/plant-traits-earth-v2/' : '/',
}))
