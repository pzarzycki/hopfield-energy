import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ command }) => ({
  plugins: [react()],
  // Use root path for dev, GitHub Pages path for production
  base: command === 'serve' ? '/' : '/hopfield-energy/',
  build: {
    chunkSizeWarningLimit: 1000,
  },
}))
