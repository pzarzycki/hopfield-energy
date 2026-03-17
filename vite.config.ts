import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ command }) => {
  const isDevContainer = process.env.PROJECT_RUNTIME === 'devcontainer'

  return {
    plugins: [react()],
    // Use root path for dev, GitHub Pages path for production
    base: command === 'serve' ? '/' : '/hopfield-energy/',
    server: isDevContainer
      ? {
          watch: {
            usePolling: true,
            interval: 250,
          },
        }
      : undefined,
    worker: {
      format: 'es',
    },
    build: {
      chunkSizeWarningLimit: 1000,
    },
  }
})
