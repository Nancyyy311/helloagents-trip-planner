import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src')
    }
  },
  server: {
    port: 5173,
    proxy: {
      // 用 127.0.0.1 而不是 localhost：新版 Node 会把 localhost 优先解析成 ::1(IPv6)，
      // 而后端 uvicorn 默认只监听 IPv4，代理会直接 502。
      '/api': {
        target: 'http://127.0.0.1:7000',
        changeOrigin: true
      },
      '/health': {
        target: 'http://127.0.0.1:7000',
        changeOrigin: true
      }
    }
  }
})
