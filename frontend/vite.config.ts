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
  },
  build: {
    // 单个 chunk 超过这个体积才警告。设成 900 是因为 ant-design-vue
    // 全量注册后本身就有 1MB 左右，500K 的默认阈值会一直报"无用"的警告。
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        // 改造点：把"几乎不变"的第三方库和"每次发版都变"的业务代码分开。
        // 注意：这不会减少首屏总字节数（该下的还得下），收益在于
        //   1) 发版后用户只需重新下载几十 KB 的业务代码，antd 走缓存；
        //   2) 三个 chunk 可以并行下载，而不是串着等一个大文件。
        // 真正减少字节的是路由懒加载和 jspdf/html2canvas 的按需加载。
        manualChunks(id: string) {
          if (!id.includes('node_modules')) return
          if (id.includes('jspdf') || id.includes('html2canvas')) return 'export-pdf'
          if (id.includes('@amap')) return 'amap'
          if (id.includes('ant-design-vue') || id.includes('@ant-design')) return 'antd'
          if (
            id.includes('vue-router') ||
            id.includes('@vue/') ||
            /node_modules[\\/]vue[\\/]/.test(id)
          ) {
            return 'vue-vendor'
          }
          return
        }
      }
    }
  }
})
