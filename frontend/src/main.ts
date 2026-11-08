import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import Antd from 'ant-design-vue'
import 'ant-design-vue/dist/reset.css'
import App from './App.vue'

// 改造点：路由级懒加载。
// 原来 Home 和 Result 都是顶层 import，等于打开首页就把结果页的
// 高德 JSAPI 加载器、html2canvas、jsPDF 全部下载一遍——首屏白等这些体积。
// 改成动态 import 后，只有真的跳到 /result 才会去取那一块。
const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'Home',
      component: () => import('./views/Home.vue')
    },
    {
      path: '/result',
      name: 'Result',
      component: () => import('./views/Result.vue')
    }
  ]
})

const app = createApp(App)

app.use(router)
app.use(Antd)

app.mount('#app')
