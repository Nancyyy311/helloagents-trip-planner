# 智能旅行规划 Agent —— 二次开发改造说明

> 本仓库基于开源项目 **[helloagents-trip-planner](https://github.com/nameless0120/helloagents-trip-planner)**（作者 nameless0120，CC BY-NC-SA 4.0）做二次开发。
> 原项目功能完整、能跑通，但更接近"能演示的 demo"；本次改造的目标是把它变成**能上线的服务**：
> 修正确性 bug、消除名不副实的实现、做并发与性能优化、治理模型幻觉、加可观测性与测试。
> 所有改造点都有**实测数据**支撑，不做无依据的"优化"；已知短板在第 9 节如实列出。

---

## 1. 这个项目做什么

用户输入城市 / 日期 / 天数 / 预算 / 偏好 → 服务调用**高德 Web 服务 API** 拿景点、天气、酒店、餐饮候选 → 交给**大模型**生成结构化行程 JSON → 用 **19 维指标 rerank** 选出最优候选 → **做 POI 真实性校验** → 返回可直接渲染的行程方案。

用户还能基于已生成的计划**改一句话**（"第二天改成室内"），服务做最小改动，只重新生成涉及的那一天。

技术栈：Python · FastAPI · Pydantic v2 · DeepSeek（OpenAI 兼容协议）· 高德 API · Vue3 前端

---

## 2. 改造前发现的 9 个真实问题

这些问题不是猜的，是读源码 + 打请求 + 计时 + 看日志验证出来的。

### 2.1 工程质量类

| # | 问题 | 现象 / 证据 | 影响 |
|---|---|---|---|
| 1 | 健康检查读错属性 | `app/api/routes/trip.py` 读 `agent.agent.name`，而 `MultiAgentTripPlanner` 并没有 `.agent` 属性 → 每次都抛 `AttributeError` 被 except 吞掉 | `GET /api/trip/health` **恒返回 503**，探活完全失效 |
| 2 | 假成功（静默降级） | 模型调用失败走模板兜底时，接口仍返回 `HTTP 200 + success=true` | 故障被伪装成成功，线上无法发现模型挂了 |
| 3 | 零可观测 | 没有 `/metrics`，响应里没有 traceId | 出问题定位不到"是哪一次请求、慢在哪一步" |
| 4 | 单请求 ~122 秒 | 3 个候选计划**串行**生成，每个约 40 秒 | 用户干等两分钟，网关默认 60s 就超时 |
| 5 | 并发串台 + 事件循环阻塞 | `async` 路由里直接调同步 `plan_trip()`（几十秒）；全局单例共享 `_history` / 状态 | 一个请求卡住所有请求；并发时状态互相覆盖 |
| 6 | 重复调用高德 + 解析不稳定 | 每次请求都实时打 10~20 次高德；JSON 解析只要前后多一句话就失败，失败即重跑一次 40 秒调用 | 慢、费额度、触发限流 |

### 2.2 名不副实类（读源码才发现）

| # | 问题 | 证据 |
|---|---|---|
| 7 | **"多智能体 + 工具调用"名不副实** | `SimpleAgent` 创建时从未传 `tool_registry`，框架里 `enable_tool_calling = enable_tool_calling and tool_registry is not None` 恒为 `False` → `run()` 直接走单次 `llm.invoke()`，**一次工具都没调过**。提示词里还明确写着"不要输出工具调用"，与"挂工具"直接冲突 |
| 8 | **两个高德方法恒返回空列表** | `services/amap_service.py` 的 `search_poi()` / `get_weather()` 写着 `# TODO: 解析实际数据` 然后 `return []`。而 `/api/map/poi`、`/api/map/weather`、`/api/poi/search` 六个端点全依赖它 → **地图功能一直是空的，接口却一直返回成功** |
| 9 | **Grounding 算完就扔** | 三套互不相通：`warn_plan_grounding()` 只 `print` 警告、`enrich_trip_plan_poi_details()` 只回填**餐饮**坐标、`rerank` 的 `*_grounding_rate` 只用于打分。**API 响应里没有任何真实性信息** |

---

## 3. 改造清单（7 个技术维度）

### 维度 A · 可靠性

| 改造 | 做法 | 效果 |
|---|---|---|
| 修健康检查 | 改为 `getattr(agent, "planner_agent", None)` 安全取值，并返回真正有意义的自检信息 | **503 → 200** |
| 消除假成功 | 只有 `llm_success` 才返回 200；走 fallback 时返回 **502 + success=false**，同时把降级计划放进 `data` | 错误语义正确，前端可区分"降级展示"与"重试" |
| LLM 限流退避 | 只对可重试错误（429 / 5xx / 连接重置 / overloaded）做指数退避（2s、4s + 抖动），最多 2 次；内容风控、400 参数错误**不重试**；**超时不重试** | 候选成功率提升，且不对无解错误做无意义等待 |

### 维度 B · 性能与并发

| 改造 | 做法 | 效果 |
|---|---|---|
| 候选并行生成 | 3 个候选用 `ThreadPoolExecutor` 并发跑；**每个候选独立 `SimpleAgent`**（各自 `_history`），共享单例 LLM 客户端；并行整体失败自动回退串行 | **122s → 45~61s** |
| 不阻塞事件循环 | `await asyncio.to_thread(...)` 把同步生成丢进线程池 | 生成期间 `/health`、`/metrics` 仍秒回 |
| 请求状态隔离 | 每个请求/任务新建独立 planner 实例，从实例自身读状态 | 并发不再串台；天然无状态，为多实例铺路 |
| 长任务异步化 | `POST /plan/async` 立即返回 `task_id`（毫秒级），后台生成，任务表 + 锁 + 1 小时 TTL 清理 | 前端即时响应，不再干等两分钟 |

### 维度 C · 缓存

| 改造 | 做法 | 效果 |
|---|---|---|
| PlannerContext TTL 缓存 | 进程级 `_TTLCache`（TTL 30 分钟），按 16 个业务字段拼 key，**深拷贝返回**防污染 | 重复请求取数 **10.36s → 0.00s** |
| LRU 容量上限 | 超限时先清过期项，再淘汰最久未使用 | 内存有上界，不会长期膨胀 |
| 缓存边界 | **只缓存取数，不缓存行程**——用户可能想换方案，行程必须每次重新生成 | 有意的边界设计 |

### 维度 D · 可观测性与工程质量

| 改造 | 做法 | 效果 |
|---|---|---|
| traceId | Starlette `BaseHTTPMiddleware` 为每个请求生成/透传 `X-Request-ID`，写响应头 + 日志 | 可定位单次请求 |
| `/metrics` | 请求数 / 状态码分布 / 各接口次数 / 平均·最大耗时 / **LLM 成功率** / **工具调用数** / **缓存命中率** / **JSON 解析成功率** / **候选有效率** / **Grounding 命中率** | 从"看不见"到"能预警" |
| SSE 流式推送 | `GET /plan/async/{id}/stream`，状态一变就推，15s 心跳，600s 超时，`X-Accel-Buffering: no` | 一次连接拿到底，省掉轮询空转 |
| JSON 解析鲁棒性 | 支持前后说明文字、代码块、尾逗号、中文引号、BOM、**截断补全**；仍失败才抛错 | 直接消掉一大批"解析失败 → 重跑 40s" |
| 弃用 API 清理 | `@app.on_event` → `lifespan`；`Field(example=)` → `json_schema_extra`；`class Config` → `model_config` | 弃用警告 **27 → 1**（剩下 1 个在第三方库） |
| 回归测试 | `pytest` **92 个用例**，用 `monkeypatch` 替换真实 LLM / 高德，**不烧额度**、秒级跑完 | 改动有保护 |

### 维度 E · 真 Agent（把"伪工具调用"补成真调用）

| 改造 | 做法 | 效果 |
|---|---|---|
| 挂载真实高德工具 | 新增 `app/agents/amap_tools.py`：`AmapPOISearchTool` / `AmapWeatherTool`，继承框架 `Tool` 基类；**按请求构建注册表**（避免并发互相吃额度），单次预算 6 次、最多 2 轮 | 模型**自主**调用工具，实测 `tool_calls=2` |
| 提示词解冲突 | 原提示词禁止"工具调用"，与挂工具直接冲突。拆出 `PLANNER_AGENT_PROMPT_WITH_TOOLS`：保留全部 JSON 硬性要求，只放开工具 + 加 6 条使用规则（什么时候该调、一次最多一个、查到就收口） | 工具可用且不滥用 |
| 工具轮次调参 | A/B 对比：2 轮 → 3/3 候选、64.66s；1 轮 → **1/3 候选、101.80s**（模型没轮次收口，输出不完整→解析失败→重试） | 默认 2 轮，是**测出来**的不是拍的 |
| 混合模式 | **保留预抓取**（保证首轮上下文充足）+ **开放按需补查** | 兼顾延迟与灵活性，不是一刀切纯工具调用 |

### 维度 F · 幻觉治理（Grounding）

| 改造 | 做法 | 效果 |
|---|---|---|
| 校验结果落地 | 新增 `app/planner/grounding.py`，把 `verified` / `verification` 写回每个 POI，把 `GroundingSummary` 写回 `TripPlan` | 前端能显示"未核实"，用户才知道哪条不可信 |
| 全量坐标校验 | 景点 / 餐饮 / 酒店**都**校验；命中候选但坐标偏差 > **800 米**时用真实坐标覆盖 | 实测最终版本纠正了 **3 个编造坐标** |
| 不删只标 | 匹配不到只标 `verified=false`，**不删除** | 高德召回有缺口，硬删会误杀真实的老店小馆 |
| 区分"没数据"与"编造" | 候选池为空时标 `skipped`，不标 `unverified` | 避免把"工具没抓到"冤枉成"模型编造" |
| 口径统一 | `warn_plan_grounding()` 直接复用已算结果，不再重复遍历三遍 | 消除三套逻辑的不一致 |
| 指标化 | `/metrics` 新增 `grounding_rate` / `grounding_unverified` / `grounding_coordinates_corrected` | 编造率可观测、可预警 |

### 维度 G · 多轮会话（增量修改）

| 改造 | 做法 | 效果 |
|---|---|---|
| 增量修改接口 | `POST /plan/{plan_id}/refine`，传一句自然语言；单次生成、温度 0.1、**不做 3 候选 rerank** | **45s → 14s（约 3 倍）** |
| 最小改动约束 | 专门的 `PLANNER_REFINE_PROMPT`：未提及的天数/景点/餐饮/酒店必须原样保留；新增地点只能从候选池选 | 实测两轮都"变更 1 天 / 未变 2 天" |
| 变更摘要 | `diff_plans()` 按 `day_index` 逐天比对，返回 `changed_days` / `unchanged_days` / `budget_delta` | "只改了一天"有硬证据，不是自说自话 |
| 计划存储 | 内存 `dict` + TTL 1 小时 + 上限 200 条；同时存**原始请求**（否则上下文缓存无法命中） | 上下文取数 **0.00s** |
| 链式修改 | 每轮返回新 `plan_id`，支持改完第二天再改第三天 | 实测两轮连续修改均正确 |
| 失败语义 | 修改失败返回 **502 + "原计划未受影响"**，不返回模板计划 | 用户已有上一版，模板计划会覆盖掉满意的内容 |

---

## 4. 实测数据

单进程 uvicorn，`PLANNER_RERANK_CANDIDATE_COUNT=3`，DeepSeek + 高德真实调用。

### 4.1 性能

| 指标 | 改造前 | 改造后 |
|---|---|---|
| 单次规划耗时（冷缓存） | **~122 s** | **45.07 s / 46.39 s / 55.75 s**（多次实测） |
| 单次规划耗时（命中缓存） | ~122 s | **37.54 s** |
| 步骤1 高德取数 | ~10.4 s | **0.00 s**（命中缓存） |
| 候选执行方式 | 串行 3×~40s | **并行**（40.98 / 41.52 / 50.43 s，取最慢） |
| **增量修改（改一句话）** | 只能整趟重来 ~45 s | **14.66 s / 13.36 s** |

> 耗时本身会随模型负载波动，所以简历上写"约 2~3 倍提速"而不是一个固定值。

### 4.2 正确性与真实性

| 指标 | 改造前 | 改造后 |
|---|---|---|
| `GET /api/trip/health` | **503 恒定** | **200** |
| `/metrics` | 404 | 200（含 10 组指标） |
| `/api/map/poi` | **0 条**（恒返回 `[]`） | **10 条真实 POI** |
| `/api/map/weather` | **0 条** | **4 天真实预报** |
| 工具调用 | **0 次**（从未真正调用） | `tool_calls=2`，模型自主调用 |
| POI 真实性标注 | 无（只 print 到日志） | 每个 POI 带 `verified` / `verification` |
| 编造坐标 | 原样返回给前端 | 命中候选后纠正 **3 个** |
| 连续相同的两次请求 | 两次都打 10~20 次高德 | 第二次 **0 次** |

### 4.3 Grounding 实测（最有价值的一组数据）

青岛 3 天，3 个候选各自的校验结果：

| 候选 | 景点 | 餐饮 | 酒店 | 坐标被纠正 |
|---|---|---|---|---|
| 候选1 | 8/8 | 6/6 | 2/2 | 1 |
| 候选2 | 9/9 | 9/9 | 2/2 | 3 |
| 候选3 | 8/8 | 9/9 | 2/2 | 4 |
| **最终选中** | **9/9** | **9/9** | **2/2** | **3** |

**结论要读对**：POI **名字命中率 100%（36/36）**，但**坐标错了 8 个**（3 个候选合计），最终给用户的版本里仍有 3 个坐标是错的。

也就是说：**只做"名字 grounding"会得到 100% 的安全感，而地图上已经画错了 3 个点。** 名字层校验必要，但远远不够。

> 诚实记录：用 `free_text_input` 诱导模型"补充候选外的小众地点"（扬州 3 天），名字仍然 **0 编造**（7/7、9/9、2/2）。所以"编造检测"逻辑由 12 个单元测试覆盖，但**没有真实业务命中样本**，不宣称"抓到了大量编造"。

### 4.4 增量修改实测

青岛 3 天，连续两轮：

| 环节 | 耗时 | 关键细节 |
|---|---|---|
| 初始生成 | **45.07 s** | 3 候选并行 |
| 第 1 轮修改 | **14.66 s** | 上下文 **0.00 s**（缓存命中）、LLM 14.65 s |
| 第 2 轮修改 | **13.36 s** | 上下文 0.00 s、LLM 13.35 s |

修改效果（第 1 轮，"第二天下午改室内"）：

```
变更 1 天 / 未变 2 天，预算 1508 → 1428
D2: [八大关, 奥帆中心, 情人坝] → [八大关, 海军博物馆, 奥帆中心]
建议首句：本次已调整：第2天下午把情人坝换成室内的中国人民解放军海军博物馆（免费），
         奥帆中心移到傍晚，其余两天行程、餐饮与酒店均未改动。
```

### 4.5 `/metrics` 快照（节选）

```json
{
  "total_requests": 9,
  "llm_calls": 6,
  "llm_failures": 0,
  "llm_success_rate": 1.0,
  "tool_calls": 2,
  "tool_failures": 0,
  "tool_calls_by_name": { "amap_search_poi": 2 },
  "candidate_target": 3,
  "candidate_valid": 3,
  "candidate_valid_rate": 1.0,
  "grounding_checked": 20,
  "grounding_verified": 20,
  "grounding_rate": 1.0,
  "grounding_coordinates_corrected": 3,
  "planner_context_cache": {
    "size": 1, "maxsize": 128, "ttl_seconds": 1800,
    "hits": 1, "misses": 1, "hit_rate": 0.5, "evictions": 0
  },
  "planner_json_parse": { "attempts": 3, "success": 2, "repaired": 1, "failed": 0 }
}
```

> `llm_calls=6` = 2 次请求 × 3 个候选，证明并行确实是 3 路；
> `grounding_checked=20` 但 `coordinates_corrected=3`，说明"名字对、坐标错"是最隐蔽的幻觉形态；
> `repaired=1` 说明有一次候选是**靠解析修复救回来的**，否则要重跑 40 秒。

### 4.6 推理模式与输出截断（根因排查）

**现象**：候选有效数长期只有 2/3，失败日志里 8 成是"响应中未找到完整的顶层TripPlan JSON对象"。

**排查顺序**（先看数据，再改代码）：

1. `backend/scripts/analyze_failures.py` 统计历史 38 条失败：25 条原文为空（早期并行路径丢原文的回归），13 条有原文的里 **12 条结尾停在半句话或半个数字上**（如 `"longitude": 116`），只有 1 条是真的结构错误；
2. 对照 `max_tokens`：同样 3 天行程、同样 8400 上限，截断位置从 480 字符到 6156 字符不等 —— 若真是额度用满，长度应该稳定，说明有别的东西在抢预算；
3. 一次最小真实调用把原始 response 打出来：`completion_tokens=16` 而内容只有两个字，`reasoning_tokens=14`，字段里有 `reasoning_content`；
4. 查 DeepSeek 官方文档确认三条：思考模式**默认开启**、推理 token **计入** `completion_tokens` 且**占用** `max_tokens`、**思考模式下 `temperature` 无效**。

**结论**：`deepseek-v4-flash` 默认开思考，3 天行程单候选平均要 **9739 token**，而预算是 **8400** —— 每个候选都注定被切断。

**A/B 实测**（同一请求，北京 3 天，3 候选）：

| 指标 | 开启思考（默认） | 关闭思考（现配置） |
|---|---|---|
| 端到端耗时 | 81.72 s | **14.95 s** |
| completion tokens | 29 216 | 7 715 |
| 其中推理 tokens | 21 699（**74.3%**） | **0** |
| `candidate_valid` | 3/3 | **3/3** |
| `grounding_rate` | 1.0 | 1.0 |
| 景点 / 餐饮 | 8 / 9 | 8 / 9 |

结构完全一致，差别只在文案长度（332 字 vs 124 字）——**思考买到的是更啰嗦的描述，不是更好的安排，代价是 5.5 倍时间**。

**代码上的处理**：

- `app/agents/llm_tracing.py`：`TracedLLM` 子类补记 `finish_reason` / `usage` / `reasoning_tokens`。不改 `venv` 里的第三方库（升级即丢），元信息存 `threading.local`（3 路候选并行共用一个客户端）。
- `PLANNER_THINKING`（默认 0）：关闭思考。若开启，`planner_max_output_tokens()` 会自动追加 8000 推理预留，避免一开就撞上同一堵墙。
- `app/planner/failure_kind.py`：把失败分成 `truncated_in_string` / `truncated_in_number` / `truncated_structure` / `trailing_content` / `schema_invalid` 等 8 类，进 `/metrics` 的 `planner_failures_by_kind`，让下次优化有方向而不是靠猜。
- 注意：关思考同时让 `temperature` 恢复生效。本项目靠 `0.2 / 0.28 / 0.36` 的温度阶梯制造候选差异，而思考模式下这个参数是被 API 忽略的——也就是说候选多样性策略在改造前一直没真正起作用。

---

### 4.7 成本与并发治理（批次8）

原项目对最贵的接口没有任何约束：一次 `POST /api/trip/plan` 会触发
**3 个候选 LLM 调用 + 十余次高德查询 + 3 个线程**，而接口可以无限调用。

| 场景 | 结果 |
|---|---|
| 普通档限流（burst=2） | `#1 200 / #2 200 / #3 429`，响应带 `Retry-After: 30`、`X-RateLimit-Tier` |
| 探活豁免 | `/api/trip/health` 连打 5 次全 200 |
| 两个相同请求同时发 | 都 200 / **11.48s**，`llm_calls` 只 **+3**（不合并应为 +6），一个 `coalesced=true` |
| 并发闸门（上限 1） | 一个 `200 / 10.27s`，另一个 **`503 / 0.0s`**，`planning_rejected=1` |
| 同一计划并发 refine | 第二个请求 **409**，不再互相覆盖 |
| 前端首屏 JS | 2.2MB（gzip 683KB）→ **1.55MB（gzip 497KB）** |
| 导出库 | jspdf 388KB + html2canvas 202KB 改为**点击导出才加载** |

三个实现取舍：

1. **令牌桶而不是固定窗口**：固定窗口在边界会双倍放行（59 秒 6 个 + 61 秒 6 个）；
2. **规划档扣 2 个令牌**：把"这个接口值两倍"写进成本模型，而不是再加一个互不相关的上限；
3. **限流和并发闸门是两件事**：限流管"一分钟能进几个"，闸门管"同一时刻在做几个"。
   6 个请求 1 秒内进来，限流会放行，但那是 18 个候选线程。

> 探活豁免这里踩过一个坑：第一版只豁免 `/health`，实测第 3 次 `/api/trip/health` 就返回 429
> —— 上线后负载均衡会直接把节点摘掉。改成按 `/health` 后缀匹配，覆盖项目里三个探活入口。

### 4.8 客户端断开之后，后端在干什么？（批次9）

这是本项目最"反直觉"的一段排查，结论是：**在 FastAPI + `BaseHTTPMiddleware` 这套栈上，
服务端根本探测不到客户端已经走了。** 所有"标准做法"都试过，全部无效。

三条路，逐一实测：

| 做法 | 结果 |
|---|---|
| `await request.is_disconnected()`（请求体未读时） | ✅ 能检测到（第 6 次探测，约 1.2s） |
| `await request.is_disconnected()`（**请求体已读**） | ❌ **恒返回 False** |
| `request.is_disconnected()` 放在独立 `create_task` + `to_thread` 里 | ❌ 恒返回 False |
| `StreamingResponse` 每 2 秒写 keepalive，靠写失败判断 | ❌ 6 秒写了 6 个空格，全被内核 socket buffer 吃掉，无异常 |

第一版的实测现场（客户端 2 秒断开，服务端毫无察觉）：

```
POST /api/trip/plan -> 200 18628ms      ← 客户端早走了，请求照样跑满 18.6 秒
```

所以代码里**没有**保留这段探测——留着会造成"这里在防御"的错觉，比没有更糟。

**改用任务模型，让客户端自己说"我不要了"：**

```
POST   /api/trip/plan/async            → 立刻拿 task_id
GET    /api/trip/plan/async/{task_id}  → 每 1.2s 轮询（同时也是心跳，刷新 last_seen）
DELETE /api/trip/plan/async/{task_id}  → pagehide 时用 fetch+keepalive 发出去
```

| 用户行为 | 谁兜住 | 停止延迟 |
|---|---|---|
| 关页面 / 刷新 / 点取消 | `DELETE` → `CancelToken` | **0.3–0.4s** |
| 切到别的 tab | 轮询被节流到 ~60s，仍在 90s 宽限内 | 不中断（正确行为） |
| 浏览器崩溃 / 断网 | 后端 90s 无轮询 → 孤儿回收 | 90s |

取消延迟还修正过一次：第一版用 `with ThreadPoolExecutor(...)`，退出时 `shutdown(wait=True)`
会把挂在 LLM 上的候选全部等完：

| | 取消生效耗时 | 期间并发位 |
|---|---|---|
| `with ThreadPoolExecutor` | **17.0s** | 一直被占（想重来的用户撞 503） |
| 手动 `wait(timeout=0.4)` + `shutdown(wait=False)` | **0.3–0.4s** | 立刻释放 |

**诚实的边界**：已经在飞的那次 `client.chat.completions.create` 拦不住——Python 没有安全的
线程 kill，它一定会跑完并计费。取消能保证的是"下一次调用不再发"。

### 4.9 轮询被自己的限流打死（批次8 的回归）

改成任务模型后第一次实测，**第 2 次轮询就 429**：

```json
{"success": false, "message": "请求过于频繁（规划档限流），请 9 秒后重试"}
```

原因：批次8 的 `classify()` 按**路径前缀**分档，`/api/trip/plan/` 开头一律算规划档
（6 次/分钟、每次扣 2 令牌）。同步接口时代这条路径下只有一个请求，没人发现问题；
加了轮询之后，`GET /plan/async/{id}` 每 1.2 秒就被扣 2 个令牌，50 次/分钟撞 6 次/分钟。

修法不是"把规划档调大"（那等于放开真正花钱的接口），而是**按成本分档**：

| 路径 | 档位 | 额度 | 会不会触发 LLM |
|---|---|---|---|
| `POST /api/trip/plan`、`POST /api/trip/plan/async`、`POST /plan/{id}/refine` | `plan` | 6/分钟，每次扣 2 | 会 |
| `GET /plan/async/{id}`、`GET .../stream`、`DELETE .../{id}` | `poll`（新增） | 240/分钟 | 不会 |
| 其他 | `default` | 120/分钟 | — |

注意 `/api/trip/plan/async` 本身是 POST（提交任务，最贵），只有它下面的 `/{task_id}` 才是纯读。

### 4.10 分级探活

| 端点 | 语义 | 缺依赖时 |
|---|---|---|
| `GET /healthz` | 存活：进程还在不在 | 永远 200（依赖挂了不代表进程该被重启） |
| `GET /readyz` | 就绪：能不能接活 | `ready` 200 / `degraded` 200 / `not_ready` 503 |

`degraded` 仍然返回 200 是刻意的：缺高德 Key 只是拿不到真实 POI，计划还能出，
这时候摘流量是自伤。真正该 503 的只有"没有 LLM Key"。

---

## 5. 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/trip/plan` | 同步生成（兼容保留），返回计划 + `plan_id` |
| POST | `/api/trip/plan/async` | 提交异步任务，立即返回 `task_id` |
| GET | `/api/trip/plan/async/{task_id}` | 轮询任务状态（同时是"我还在等"的心跳） |
| GET | `/api/trip/plan/async/{task_id}/stream` | **SSE 推送**任务进度与结果 |
| DELETE | `/api/trip/plan/async/{task_id}` | **取消任务**：pending 立即 canceled，running 0.4s 内停下 |
| POST | `/api/trip/plan/{plan_id}/refine` | **增量修改**：改一句话，返回新计划 + 变更摘要 |
| GET | `/api/trip/health` | 旅行规划子服务健康检查 |
| GET | `/health` | 应用健康检查 |
| GET | `/healthz` | **存活探针**（永远 200，不检查依赖） |
| GET | `/readyz` | **就绪探针**（ready / degraded / not_ready 三档） |
| GET | `/metrics` | 指标快照（含限流/并发/合并/取消计数） |
| GET | `/docs` · `/redoc` | OpenAPI 文档 |

行程相关端点：**2 → 6 个**。

失败语义（全部有据可查，不再一律 500）：

| 状态码 | 含义 |
|---|---|
| 429 | 触发限流，带 `Retry-After` |
| 503 | 并发规划已满（`PLANNER_MAX_CONCURRENCY`），带 `Retry-After` |
| 409 | 该计划正在被另一轮修改占用；或试图取消一个已经结束的任务（结果已经在那儿了，不假装成功） |
| 502 | LLM 失败走了 fallback（返回降级计划 + `success=false`）或增量修改失败（原计划未受影响） |

---

## 6. 运行

```bash
cd backend
python -m venv venv
venv\Scripts\activate            # Windows；macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
# 复制 .env.example 为 .env，填入 LLM_API_KEY / AMAP_API_KEY
python run.py                    # 默认 http://localhost:7000
```

跑测试（不调用真实 LLM，不消耗额度）：

```bash
venv\Scripts\python.exe -m pytest -q
```

体验增量修改：

```bash
# 1) 生成，记下返回的 plan_id
curl -X POST http://localhost:7000/api/trip/plan -H "Content-Type: application/json" -d @request.json

# 2) 改一句话
curl -X POST http://localhost:7000/api/trip/plan/<plan_id>/refine \
     -H "Content-Type: application/json" \
     -d '{"instruction":"第二天下午太晒，改成室内场馆，其他两天不要动"}'
```

---

## 7. 关键设计取舍（面试常问）

| 问题 | 取舍 |
|---|---|
| 为什么不用 Redis 做缓存？ | 单机单进程够用；缓存封装在 `_TTLCache` 一个类里，要共享时换实现即可，调用方不动 |
| 为什么不用 Celery / MQ 做任务队列？ | 单机、低并发，没有削峰和解耦需求。`BackgroundTasks` 零依赖就够；规模上来再换，接口不变 |
| 为什么不用多 worker？ | 多 worker 是**多进程**，会放大原来的全局单例问题，也不解决进程内串台。先做"不阻塞 + 无状态"，再谈横向扩容 |
| 多线程有 GIL 为什么能加速？ | 瓶颈是等 LLM 的网络 IO，**阻塞时 GIL 会释放**；CPU 密集场景才无效 |
| 为什么每个候选用独立 Agent？ | `SimpleAgent._history` 是实例状态，共享会互相清空 → 串台 |
| 为什么超时不重试？ | 单次超时已 600 秒，重试只会让用户多等十分钟，应交给降级/异步任务 |
| 为什么不上 Prometheus？ | 单机项目增重；指标封装在 `MetricsRegistry`，生产可整体替换 |
| 为什么工具轮次设 2 而不是 1？ | 实测：1 轮反而更差（候选有效率 1/3、101.8s），模型没轮次收口，输出不完整导致解析失败重试 |
| grounding 匹配不到为什么不删？ | 高德召回有缺口，删掉会把真实的老店小馆误杀；只标 `verified=false`，判断权交给前端 |
| 坐标容差为什么是 800 米？ | 同景区不同入口通常几百米内；再往上基本是另一个地方。常量可调，属于经验值 |
| 增量修改为什么不做 3 候选 rerank？ | rerank 倾向选"差异最大"的方案，正好违背"最小改动"；单次 + 低温 0.1 更可控 |
| 为什么不给 token 级流式？ | **和 rerank 原理互斥**：流式要边生成边推，rerank 要等全部候选。选择保 rerank 质量，用任务级 SSE 缓解等待 |
| 增量修改失败为什么不降级？ | 用户手里已有上一版计划，返回模板计划会把满意的内容覆盖掉，比报错更糟 |

---

## 8. 测试策略

| 层次 | 做法 | 说明 |
|---|---|---|
| 单元/接口测试 | `pytest` **92 个用例**，`TestClient` + `monkeypatch` | 全部不烧额度，秒级完成，可随时跑 |
| 真实链路验证 | 手工打真实请求（DeepSeek + 高德），记录耗时与指标 | 单元测试不验证模型行为，真实行为必须实测 |
| 覆盖重点 | 解析鲁棒性（11）、Grounding（12）、增量修改（8）、缓存与退避（10）、工具预算（16）、错误语义（3+） | 优先覆盖"改动过的地方"和"出错代价高的地方" |

---

## 9. 已知短板（不做粉饰）

1. 缓存只有 TTL + LRU，**没有分布式共享**，多实例部署会各自为战；
2. 任务表、计划表都在进程内存，**重启即丢**，不支持多实例；
3. SSE 未实现 `Last-Event-ID` 断点续传（任务状态可直接重查，属有意简化）；
4. 流式连接的耗时会算进 `avg_latency_ms`，指标有偏差；
5. **Grounding 是闭集校验**：只保证"输出与候选池一致"，不保证"候选池本身正确"。开放域场景需要外部检索或知识库，这套方案不适用；
6. **增量修改没有自动纠偏**：如果模型超范围改了不该改的天，目前只能靠 diff 事后发现，没有"检测到越界就重试"；
7. 增量修改存在**累积漂移**风险（每轮都是一次生成），缺少版本历史与回滚；
8. `trip_planner_agent.py` 已 800+ 行，`plan_trip()` 职责偏多，下一步应拆出"生成策略"模块；
9. 坐标容差 800 米是经验值，**没做阈值敏感性实验**。
10. 限流与并发闸门都是**进程内状态**：多 worker 下每个进程各算一份，实际额度是 N 倍；
11. ant-design-vue 仍是 `app.use(Antd)` **全量注册**，首屏还有 1.43MB 是它；
12. 前端改动只验证到 `npm run build`（含 `vue-tsc` 类型检查）与接口联通，**没有 E2E**。

---

## 10. 后续可做

- 结构化输出（JSON mode / 约束解码）从源头降低解析失败率
- 增量修改加"越界检测 + 自动重试"，并把版本历史落库支持回滚
- Grounding 引入第二数据源交叉验证（解决"候选本身脏"的问题）
- 统一错误码体系（可重试 / 不可重试 / 降级）
- 前端接入 SSE 与 refine 接口
- 限流/闸门换成 Redis 实现（`guard.py` 的 `check()` 接口不变）
- ant-design-vue 改按需注册，首屏可再降约 1MB，但需在真实浏览器逐页验证 CSS-in-JS 注入

---

## 11. 致谢与许可

原项目：[helloagents-trip-planner](https://github.com/nameless0120/helloagents-trip-planner)（作者 nameless0120），采用 **CC BY-NC-SA 4.0** 许可。
本改造为**学习用途的二次开发**，遵循相同许可：署名、非商业使用、相同方式共享。
