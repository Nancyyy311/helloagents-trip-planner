import { describe, expect, it } from 'vitest'
import {
  buildAttractionPlaceholder,
  canRemoveAttraction,
  getMealLabel,
  getWeatherCardClass,
  getWeatherIcon,
  isLodgingBreakfastMeal,
  moveInList
} from '../planView'

describe('getMealLabel', () => {
  it('把已知餐次枚举翻成中文', () => {
    expect(getMealLabel('breakfast')).toBe('早餐')
    expect(getMealLabel('lunch')).toBe('午餐')
    expect(getMealLabel('dinner')).toBe('晚餐')
    expect(getMealLabel('snack')).toBe('小吃')
  })

  it('未知枚举原样返回，不返回 undefined', () => {
    // 模型偶尔会给出 brunch 这类不在词表里的值，页面不能显示空白
    expect(getMealLabel('brunch')).toBe('brunch')
    expect(getMealLabel('')).toBe('')
  })
})

describe('getWeatherIcon', () => {
  it('雷优先于雨：雷阵雨必须是雷电而不是普通雨', () => {
    // 判断顺序错误时这里会退化成 🌦️
    expect(getWeatherIcon('雷阵雨', 'day')).toBe('⛈️')
    expect(getWeatherIcon('雷阵雨', 'night')).toBe('⛈️')
  })

  it('暴雨/大雨区别于小雨', () => {
    expect(getWeatherIcon('暴雨', 'day')).toBe('🌧️')
    expect(getWeatherIcon('大雨', 'day')).toBe('🌧️')
    expect(getWeatherIcon('小雨', 'day')).toBe('🌦️')
  })

  it('晴和云按昼夜给不同图标', () => {
    expect(getWeatherIcon('晴', 'day')).toBe('☀️')
    expect(getWeatherIcon('晴', 'night')).toBe('🌙')
    expect(getWeatherIcon('多云', 'day')).toBe('⛅')
    expect(getWeatherIcon('多云', 'night')).toBe('☁️')
  })

  it('未知/空/缺失一律返回破折号，不猜', () => {
    expect(getWeatherIcon('未知', 'day')).toBe('—')
    expect(getWeatherIcon('暂无数据', 'day')).toBe('—')
    expect(getWeatherIcon('', 'day')).toBe('—')
    expect(getWeatherIcon(null, 'day')).toBe('—')
    expect(getWeatherIcon(undefined, 'night')).toBe('—')
  })

  it('未覆盖的天气文案有兜底，不会返回空字符串', () => {
    expect(getWeatherIcon('浮尘转晴', 'day')).toBe('🌪️')
    expect(getWeatherIcon('说不清的天气', 'day')).toBe('🌤️')
    expect(getWeatherIcon('说不清的天气', 'night')).toBe('🌙')
  })
})

describe('getWeatherCardClass', () => {
  it('雨和雷优先于其他天气', () => {
    expect(getWeatherCardClass('多云', '雷阵雨')).toBe('weather-card-rain')
  })

  it('雪、霾、云分别落到对应配色', () => {
    expect(getWeatherCardClass('小雪', '晴')).toBe('weather-card-snow')
    expect(getWeatherCardClass('霾', '晴')).toBe('weather-card-haze')
    expect(getWeatherCardClass('阴', '多云')).toBe('weather-card-cloud')
    expect(getWeatherCardClass('晴', '晴')).toBe('weather-card-sun')
  })

  it('两段都缺失时走 unknown，而不是当成晴天', () => {
    expect(getWeatherCardClass(null, undefined)).toBe('weather-card-unknown')
    expect(getWeatherCardClass('暂无', '未知')).toBe('weather-card-unknown')
  })
})

describe('isLodgingBreakfastMeal', () => {
  it('只认早餐，午餐名里带酒店也不算', () => {
    expect(isLodgingBreakfastMeal('lunch', '酒店早餐')).toBe(false)
    expect(isLodgingBreakfastMeal('breakfast', '酒店早餐')).toBe(true)
  })

  it('四种住宿早餐写法都能识别', () => {
    for (const name of ['酒店早餐', '民宿早餐', '客栈早餐', '住宿早餐']) {
      expect(isLodgingBreakfastMeal('breakfast', name)).toBe(true)
    }
  })

  it('普通早餐店名不误判', () => {
    expect(isLodgingBreakfastMeal('breakfast', '老门东早点铺')).toBe(false)
  })

  it('名字为 null 时不抛异常', () => {
    // 模型可能给出 name: null，这里必须返回 false 而不是崩掉整页回填流程
    expect(isLodgingBreakfastMeal('breakfast', null as unknown as string)).toBe(false)
  })
})

describe('buildAttractionPlaceholder', () => {
  const decode = (dataUri: string) =>
    decodeURIComponent(escape(atob(dataUri.replace('data:image/svg+xml;base64,', ''))))

  it('产出可用的 SVG data URI', () => {
    const uri = buildAttractionPlaceholder('中山陵', 0)
    expect(uri.startsWith('data:image/svg+xml;base64,')).toBe(true)
    const svg = decode(uri)
    expect(svg).toContain('<svg')
    expect(svg).toContain('#1677ff')
  })

  it('中文景点名编码后能完整还原', () => {
    // btoa 不能直接吃非 ASCII，原实现靠 unescape(encodeURIComponent()) 绕开；
    // 这里验证绕开之后确实没有乱码
    const svg = decode(buildAttractionPlaceholder('南京博物院', 1))
    expect(svg).toContain('南京博物院')
  })

  it('不同 index 轮换配色，避免同页撞色', () => {
    const first = decode(buildAttractionPlaceholder('A', 0))
    const second = decode(buildAttractionPlaceholder('B', 1))
    expect(first).not.toBe(second)
    expect(first).toContain('#1677ff')
    expect(second).toContain('#0f766e')
  })

  it('负数索引不越界（原实现会取到 undefined 并抛错）', () => {
    expect(() => buildAttractionPlaceholder('A', -1)).not.toThrow()
    // -1 应归一化到最后一组配色
    expect(decode(buildAttractionPlaceholder('A', -1))).toContain('#334155')
  })

  it('超大索引也能安全取色', () => {
    expect(() => buildAttractionPlaceholder('A', 100000)).not.toThrow()
    expect(decode(buildAttractionPlaceholder('A', 100000))).toContain('#1677ff')
  })
})

describe('moveInList', () => {
  it('上移一位并原地修改', () => {
    const list = ['a', 'b', 'c']
    expect(moveInList(list, 1, 'up')).toBe(true)
    expect(list).toEqual(['b', 'a', 'c'])
  })

  it('下移一位并原地修改', () => {
    const list = ['a', 'b', 'c']
    expect(moveInList(list, 1, 'down')).toBe(true)
    expect(list).toEqual(['a', 'c', 'b'])
  })

  it('首项上移是无操作，且返回 false', () => {
    const list = ['a', 'b']
    expect(moveInList(list, 0, 'up')).toBe(false)
    expect(list).toEqual(['a', 'b'])
  })

  it('末项下移是无操作，且返回 false', () => {
    const list = ['a', 'b']
    expect(moveInList(list, 1, 'down')).toBe(false)
    expect(list).toEqual(['a', 'b'])
  })

  it('越界或非法索引不抛异常', () => {
    const list = ['a']
    for (const bad of [-1, 1, 99, 1.5, NaN]) {
      expect(() => moveInList(list, bad, 'up')).not.toThrow()
      expect(moveInList(list, bad, 'down')).toBe(false)
    }
    expect(list).toEqual(['a'])
  })

  it('空数组不崩', () => {
    expect(moveInList([], 0, 'up')).toBe(false)
  })
})

describe('canRemoveAttraction', () => {
  it('只剩一个景点时不允许删除', () => {
    expect(canRemoveAttraction(['only'])).toBe(false)
  })

  it('两个及以上允许删除', () => {
    expect(canRemoveAttraction(['a', 'b'])).toBe(true)
  })

  it('空数组与非数组都返回 false', () => {
    expect(canRemoveAttraction([])).toBe(false)
    expect(canRemoveAttraction(null as unknown as unknown[])).toBe(false)
  })
})
