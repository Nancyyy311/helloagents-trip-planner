"""把"解析失败"拆成可执行的病因分类。

背景
----
改造前所有失败都长一个样：`响应中未找到完整的顶层TripPlan JSON对象`。
可实际上至少有两类完全不同的病因混在里面：

* 输出被 max_tokens 截断 → JSON 天然不闭合，改 prompt 没用，要动输出预算；
* 模型正常收尾但结构不合 schema → 这才是 prompt/格式约束的问题。

把 38 条历史失败日志里的原文捞出来看，13 条有原文的样本里 12 条结尾停在
半句话或半个数字上（例如 `"longitude": 116`），只有 1 条是完整 JSON。
也就是说**绝大多数失败根本不是"模型不会输出 JSON"**，而是输出没写完。
分类错误会直接把优化方向带偏，所以这里单独做一层归因。

判定优先级
----------
1. `finish_reason == "length"` 是最权威的信号（API 明确告知被截断）；
2. 没有元信息时（历史样本、第三方兼容实现不给该字段），退化为结构扫描：
   括号是否闭合、结尾是否停在字符串/数字中间；
3. 结构完整但校验不过，才算 `schema_invalid`。
"""

import json
from typing import Optional

# 分类取值集中定义，避免各处写字符串字面量对不上
EMPTY = "empty"
NOT_JSON = "not_json"
TRUNCATED_LENGTH = "truncated_length"
TRUNCATED_IN_STRING = "truncated_in_string"
TRUNCATED_IN_NUMBER = "truncated_in_number"
TRUNCATED_STRUCTURE = "truncated_structure"
BROKEN_STRUCTURE = "broken_structure"
TRAILING_CONTENT = "trailing_content"
SCHEMA_INVALID = "schema_invalid"

# 前端/文档展示用：哪些分类属于"输出没写完"，哪些属于"写完了但不合规"
TRUNCATION_KINDS = frozenset(
    {
        TRUNCATED_LENGTH,
        TRUNCATED_IN_STRING,
        TRUNCATED_IN_NUMBER,
        TRUNCATED_STRUCTURE,
    }
)

_STRUCTURAL_KINDS = frozenset({TRUNCATED_STRUCTURE, TRUNCATED_IN_NUMBER, TRUNCATED_IN_STRING})

_DESCRIPTIONS = {
    EMPTY: "模型返回空内容",
    NOT_JSON: "返回内容里没有 JSON 对象（可能是纯文本解释）",
    TRUNCATED_LENGTH: "finish_reason=length，输出被 max_tokens 截断",
    TRUNCATED_IN_STRING: "截断在字符串中间（括号未闭合）",
    TRUNCATED_IN_NUMBER: "截断在数字中间（括号未闭合）",
    TRUNCATED_STRUCTURE: "截断在结构中间（括号未闭合）",
    BROKEN_STRUCTURE: "括号错配，结构已损坏",
    TRAILING_CONTENT: "JSON 已完整，后面又多了内容（尾部重复/补充说明）",
    SCHEMA_INVALID: "JSON 完整但不符合 TripPlan schema",
}

# 这些分类其实**已经解出了完整对象**，属于"解析窗口没找准"，
# 一个会扫描的解析器（extract_json_object 就是这么做的）能自动救回。
RECOVERABLE_KINDS = frozenset({TRAILING_CONTENT})


def describe(kind: str) -> str:
    """给分类一句人话，用于日志和报表。"""
    return _DESCRIPTIONS.get(kind, kind)


def _structural_state(text: str) -> dict:
    """扫描一遍文本，返回括号闭合状态与结尾字符类型。

    字符串内部（含转义）不参与括号计数，否则中文描述里的 `{` 会误判。
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    mismatched = False
    last_significant = ""

    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if stack and stack[-1] == char:
                stack.pop()
            else:
                mismatched = True
                break

        if not char.isspace():
            last_significant = char

    return {
        "unclosed": len(stack),
        "in_string": in_string,
        "mismatched": mismatched,
        "last": last_significant,
    }


def _ends_in_number(last_significant: str) -> bool:
    return last_significant.isdigit() or last_significant in ".-+eE"


def _complete_prefix_end(text: str) -> int:
    """若从第一个 `{` 起能解出完整 JSON 对象，返回该对象结束后的偏移；否则 -1。

    存在的意义：实测样本里有一种很隐蔽的失败——模型输出完完整 JSON 后，
    又把 `,"budget":{...}}` 这段尾巴重复输出了一遍。整体看括号是"错配"的，
    但前面其实是一个完全合法的 TripPlan。这种要归到 `trailing_content`，
    因为解析器只要肯扫描就能救回来，而 `broken_structure` 会让人误以为原文废了。
    """
    start = text.find("{")
    if start < 0:
        return -1
    try:
        obj, end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return -1
    if not isinstance(obj, dict) or not obj:
        return -1
    return start + end


def classify_failure(
    raw_response: str,
    finish_reason: Optional[str] = None,
    *,
    json_parsed: bool = False,
) -> str:
    """判定一次失败的病因。

    Args:
        raw_response: 模型原始输出。
        finish_reason: 来自 LLM 元信息，缺失时传 None（历史样本就是这种情况）。
        json_parsed: 是否已经成功解析出 Python 对象。为 True 说明结构没问题，
            失败发生在 schema 校验阶段。

    Returns:
        上面定义的分类常量之一。
    """
    if json_parsed:
        return SCHEMA_INVALID

    text = (raw_response or "").strip()
    if not text:
        return EMPTY

    # API 明确说被截断了，比任何结构推断都可信
    if finish_reason == "length":
        return TRUNCATED_LENGTH

    if "{" not in text:
        return NOT_JSON

    # 先看"其实已经解出完整对象"的情况，否则会被后面的括号错配判成结构损坏
    prefix_end = _complete_prefix_end(text)
    if prefix_end > 0 and text[prefix_end:].strip():
        return TRAILING_CONTENT

    state = _structural_state(text)
    if state["mismatched"]:
        return BROKEN_STRUCTURE

    if state["in_string"] or state["unclosed"]:
        if state["in_string"]:
            return TRUNCATED_IN_STRING
        if _ends_in_number(state["last"]) or state["last"] in ",:":
            return TRUNCATED_IN_NUMBER
        return TRUNCATED_STRUCTURE

    # 括号闭合、也不在字符串里，却仍然解析不出对象：
    # 说明 JSON 被别的东西污染了（前后夹解释文字、非法转义等），
    # 归到 not_json 更贴近事实
    return NOT_JSON


def is_truncation(kind: str) -> bool:
    """该分类是否属于"输出没写完"。"""
    return kind in TRUNCATION_KINDS


def is_recoverable(kind: str) -> bool:
    """该分类是否属于"原文其实够用，是解析没找准"。"""
    return kind in RECOVERABLE_KINDS
