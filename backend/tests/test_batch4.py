"""批次4改造的回归测试：LLM 输出的 JSON 解析鲁棒性。

背景：实测日志里大量"输出解析失败：响应中未找到完整的顶层 TripPlan JSON对象"，
每次失败都要重跑一次几十秒的 LLM 调用。这里把模型实际会犯的几种错都固化成用例。
"""

import json

import pytest

from app.planner.output import extract_json_object, output_parse_stats

PLAN = {
    "city": "成都",
    "start_date": "2026-11-25",
    "end_date": "2026-11-27",
    "days": [],
    "overall_suggestions": "建议提前查看景点开放时间",
}


def test_plain_json():
    """最基本情况：整段就是一个合法 JSON。"""
    assert extract_json_object(json.dumps(PLAN, ensure_ascii=False)) == PLAN


def test_json_in_markdown_fence():
    """包在 ```json 代码块里。"""
    text = "```json\n" + json.dumps(PLAN, ensure_ascii=False) + "\n```"
    assert extract_json_object(text) == PLAN


def test_text_before_and_after_json():
    """模型前后加客套话（原实现会直接失败的两大高频场景）。"""
    text = (
        "好的，我为您规划了成都三日游，以下是完整行程：\n\n"
        + json.dumps(PLAN, ensure_ascii=False)
        + "\n\n希望对您有帮助！如需调整请告诉我。"
    )
    assert extract_json_object(text) == PLAN


def test_trailing_comma():
    """尾逗号：模型最常犯的 JSON 语法错误。"""
    text = json.dumps(PLAN, ensure_ascii=False).replace("}", ",}", 1)
    assert extract_json_object(text) == PLAN


def test_chinese_quotes():
    """模型把英文引号写成中文引号。"""
    text = json.dumps(PLAN, ensure_ascii=False).replace('"', "“", 1)
    text = text.replace('"', "”")
    assert extract_json_object(text) == PLAN


def test_truncated_inside_string():
    """max_tokens 用满，JSON 被切在字符串中间 -> 应补全引号和括号。"""
    text = (
        '{"city": "成都", "start_date": "2026-11-25", "end_date": "2026-11-27", '
        '"days": [], "overall_suggestions": "建议提前查看景点开放时间'
    )
    result = extract_json_object(text)
    assert result["city"] == "成都"
    assert result["days"] == []
    assert result["overall_suggestions"].startswith("建议提前查看")


def test_truncated_inside_array():
    """截断在数组内部 -> 应把 ] 和 } 补齐。

    注意：必需字段要出现在截断点之前，否则补全后依然缺字段（这是预期行为，
    缺字段的计划本来就不该被接受）。
    """
    text = (
        '{"city": "成都", "start_date": "2026-11-25", "end_date": "2026-11-27", '
        '"overall_suggestions": "备注", "days": [{"day_index": 0, "date": "2026-11-25"'
    )
    result = extract_json_object(text)
    assert len(result["days"]) == 1
    assert result["days"][0]["day_index"] == 0


def test_invisible_chars_before_brace():
    """BOM / 零宽字符黏在 `{` 前面。"""
    text = "\ufeff\u200b" + json.dumps(PLAN, ensure_ascii=False)
    assert extract_json_object(text) == PLAN


def test_nested_object_only_must_fail():
    """只有内层 day 对象时必须失败，不能把 day 当成 TripPlan 返回。"""
    text = '{"day_index": 0, "date": "2026-11-25", "description": "第一天"}'
    with pytest.raises(ValueError):
        extract_json_object(text)


def test_broken_structure_must_fail():
    """括号不匹配的坏结构必须报错，而不是硬拼出一个错的计划。"""
    text = '{"city": "成都"}]'
    with pytest.raises(ValueError):
        extract_json_object(text)


def test_parse_stats_are_recorded():
    """解析统计要能区分三种结果：直接成功 / 修复救回 / 彻底失败。

    注意：前后带说明文字的情况属于"直接成功"——因为扫描策略会跳过前缀废话
    找到 JSON，不需要走修复流程。只有尾逗号、截断、中文引号才计入 repaired。
    """
    before = output_parse_stats()
    extract_json_object(json.dumps(PLAN, ensure_ascii=False))  # 直接成功

    with_trailing_comma = json.dumps(PLAN, ensure_ascii=False).replace("}", ",}", 1)
    extract_json_object(with_trailing_comma)  # 靠"去尾逗号"修复救回

    with pytest.raises(ValueError):
        extract_json_object('{"city": "成都"}]')  # 彻底失败

    after = output_parse_stats()
    assert after["attempts"] - before["attempts"] == 3
    assert after["success"] - before["success"] >= 1
    assert after["repaired"] - before["repaired"] >= 1
    assert after["failed"] - before["failed"] == 1
