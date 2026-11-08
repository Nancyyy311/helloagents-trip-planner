"""回归测试：在线失败样本的 jsonl 必须有体积上限。

缺陷：`append_jsonl` 原来只做 append，没有任何上限。
一条失败记录实测 48~56KB（带完整 prompt + 模型原始输出），
一天几百次失败就是几十 MB，长期运行会把磁盘写满；
更麻烦的是磁盘写满之后，连"写失败记录"这个动作本身都会抛异常。

修法：写入前检查体积，超过上限（默认 32MB，可用环境变量覆盖）
就改名成 `.1`，只保留一代。
"""

import app.agents.planner_feedback as pf


def _row(i: int, pad: int = 40) -> dict:
    return {"i": i, "pad": "x" * pad}


def test_no_rotation_below_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "JSONL_MAX_BYTES", 10_000)
    log = tmp_path / "fails.jsonl"

    for i in range(3):
        pf.append_jsonl(log, _row(i))

    assert log.exists()
    assert len(log.read_text(encoding="utf-8").splitlines()) == 3
    assert not (tmp_path / "fails.jsonl.1").exists()


def test_rotate_when_oversized(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "JSONL_MAX_BYTES", 200)
    log = tmp_path / "fails.jsonl"

    for i in range(10):
        pf.append_jsonl(log, _row(i))

    backup = tmp_path / "fails.jsonl.1"
    assert backup.exists(), "超过上限必须轮转，否则文件无限增长"
    assert log.exists(), "轮转后主文件应重新开始写"
    # 轮转发生了一次：旧的进了 .1，新的在主文件，两边都不空
    assert len(backup.read_text(encoding="utf-8").splitlines()) > 0
    assert len(log.read_text(encoding="utf-8").splitlines()) > 0


def test_only_one_generation_kept(tmp_path, monkeypatch):
    """只保留一代：多的直接被覆盖，不能让备份无限增殖。"""
    monkeypatch.setattr(pf, "JSONL_MAX_BYTES", 150)
    log = tmp_path / "fails.jsonl"

    for i in range(30):
        pf.append_jsonl(log, _row(i))

    backups = sorted(p.name for p in tmp_path.glob("fails.jsonl.*"))
    assert backups == ["fails.jsonl.1"], backups


def test_rotation_failure_does_not_break_caller(tmp_path, monkeypatch):
    """落盘出问题不能拖垮一次正常的规划请求。"""
    monkeypatch.setattr(pf, "JSONL_MAX_BYTES", 10)
    log = tmp_path / "fails.jsonl"
    log.write_text("x" * 100, encoding="utf-8")  # 先让它超限

    def _boom(self, target):
        raise OSError("disk unavailable")

    monkeypatch.setattr(pf.Path, "rename", _boom, raising=True)
    pf.append_jsonl(log, _row(0))  # 不应抛出，主流程不能被日志拖死
    assert log.exists()
