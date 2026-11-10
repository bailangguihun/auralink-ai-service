"""Offline prompt checks cover policy and image-reference scope, not model quality."""
from types import SimpleNamespace
from app.assistant.prompts import system_prompt


def request(purpose="CHAT", images=None):
    return SimpleNamespace(
        purpose=purpose, citations=[], retrievalStates={},
        messages=[SimpleNamespace(role="user", images=images or [])],
    )


def test_analysis_connects_observation_relationships_and_interpretation():
    prompt = system_prompt(request())
    assert "画面依据" in prompt and "呼应、对照或空间关系" in prompt
    assert "不能仅凭枯树或飞鸟推定" in prompt
    assert "默认以自然语言回答，不追加选择卡" in prompt
    assert "不要求用户查找系统内部编号" in prompt
    assert "assistant-crop:" not in prompt


def test_crop_instructions_require_current_user_images_and_chat():
    prompt = system_prompt(request(images=[object()]))
    assert "assistant-crop:N,x,y,w,h" in prompt
    assert "只引用本轮用户上传的图片" in prompt
    assert "无法定位时省略配图" in prompt
    summary = system_prompt(request(purpose="SUMMARY", images=[object()]))
    assert "assistant-crop:" not in summary
    assert "本次仅摘要" in summary
