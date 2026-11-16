"""Offline prompt checks cover policy and image-reference scope, not model quality."""
from types import SimpleNamespace

from app.assistant.prompts import system_prompt
from app.assistant.providers import provider_body
from app.assistant.schemas import Execution


def request(purpose="CHAT", images=None, content=""):
    return SimpleNamespace(
        purpose=purpose, citations=[], retrievalStates={},
        messages=[SimpleNamespace(role="user", images=images or [], content=content)],
    )


def test_analysis_connects_observation_relationships_and_interpretation():
    prompt = system_prompt(request())
    assert "先给出一个整体读法" in prompt
    assert "呼应、对照或空间关系" in prompt and "墨色和留白如何引导视线" in prompt
    assert "断代、作者、材质、题跋释读、印章归属或创作意图" in prompt
    assert "仅有文字描述时，明确依据来自该描述" in prompt
    assert "默认以自然语言回答" in prompt
    assert "assistant-choices" not in prompt
    assert "不要求用户查找系统内部编号" in prompt
    assert "逐项对照画面" not in prompt
    assert "assistant-crop:" not in prompt


def test_crop_instructions_require_current_user_images_and_chat():
    prompt = system_prompt(request(images=[object()]))
    assert "物象、数量、颜色、位置和相互关系逐项对照画面" in prompt
    assert "具体物种、飞行方向、季节、是否裁切或不可辨的文字" in prompt
    assert "条件性的联想" in prompt
    assert "默认给可以定位的主要物象配上原图局部" in prompt
    assert "assistant-crop:N,bbox_2d,x1,y1,x2,y2" in prompt
    assert "只引用本轮用户上传的图片" in prompt
    assert "无法定位时省略配图" in prompt
    summary = system_prompt(request(purpose="SUMMARY", images=[object()]))
    assert "assistant-crop:" not in summary
    assert "逐项对照画面" not in summary
    assert "本次只整理当前分支" in summary
    assert all(section in summary for section in ("对话概览", "关键要点", "已形成结论", "后续行动", "待确认问题"))
    assert "assistant-crop:" not in system_prompt(request(images=[object()], content="请纯文字赏析这幅图"))


def test_follow_up_can_analyze_history_image_without_referencing_it_as_current_crop():
    follow_up = request(images=[object()])
    follow_up.messages.extend([
        SimpleNamespace(role="assistant", images=[]),
        SimpleNamespace(role="user", images=[], content="再说说留白"),
    ])
    prompt = system_prompt(follow_up)
    assert "逐项对照画面" in prompt
    assert "短追问直接回答" in prompt
    assert "assistant-crop:" not in prompt


def test_visual_guidance_reaches_provider_with_the_uploaded_image():
    execution = Execution(
        contractVersion=1,
        runId="00000000-0000-4000-8000-000000000001",
        attemptId="00000000-0000-4000-8000-000000000002",
        fence=1,
        configSha256="0" * 64,
        purpose="CHAT",
        modelKey="vision",
        messages=[{
            "role": "user",
            "content": "分析一下这幅国画的意象",
            "images": [{"mimeType": "image/png", "dataBase64": "AA=="}],
        }],
    )
    body = provider_body(execution, SimpleNamespace(model="qwen3-vl-plus", thinkingMode="provider-default"))
    guidance, question = body["messages"]
    assert guidance["role"] == "system"
    assert "逐项对照画面" in guidance["content"]
    assert question == {"role": "user", "content": [
        {"type": "text", "text": "分析一下这幅国画的意象"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
    ]}
    assert body["model"] == "qwen3-vl-plus" and body["stream"] is True


def test_choice_prompt_applies_only_to_relevant_text_questions():
    prompt = system_prompt(request(content="在横幅和立轴之间如何选择？"))
    assert "两个至四个可执行、互斥的创作方向" in prompt
    assert "assistant-choices" in prompt
    assert "assistant-choices" not in system_prompt(request(images=[object()], content="比较这幅图的构图方向"))
