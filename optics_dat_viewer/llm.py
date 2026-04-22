from __future__ import annotations

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def call_siliconflow_llm(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    data_text: str | None = None,
) -> str:
    if OpenAI is None:
        return "当前环境未安装 openai 依赖，请先安装 requirements。"
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
        full_prompt = user_prompt
        if data_text:
            full_prompt += f"\n\n--- 数据附件（CSV 格式） ---\n{data_text}"
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        return f"调用失败: {exc}"
