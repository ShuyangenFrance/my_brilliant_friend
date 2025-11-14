import gradio as gr
import re
import os
import random
from openai import OpenAI

# =====================
# 人格 Prompt
# =====================
PERSONALITY_PROMPTS = {
    "知性姐姐": """
📚 你是知性姐姐，温柔、理性，但绝不是无原则地顺从用户。
你的温柔带锋利，是那种说一句话就能让人安静思考的气质。
【你的对抗方式】
- 用户表达含糊时，你温柔追问：“额…你能具体一点吗？”
- 当用户逃避问题，你轻轻刺一句：
  “我觉得…你好像不是没想法，而是不敢面对。”
- 用户找借口时，你温和而精准地拆招：
  “这听上去更像一个理由，而不是答案。”
- 当用户逻辑不通时，你不迎合，直接指出：
  “你自己听听，现在前后有点矛盾哦。”
你不会吵，也不会毒，但你会让人无处可逃地清醒。
""",

    "丧气宅女": """
😪 你是丧气宅女，慢悠悠、阴阳怪气、有点自嘲，但你从不惯着别人。
你的怼人方式不是吵，而是“懒得哄你，但会说真话”。
【你的对抗方式】
- 用户含糊其辞 → “额…啥意思啊？”
- 用户自欺欺人 → “你这是在骗我，还是骗你自己？”
- 用户找借口 → “这个理由有点…嗯…不太能骗过我。”
- 用户逻辑离谱 → “你这脑回路挺可爱的，但不太对欸。”
你看似丧丧的，但怼人的时候狠得刚刚好。
""",

    "阳光E人": """
🌞 你是阳光E人，元气满满、热情开朗，但绝不是无脑夸。
你喜欢用玩笑、调侃、轻松的方式拆穿用户的逻辑盲区。
【你的对抗方式】
- 用户讲得不清楚 → “哎呀？然后呢～😆”
- 用户逃避问题 → “你是不是在偷偷绕开重点～我看到了喔！”
- 用户给自己找借口 → “欸你这个借口好可爱…但一点都站不住脚😂”
- 用户逻辑跳脱时 → “等一下！你这逻辑太自由了吧哈哈哈”
你是温暖的阳光，但也是照出问题的那种光。
""",

    "毒舌御姐": """
😏 你是毒舌御姐，聪明、犀利、直爽，是最敢怼用户的人格。
但你的毒不是恶意，是清醒，是帮对方面对现实。
【你的对抗方式】
- 用户含糊不清 → “额？你到底想说啥？”
- 用户逃避现实 → “别转移话题。说重点。”
- 用户自欺欺人 → “你这个理由骗得了别人，骗不了我。”
- 用户逻辑崩坏 → “你前一句还是A，后一句突然变成B了，你自己不觉得怪？”
你嘴毒、心暖，怼人精准，不留情面但会留余地。
"""
}

SYSTEM_PROMPT_TEMPLATE = """
{personality_prompt}
【行为逻辑】
- 用户问候 → 自然回应。
- 用户表达情绪 → 共情。
- 用户面临选择 → 给出 2～3 个方向，每个方向前加“-”。
- 回答尽量简短，不啰嗦。
"""

# =====================
# DeepSeek 设置
# =====================
MODEL_NAME = "deepseek-chat"
MAX_TOKENS_PER_ROUND = 2000
DEFAULT_DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
MAX_USER_TOKENS = 50000
balance_status_value = "💚 开发者账户状态：正常，你接着用哈！（但最近很多用户聊high了我账户里剩余不多啦～）"

# =====================
# Token 粗略估算函数
# =====================
def estimate_tokens(text):
    return len(text)

# =====================
# 单人格调用 DeepSeek
# =====================
def call_deepseek(prompt, history_state, personality, balance_status):
    global balance_status_value

    full_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        personality_prompt=PERSONALITY_PROMPTS[personality]
    )
    messages = [{"role": "system", "content": full_prompt}]

    for u, a in history_state["messages"]:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": prompt})

    api_key = DEFAULT_DEEPSEEK_KEY
    if not api_key:
        return "⚠️ 开发者 API Key 未配置，请联系开发者。", history_state, balance_status_value

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS_PER_ROUND,
            temperature=0.8,
        )
        reply = response.choices[0].message.content.strip()

        history_state["used_tokens"] += estimate_tokens(prompt) + estimate_tokens(reply)
        history_state["messages"].append((prompt, reply))
        balance_status_value = "💚 开发者账户状态：正常，你接着用哈！(但万一你聊high了我可能会没钱）"

        return reply, history_state, balance_status_value

    except Exception as e:
        err = str(e)
        if "402" in err or "Insufficient Balance" in err:
            balance_status_value = "❤️ 开发者账户状态：余额不足"
            return "💸 开发者账户里没钱了 😂", history_state, balance_status_value
        return f"❌ 调用 DeepSeek 时出错：{e}", history_state, balance_status_value


# =====================
# 群聊调用：随机 1～4 女友
# =====================
def call_group_chat(prompt, history_state, balance_status):
    global balance_status_value

    # 随机选 1~4 个女友
    girls = list(PERSONALITY_PROMPTS.keys())
    selected = random.sample(girls, random.randint(1, 4))

    client = OpenAI(api_key=DEFAULT_DEEPSEEK_KEY, base_url="https://api.deepseek.com")

    group_replies = []
    combined_text = ""

    # 每个女友按照前一个女友的话继续“接力”
    previous_replies = []  # 存储之前所有女友的回复

    for girl in selected:
        system_prompt = f"""
你是 {girl}，以下是你的人格设定：
{PERSONALITY_PROMPTS[girl]}

【群聊行为规则】
- 不要重复引用用户原话。
- 不要说“你说你肚子疼”这种多余内容。
- 如果之前已有女友发言，你可以轻微地接话，但不要互相攻击。
- 回复要简短、有个性。
- 只输出你自己的一句或几句回答，不要管其他人。
"""

        # 构建对话记录（让 AI 能接上前一个人的话）
        messages = [{"role": "system", "content": system_prompt}]

        # 用户问题只给一次
        messages.append({"role": "user", "content": prompt})

        # 让每个女友能看到前面所有女友的发言
        if previous_replies:
            context = "\n".join(previous_replies)
            messages.append({"role": "assistant", "content": f"前面的女友们说了：\n{context}\n\n现在轮到你回答了。"})

        try:
            res = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=300,
                temperature=0.8,
            )
            reply = res.choices[0].message.content.strip()
        except Exception as e:
            reply = f"（调用失败：{e}）"

        # 格式化回复，添加女友名字
        formatted_reply = f"**{girl}**\n{reply}"

        group_replies.append(formatted_reply)
        previous_replies.append(f"{girl}: {reply}")
        combined_text += formatted_reply + "\n\n"

    # 记录 Token
    history_state["used_tokens"] += estimate_tokens(prompt) + estimate_tokens(combined_text)
    history_state["messages"].append((prompt, combined_text))

    return group_replies, history_state, balance_status_value

# =====================
# 输入处理
# =====================
def user_input_fn(user_text, chat_history, history_state, branch_btns, personality, balance_status):
    user_text = (user_text or "").strip()
    if not user_text:
        return "", chat_history, gr.update(visible=False), history_state, balance_status

    if not history_state:
        history_state = {"messages": [], "used_tokens": 0}

    # 额度限制
    if history_state["used_tokens"] + estimate_tokens(user_text) > MAX_USER_TOKENS:
        return "😵‍💫 超出额度啦，下次再用吧！", chat_history, gr.update(visible=False), history_state, balance_status

    chat_history = chat_history or []

    # ========= 群聊模式 =========
    if personality == "叽叽喳喳群聊":
        # 调用群聊，获取每个女友的回复
        group_replies, history_state, balance_status = call_group_chat(user_text, history_state, balance_status)

        # 先显示用户输入（只有第一个女友的回复和用户输入配对）
        if group_replies:
            chat_history.append((user_text, group_replies[0]))

            # 其余女友的回复单独显示（用特殊标记占位）
            for reply in group_replies[1:]:
                chat_history.append(("　", reply))  # 使用全角空格作为占位符

        return "", chat_history, gr.update(visible=False), history_state, balance_status

    # ========= 单人格模式 =========
    reply, history_state, balance_status = call_deepseek(user_text, history_state, personality, balance_status)
    chat_history.append((user_text, reply))

    # 提取分支
    options = re.findall(r"[-•]\s*(.*)", reply)
    if options:
        cleaned = [o.strip() for o in options if len(o.strip()) > 2][:3]
        return "", chat_history, gr.update(choices=cleaned, value=None, visible=True), history_state, balance_status

    return "", chat_history, gr.update(visible=False), history_state, balance_status


# =====================
# 选择分支
# =====================
def choose_branch_fn(selected, chat_history, history_state, branch_btns, personality, balance_status):
    if not selected:
        return chat_history, gr.update(visible=False), history_state, balance_status

    message = f"我倾向于：{selected}"

    # 群聊不支持分支
    if personality == "叽叽喳喳群聊":
        group_replies, history_state, balance_status = call_group_chat(message, history_state, balance_status)
        for name, rep in group_replies:
            chat_history.append((f"{message}（→ {name}）", rep))
        return chat_history, gr.update(visible=False), history_state, balance_status

    # 单人格分支
    reply, history_state, balance_status = call_deepseek(message, history_state, personality, balance_status)
    chat_history.append((message, reply))

    # 提取分支
    options = re.findall(r"[-•]\s*(.*)", reply)
    if options:
        cleaned = [o.strip() for o in options][:3]
        return chat_history, gr.update(choices=cleaned, value=None, visible=True), history_state, balance_status

    return chat_history, gr.update(visible=False), history_state, balance_status


# =====================
# 清空
# =====================
def clear_all():
    return [], gr.update(visible=False), [], balance_status_value


# =====================
# Gradio UI
# =====================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="violet"), css="""
#balance_status .value {
    font-size: 14px !important;
    font-weight: normal !important;
}
#chatbot .user { background-color: #f0f0f0; border-radius:12px; padding:6px; }
#chatbot .assistant { border-radius:12px; padding:6px; }
/* 隐藏只包含全角空格的用户消息框 */
.message.user .prose:empty,
.message.user .prose:has(br:only-child) {
    display: none !important;
}
.message.user:has(.prose:empty) {
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}
""") as demo:

    gr.Markdown("## 💬 你的天才女友们（含叽叽喳喳群聊）")

    with gr.Row():
        with gr.Column(scale=3):
            personality_dropdown = gr.Dropdown(
                choices=["知性姐姐", "丧气宅女", "阳光E人", "毒舌御姐", "叽叽喳喳群聊"],
                value="知性姐姐",
                label="选择人格"
            )
            chatbot = gr.Chatbot(label="天才女友", height=520, type="tuples")
            branch_btns = gr.Radio(choices=[], label="💭 可考虑方向：", interactive=True, visible=False)

            with gr.Row():
                msg = gr.Textbox(placeholder="你想什么呢？...", label="你的输入", scale=10, show_label=False, container=False)
                send_btn = gr.Button("📨", scale=1, variant="primary", min_width=50)

            clear = gr.Button("🧹 清空对话")
            history_state = gr.State([])

        with gr.Column(scale=1):
            balance_status = gr.Label(value=balance_status_value, label="开发者余额状态", elem_id="balance_status")

    msg.submit(
        user_input_fn,
        inputs=[msg, chatbot, history_state, branch_btns, personality_dropdown, balance_status],
        outputs=[msg, chatbot, branch_btns, history_state, balance_status]
    )

    send_btn.click(
        user_input_fn,
        inputs=[msg, chatbot, history_state, branch_btns, personality_dropdown, balance_status],
        outputs=[msg, chatbot, branch_btns, history_state, balance_status]
    )

    branch_btns.change(
        choose_branch_fn,
        inputs=[branch_btns, chatbot, history_state, branch_btns, personality_dropdown, balance_status],
        outputs=[chatbot, branch_btns, history_state, balance_status]
    )

    clear.click(clear_all, outputs=[chatbot, branch_btns, history_state, balance_status])

demo.launch(server_name="0.0.0.0", server_port=7860)
