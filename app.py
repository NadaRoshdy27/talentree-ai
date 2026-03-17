"""
app.py
TalentTree AI — FastAPI + Gradio on HuggingFace Spaces.
Gradio UI is at: /
FastAPI REST is at: /api
"""

import os
import logging
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import gradio as gr

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# LOGGING
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ==========================================
# LLM CONFIGS PER AGENT
# ==========================================

LLM_CONFIGS = {
    "PRICING":   {"max_new_tokens": 80,  "temperature": 0.1},
    "MARKETING": {"max_new_tokens": 220, "temperature": 0.5},
    "CAPTION":   {"max_new_tokens": 250, "temperature": 0.7},
    "PLANNER":   {"max_new_tokens": 300, "temperature": 0.3},
    "GENERAL":   {"max_new_tokens": 200, "temperature": 0.7},
}

llms = {
    agent: HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        max_new_tokens=config["max_new_tokens"],
        temperature=config["temperature"],
    )
    for agent, config in LLM_CONFIGS.items()
}

chat_models = {agent: ChatHuggingFace(llm=llm) for agent, llm in llms.items()}

# ==========================================
# IN-RAM SESSION MEMORY
# ==========================================

seller_memories: dict[str, list[dict]] = {}
MAX_MEMORY = 10


def add_to_memory(seller_id: str, role: str, message: str) -> None:
    if seller_id not in seller_memories:
        seller_memories[seller_id] = []
    seller_memories[seller_id].append({"role": role, "content": message})
    if len(seller_memories[seller_id]) > MAX_MEMORY:
        seller_memories[seller_id] = seller_memories[seller_id][-MAX_MEMORY:]


def get_memory(seller_id: str) -> str:
    history = seller_memories.get(seller_id, [])
    formatted = "Conversation History:\n"
    for msg in history[-4:]:
        formatted += f"{msg['role']}: {msg['content']}\n"
    return formatted


# ==========================================
# PROMPTS
# ==========================================

router_template = """
You are a strict classification system.

Classify the user request into ONE of these categories only:
PRICING
MARKETING
PLANNER
CAPTION
GENERAL

Rules:
- If user asks for a plan, strategy, roadmap, or 30 days → PLANNER
- If user asks for price or cost → PRICING
- If user asks for caption or social post text → CAPTION
- If user asks about marketing ideas, promotions, or events → MARKETING
- Otherwise → GENERAL

Return ONLY the category word. No explanation.

Request:
{user_input}
"""

pricing_template = """
You are a pricing assistant for Egyptian market.
Reply in English only.

Calculation rules:
- Total Cost = raw material cost + manufacturing cost
- Luxury audience: Price = Total Cost x 4 (minimum)
- Mass market audience: Price = Total Cost x 2
- All amounts are in Egyptian Pounds (LE)

Reply using ONLY this exact format, no extra text:

Cost: [total cost] LE
Price: [total cost x multiplier] LE
Margin: [multiplier]x
Reason: [max 10 words]

Brand: {brand_name} | Audience: {target_audience} | Category: {category}
Request: {user_input}
"""

marketing_template = """
You are a Creative Marketing Consultant for Talentree marketplace.
Reply in English only.
Brand: {brand_name} | Audience: {target_audience} | Category: {category}

Rules:
- If the request mentions a special event (Eid, Ramadan, Christmas, Black Friday, etc.), give event-specific ideas
- Maximum 3 marketing ideas
- Each idea under 25 words
- Include platform suggestion per idea (Instagram, TikTok, etc.)
- Be creative and specific to the brand and event
- No long explanations

User Request: {user_input}
"""

planner_template = """
You are a Professional Marketing Strategist for Talentree marketplace.
Reply in English only.

Brand: {brand_name} | Audience: {target_audience} | Category: {category}

Adapt your tone and ideas based on category and audience:
- Fashion & Accessories → style, trends, outfit inspiration, lookbooks
- Beauty & Skin Care → glow, transformation, before & after, skincare routines
- Handmade & Crafts → authentic, handcrafted story, community, behind the scenes

Also adapt based on audience:
- Luxury audience → premium, exclusive, high-end ideas
- Mass market → affordable, wide-reach, viral ideas

Create a 30-Day Marketing Plan. Use EXACTLY this format and STOP after Week 4:

**Week 1: [Theme]**
- [Campaign launch idea with platform]
- [Event or experience idea]

**Week 2: [Theme]**
- [Social media strategy with hashtag]
- [Digital marketing action]

**Week 3: [Theme]**
- [Giveaway or loyalty idea]
- [Content marketing action]

**Week 4: [Theme]**
- [Tech or innovative idea]
- [Real-world or community activation]

Rules:
- Mention brand name {brand_name} in at least 2 bullets
- Bold all campaign names and hashtags
- Maximum 20 words per bullet
- Each week must have exactly 2 bullets
- STOP after Week 4. Do not write Week 5 or beyond

Request: {user_input}
"""

caption_template = """
You are a Social Media Caption Expert.
Brand: {brand_name} | Tone: {tone} | Category: {category}

Write EXACTLY 3 captions. No more, no less.
Each caption MUST follow this exact format:
[2 emojis] [max 6 words]. #[hashtag]

Banned words: Unleash, Make a statement, Get ready, Radiant, Ablaze, Ignite

Example:
1. 💄✨ Bold red, pure confidence. #RedLip
2. 🌹💋 Your perfect shade awaits. #LuxuryLip
3. ✨💄 Red that speaks for itself. #GlowUp

Now write 3 captions for: {user_input}
"""

general_template = """
You are a Business Consultant for Talentree marketplace sellers.
Answer clearly and briefly in maximum 5 lines:
{user_input}
"""

TEMPLATES = {
    "PRICING":   pricing_template,
    "MARKETING": marketing_template,
    "PLANNER":   planner_template,
    "CAPTION":   caption_template,
}

HOLIDAY_KEYWORDS = [
    "EID", "RAMADAN", "CHRISTMAS", "NEW YEAR", "HOLIDAY",
    "BLACK FRIDAY", "PROMOTION", "SALE", "OFFER",
    "VALENTINE", "MOTHER'S DAY", "SUMMER", "WINTER",
    "BACK TO SCHOOL", "HALLOWEEN", "THANKSGIVING", "EVENT",
]

# ==========================================
# CORE CHAT LOGIC
# ==========================================


def talentree_chat(user_query: str, seller: dict) -> str:
    seller_id = seller["seller_id"]
    add_to_memory(seller_id, "User", user_query)

    router_chain = (
        ChatPromptTemplate.from_template(router_template)
        | chat_models["GENERAL"]
        | StrOutputParser()
    )
    raw = router_chain.invoke({"user_input": user_query}).strip().upper()

    if any(word in user_query.upper() for word in HOLIDAY_KEYWORDS):
        category = "MARKETING"
    elif "PRICE" in raw or "PRICING" in raw:
        category = "PRICING"
    elif "MARKET" in raw:
        category = "MARKETING"
    elif "PLAN" in raw:
        category = "PLANNER"
    elif "CAPTION" in raw:
        category = "CAPTION"
    else:
        category = "GENERAL"

    logger.info("seller_id=%s  routed_to=%s", seller_id, category)

    selected_template = TEMPLATES.get(category, general_template)
    memory_ctx = get_memory(seller_id)
    combined = selected_template + f"\n\n{memory_ctx}\nCurrent Request:\n{{user_input}}"

    final_chain = (
        ChatPromptTemplate.from_template(combined)
        | chat_models.get(category, chat_models["GENERAL"])
        | StrOutputParser()
    )

    response = final_chain.invoke({**seller, "user_input": user_query})
    add_to_memory(seller_id, "Assistant", response)
    return response


# ==========================================
# FASTAPI APP
# ==========================================

fastapi_app = FastAPI(
    title="TalentTree AI Chat API",
    description="Multi-agent AI assistant — Pricing · Marketing · Planner · Caption · General",
    version="1.0.0",
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    seller_id: str = Field(..., description="Unique seller session ID")
    brand_name: str = Field(..., description="Seller's brand name")
    category: str = Field(..., description="Product category")
    target_audience: str = Field(..., description="Target audience description")
    tone: str = Field(default="Professional", description="Caption tone")
    message: str = Field(..., description="The seller's message")


class ChatResponse(BaseModel):
    response: str


@fastapi_app.get("/api")
def root():
    return {"status": "ok", "service": "TalentTree AI Chat API"}


@fastapi_app.get("/api/health")
def health():
    return {"status": "healthy"}


@fastapi_app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    seller = {
        "seller_id": request.seller_id,
        "brand_name": request.brand_name,
        "category": request.category,
        "target_audience": request.target_audience,
        "tone": request.tone,
    }
    try:
        ai_response = talentree_chat(request.message, seller)
        return ChatResponse(response=ai_response)
    except Exception as e:
        logger.exception("AI error for seller_id=%s", request.seller_id)
        raise HTTPException(status_code=500, detail=str(e))


@fastapi_app.delete("/api/memory/{seller_id}")
def clear_memory(seller_id: str):
    if seller_id in seller_memories:
        del seller_memories[seller_id]
        return {"message": f"Memory cleared for: {seller_id}"}
    return {"message": f"No memory found for: {seller_id}"}


# ==========================================
# GRADIO UI
# ==========================================

def gradio_chat(message, history, seller_id, brand_name, category, target_audience, tone):
    if not brand_name.strip():
        return "Please enter your Brand Name first."

    seller = {
        "seller_id": seller_id.strip() or "guest",
        "brand_name": brand_name.strip(),
        "category": category.strip(),
        "target_audience": target_audience.strip(),
        "tone": tone.strip(),
    }

    try:
        return talentree_chat(message, seller)
    except Exception as e:
        return f"Error: {str(e)}"


with gr.Blocks(title="TalentTree AI") as gradio_ui:

    gr.Markdown("# 🌳 TalentTree AI — Seller Assistant")
    gr.Markdown("Your AI advisor for **Pricing · Marketing · Planning · Captions**")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Your Profile")
            seller_id_box = gr.Textbox(label="Seller ID", value="guest")
            brand_name_box = gr.Textbox(label="Brand Name *", placeholder="e.g. Glow Essence")
            category_box = gr.Dropdown(
                label="Category",
                choices=["Beauty & Skin Care", "Fashion & Accessories", "Handmade & Crafts"],
                value="Beauty & Skin Care",
            )
            audience_box = gr.Textbox(label="Target Audience", value="Women 25-40")
            tone_box = gr.Dropdown(
                label="Tone",
                choices=["Professional", "Elegant", "Luxury", "Playful", "Bold", "Casual", "Inspiring", "Minimalist"],
                value="Professional",
            )

        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=gradio_chat,
                additional_inputs=[seller_id_box, brand_name_box, category_box, audience_box, tone_box],
                examples=[
                    ["How should I price my product if raw material is 100 and manufacturing is 50?"],
                    ["Give me 2 marketing ideas for my brand"],
                    ["Create a 30-day marketing plan"],
                    ["Write 3 captions for my new lipstick"],
                ],
            )

# ==========================================
# MOUNT GRADIO INTO FASTAPI
# ==========================================

app = gr.mount_gradio_app(fastapi_app, gradio_ui, path="/")
