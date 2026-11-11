"""Brief writers. Each one turns DailyFacts (plus optional fact-check feedback) into text.

    template  deterministic sentences, no LLM. The baseline every LLM must beat on
              readability without losing accuracy. Also used in demo mode and tests.
    ollama    local open model through Ollama (free, runs on your laptop)
    groq      hosted open model through the Groq API (free tier)
    hf        your fine-tuned model from finetune/, loaded with transformers
"""

from __future__ import annotations

import json
from typing import Protocol

from marketpulse.config import Config
from marketpulse.generation.facts import DailyFacts, fmt_change, fmt_level

SYSTEM_PROMPT = """You write the morning market brief for a sales and trading desk.

Rules:
1. Use only numbers that appear in FACTS. Copy them exactly as written, including sign and unit.
2. Never calculate anything new: no sums, differences, averages, or conversions.
3. Start with one sentence on the market regime and its probability.
4. Then cover each item in unusual_moves: what moved, by how much, and the likely driver.
   Name a driver only if one of its headlines clearly explains the move. Otherwise write
   that the news does not show a clear driver.
5. End with one sentence on other_markets.
6. 120 to 180 words. Plain sentences, no bullet points, no forecasts, no advice."""


def user_prompt(facts: DailyFacts, feedback: str | None = None) -> str:
    msg = "FACTS:\n" + json.dumps(facts.prompt_view(), indent=2)
    if feedback:
        msg += ("\n\nYour previous draft failed the fact check. Fix these problems and "
                "rewrite the whole brief:\n" + feedback)
    return msg


class Writer(Protocol):
    name: str

    def write(self, facts: DailyFacts, feedback: str | None = None) -> str: ...


class TemplateWriter:
    name = "template"

    def write(self, facts: DailyFacts, feedback: str | None = None) -> str:
        lines = []
        regime = f"Markets are in a {facts.regime} regime ({facts.regime_probability}% probability)"
        if facts.previous_regime:
            regime += f", switching from {facts.previous_regime}"
        lines.append(regime + ".")
        if not facts.flagged:
            lines.append("No market made an unusual move today.")
        for m in facts.flagged:
            verb = "rose" if m.change > 0 else "fell"
            s = f"{m.name} {verb} {fmt_change(m).lstrip('+-')} to {fmt_level(m)}, a {abs(m.z or 0):.1f} standard deviation move for this regime."
            heads = facts.news.get(m.instrument_id, [])
            s += f" Top related headline: \"{heads[0].title}\"." if heads else " The news does not show a clear driver."
            lines.append(s)
        others = [f"{m.name} {fmt_change(m)}" for m in facts.moves if not m.flagged][:4]
        if others:
            lines.append("Elsewhere: " + ", ".join(others) + ".")
        return " ".join(lines)


class ChatWriter:
    """Any LangChain chat model."""

    def __init__(self, name: str, chat_model):
        self.name = name
        self.chat = chat_model

    def write(self, facts: DailyFacts, feedback: str | None = None) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        msgs = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt(facts, feedback))]
        return self.chat.invoke(msgs).content.strip()


class HFWriter:
    """A local transformers model, e.g. the LoRA model merged by finetune/train_lora.py."""

    name = "hf"

    def __init__(self, model_path: str, temperature: float = 0.2, max_new_tokens: int = 350):
        from transformers import pipeline

        self.pipe = pipeline("text-generation", model=model_path, device_map="auto")
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def write(self, facts: DailyFacts, feedback: str | None = None) -> str:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(facts, feedback)}]
        out = self.pipe(msgs, max_new_tokens=self.max_new_tokens,
                        do_sample=self.temperature > 0, temperature=max(self.temperature, 1e-3))
        return out[0]["generated_text"][-1]["content"].strip()


def get_writer(cfg: Config, backend: str | None = None) -> Writer:
    g = cfg["generation"]
    backend = backend or g["backend"]
    if backend == "template":
        return TemplateWriter()
    if backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatWriter("ollama", ChatOllama(model=g["ollama_model"], temperature=g["temperature"]))
    if backend == "groq":
        from langchain_groq import ChatGroq

        return ChatWriter("groq", ChatGroq(model=g["groq_model"], temperature=g["temperature"]))
    if backend == "hf":
        return HFWriter(g["hf_model_path"], g["temperature"])
    raise ValueError(f"Unknown generation backend: {backend}")
