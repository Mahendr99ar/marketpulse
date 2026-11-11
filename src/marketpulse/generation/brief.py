"""Write -> fact-check -> retry loop, built as a LangGraph graph.

    write ──> check ──(passed)──────────────> END  status = "passed"
                 └──(failed, retries left)──> write (with the fact-check feedback)
                 └──(failed, no retries)────> END  status = "needs_review"

A brief that still fails after the retries is not thrown away. It is marked
"needs_review" so a person checks it before it goes out (human in the loop).
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from marketpulse.config import Config
from marketpulse.generation.facts import DailyFacts
from marketpulse.generation.factcheck import FactCheckReport, fact_check
from marketpulse.generation.writers import Writer


class BriefState(TypedDict, total=False):
    facts: DailyFacts
    draft: str
    report: FactCheckReport
    attempts: int
    feedback: str | None
    history: list[dict]
    status: str


def build_graph(writer: Writer, cfg: Config, max_retries: int):
    def write(state: BriefState) -> BriefState:
        draft = writer.write(state["facts"], state.get("feedback"))
        return {"draft": draft, "attempts": state.get("attempts", 0) + 1}

    def check(state: BriefState) -> BriefState:
        report = fact_check(state["draft"], state["facts"], cfg)
        history = state.get("history", []) + [{
            "attempt": state["attempts"], "draft": state["draft"],
            "number_accuracy": report.number_accuracy, "passed": report.passed,
            "problems": report.feedback(),
        }]
        if report.passed:
            status = "passed"
        elif state["attempts"] > max_retries:
            status = "needs_review"
        else:
            status = "retry"
        return {"report": report, "feedback": report.feedback(), "history": history, "status": status}

    def route(state: BriefState) -> str:
        return "write" if state["status"] == "retry" else END

    g = StateGraph(BriefState)
    g.add_node("write", write)
    g.add_node("check", check)
    g.set_entry_point("write")
    g.add_edge("write", "check")
    g.add_conditional_edges("check", route, {"write": "write", END: END})
    return g.compile()


def generate_brief(facts: DailyFacts, writer: Writer, cfg: Config, max_retries: int | None = None) -> BriefState:
    if max_retries is None:
        max_retries = cfg["generation"]["max_retries"]
    graph = build_graph(writer, cfg, max_retries)
    return graph.invoke({"facts": facts, "attempts": 0, "feedback": None, "history": []})
