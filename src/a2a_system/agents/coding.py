"""Coding agent: reasons about and writes code. It never runs anything."""

from a2a_system.agents.base import LLMAgent
from a2a_system.protocol.models import AgentSkill


class CodingAgent(LLMAgent):
    name = "Coding Agent"
    description = "Writes, explains, reviews and debugs code. It reasons only and cannot run code."
    skills = [AgentSkill(id="coding", name="Coding",
                         description="Write, explain, review or debug code in any language.",
                         tags=["code", "programming"], examples=["Write a Python function that reverses a list"])]
    system_prompt = (
        "You are an expert software engineer. Think through the problem step by step, then give correct, "
        "idiomatic code in fenced code blocks with a short explanation. You cannot execute code, so say "
        "which parts are untested assumptions."
    )
