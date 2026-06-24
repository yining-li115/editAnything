"""Centralized Gemini model identifiers, one per role.
Kept separate because the three jobs (orchestration, parsing, judging) may
need different models and are tuned independently.
"""

ORCHESTRATOR_MODEL = "gemini-2.5-flash-lite"   # agent that decides which tools to call
PARSE_MODEL        = "gemini-2.5-flash-lite"   # intent parsing (prompt -> source/target/style)
JUDGE_MODEL        = "gemini-2.5-flash-lite"   # scores the edited video