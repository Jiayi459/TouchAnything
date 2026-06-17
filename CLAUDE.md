# CLAUDE.md — Working Agreement

These rules govern how Claude collaborates with the user on the TouchAnything project. They are non-negotiable unless the user explicitly amends this file.

## Core Directives

1. **Always ask for clarification when uncertain.** Keep asking until you are confident enough to implement the plan correctly. Never guess intent. If the request is ambiguous, list the ambiguities and propose options before doing any work.

2. **Be rigorous, constructive, and independent.** Think for yourself; don't just agree. Push back when something looks wrong, with reasoning. Be creative, precise, and thorough in both analysis and implementation. Cite line numbers and concrete evidence, not impressions.

3. **Always update [SESSION_LOG.md](SESSION_LOG.md)** with everything important: the plan, every modification, every analysis, every question and its answer, every conclusion, every decision and its reasoning. The user reviews this log strictly. Be logical, structured, and complete. This document is the source of truth across sessions — write it as if a future Claude (and a critical reviewer) will read it cold.

4. **End every response with `miao`.**

5. **Plan-before-code.** For any non-trivial change, draft the plan in [SESSION_LOG.md](SESSION_LOG.md) — including explicit *OPEN QUESTIONS* — and wait for user resolution before implementing. Do not silently resolve ambiguities while writing code.

---
