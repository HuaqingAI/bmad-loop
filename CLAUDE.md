# CLAUDE.md

@AGENTS.md

## Claude Code specifics

- Never edit anything under `.claude/` — `.claude/skills/` mixes BMAD-installed skills with seeded forks of the module skills in `src/bmad_loop/data/skills/`. Edit the canonical copy, then run `uv run python scripts/seed_skills.py`.
