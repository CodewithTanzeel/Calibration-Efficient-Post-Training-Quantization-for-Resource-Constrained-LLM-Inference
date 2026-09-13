# Project Template — SmoothQuant Calibration Project

## Which Template
- GitHub Project Template: Kanban (Basic) or Table (recommended: Table with grouped columns)
- Repo-level Project: Must be created manually (REST `repos/{repo}/projects` deprecated; `gh project create --title ...` requires `project` token scope)

## Recommended Project Board Template (manual setup)
1. Go to: `https://github.com/CodewithTanzeel/Calibration-Efficient-Post-Training-Quantization-for-Resource-Constrained-LLM-Inference/projects/new`
2. Title: `SmoothQuant Calibration Project`
3. Template: **Kanban (Basic)**
4. Columns: Backlog / In Progress / In Review / Blocked / Needs Verification / Done
5. Custom fields:
   - `Milestone Phase` (Single Select): Phase 1–8
   - `Priority` (Single Select): High / Medium / Low
   - `Category` (Multi-Select): Calibration / Quantization / Benchmark / Reasoning / Documentation
   - `Verification` (Single Select): Not Started / In Progress / Passed / Failed

## Automation
- PR merge with `Closes #X` / `Fixes #X` → close linked issue and move to Done
- Add `needs verification` label → move to Blocked

## Milestones Linked (8 phases)
- Phase 1 Foundation (due 2026-09-14) — #9, #10, #11
- Phase 2 Basic Implementation (due 2026-09-21) — #1, #2, #3
- Phase 3 Baseline Experiments (2026-09-28)
- Phase 4 Calibration Study (2026-10-05) — #4
- Phase 5 Optimization (2026-10-12) — #5
- Phase 6 Reasoning (2026-10-19) — #6
- Phase 7 Benchmarking (2026-10-26) — #7
- Phase 8 Final Analysis (2026-11-02) — #8, #12

## Blocker Note
`gh auth refresh -s project` needed for CLI creation; token has `gist`, `read:org`, `repo`, `workflow` only.
