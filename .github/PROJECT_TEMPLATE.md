# Project Template: SmoothQuant Calibration Project

## Board Setup
Type: Kanban (Table / Board view recommended)
Views: Backlog, In Progress, In Review, Done, Blocked

## Columns
1. **Backlog** — All open issues (research, calibration, optimization, reasoning, benchmark)
2. **In Progress (Milestone Active)** — Issues tied to current milestone target
3. **In Review** — Pull requests or verification tasks
4. **Done** — Completed and verified
5. **Blocked / Needs Verification** — Waiting for external validation (tests, benchmarks)

## Milestone Tracking
Link each milestone to board columns via issue labels:
- `milestone: phase-1` through `milestone: phase-8`
- `priority: high` / `priority: medium` / `priority: low`

## Filter Presets
- **This Sprint:** `milestone: phase-2` + `priority: high`
- **Calibration Study:** `label:calibration`
- **Benchmarking:** `label:benchmark` + `label:cpu`
- **Reasoning:** `label:reasoning`
- **Good First Issue:** `label:"good first issue"`
- **Needs Verification:** `label:"needs verification"`

## Custom Fields
| Field | Type | Values |
|-------|------|--------|
| Milestone Phase | Single Select | Phase 1–8 |
| Priority | Single Select | High / Medium / Low |
| Category | Multi-Select | Calibration / Benchmark / Reasoning / Quantization / Documentation |
| Verification | Single Select | Not Started / In Progress / Passed / Failed |

## Automation Rules
- When milestone `Phase 2` starts → move `label:task` items to In Progress
- When `label:"needs verification"` is added → move to Blocked
- When `label:"good first issue"` + `label:documentation` → add to Backlog with `good first issue` tag
- On PR merge → auto-close linked issue (if title contains `Fixes #` or `Closes #`)

## Project Settings
- Visibility: Public (research transparency)
- Readme: Link to `CLAUDE.md` and `docs/project_management/MILESTONES.md`
- Default Issue Template: `task.md`
- Default PR Template: `.github/PULL_REQUEST_TEMPLATE.md`
