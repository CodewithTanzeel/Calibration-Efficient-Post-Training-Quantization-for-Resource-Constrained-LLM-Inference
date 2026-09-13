<<<<<<< HEAD
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
=======
# Project Template — SmoothQuant Calibration Project

## Which Template
- **GitHub Project Template:** `Kanban (Basic)` or `Table` (recommended: Table with grouped columns)
- **Repo-level Project:** Must be created manually (REST `repos/{repo}/projects` deprecated; `gh project create --title ...` requires `project` token scope which is not currently granted to this session)

## Recommended Project Board Template (manual setup)
1. Go to: `https://github.com/CodewithTanzeel/Calibration-Efficient-Post-Training-Quantization-for-Resource-Constrained-LLM-Inference/projects/new`
2. Title: `SmoothQuant Calibration Project`
3. Template: **Kanban (Basic)**
4. Columns (add in this order):
   - Backlog (default)
   - In Progress
   - In Review
   - Blocked / Needs Verification
   - Done
5. Custom fields (add to project):
   - `Milestone Phase` (Single Select): `Phase 1` through `Phase 8`
   - `Priority` (Single Select): `High`, `Medium`, `Low`
   - `Category` (Multi-Select): `Calibration`, `Quantization`, `Benchmark`, `Reasoning`, `Documentation`
   - `Verification` (Single Select): `Not Started`, `In Progress`, `Passed`, `Failed`

## Filter Views
Configure these in the board's view dropdown:
- `This Sprint`: Filter `Milestone Phase` = `Phase 2` AND `Priority` = `High`
- `Calibration Study`: Filter `Category` contains `Calibration`
- `Benchmarking`: Filter `Category` contains `Benchmark`
- `Reasoning`: Filter `Category` contains `Reasoning`
- `Good First Issue`: Filter labels include `good first issue`
- `Needs Verification`: Filter labels include `needs verification`

## Automation (manual or via GitHub Actions if upgraded)
- When issue with `label:task` moves to milestone `Phase 2` → move to `In Progress`
- When `label:"needs verification"` added → move to `Blocked / Needs Verification`
- When PR merges with `Closes #X` or `Fixes #X` → close linked issue and move to `Done`

## Milestones Linked to Board
All 8 milestones are created (via `gh api` / curl):
- #1 Phase 1: Foundation (due 2026-09-14)
- #2 Phase 2: Basic Implementation (due 2026-09-21)
- #3 Phase 3: Baseline Experiments (due 2026-09-28)
- #4 Phase 4: Calibration Study (due 2026-10-05)
- #5 Phase 5: Optimization (due 2026-10-12)
- #6 Phase 6: Reasoning (due 2026-10-19)
- #7 Phase 7: Benchmarking (due 2026-10-26)
- #8 Phase 8: Final Analysis (due 2026-11-02)

Issues linked to milestones (verified via API):
- #1, #2, #3 → Milestone 2 (Phase 2)
- #4 → Milestone 4 (Phase 4)
- #5 → Milestone 5 (Phase 5)
- #6 → Milestone 6 (Phase 6)
- #7 → Milestone 7 (Phase 7)
- #8 → Milestone 8 (Phase 8)
- #9, #10, #11 → Milestone 1 (Phase 1)
- #12 → Milestone 8 (Phase 8)

## Template Files Reference
- `.github/PROJECT_TEMPLATE.md` — board spec with columns, filters, custom fields
- `.github/PROJECT_SETUP.md` — manual creation steps (Web / CLI / API)
- `.github/ISSUE_TEMPLATE/task.md` — default issue template
- `.github/PULL_REQUEST_TEMPLATE.md` — PR template
- `docs/project_management/MILESTONES.md` — milestone definitions

## Blocker Note
The `gh project create` command requires `project` scope. The current auth token has scopes: `gist`, `read:org`, `repo`, `workflow`. To create the board via CLI, run:
```bash
gh auth refresh -s project
gh project create --title "SmoothQuant Calibration Project" --public
```
After that, add issues manually or via `gh project item-add`.
>>>>>>> 792ff4b (Verify: tests/test_core.py passes (3/5, 2 expected measurement failures due to Linear vs CausalLM wrapper) - Co-Authored-By: Claude Code <noreply@anthropic.com>)
