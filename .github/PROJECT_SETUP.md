# GitHub Project Template — How to Initialize

This file describes how to turn the repo into a tracked GitHub Project (Board / Table view) if the REST API is unavailable. Use these steps manually or via `gh` where supported.

## Option A: Create via GitHub Web UI (Recommended)
1. Go to: `https://github.com/CodewithTanzeel/Calibration-Efficient-Post-Training-Quantization-for-Resource-Constrained-LLM-Inference/projects/new`
2. Name: **SmoothQuant Calibration Project**
3. Description: Copy from `.github/PROJECT_TEMPLATE.md`
4. Template: Select **Kanban (basic)**
5. Visibility: Public
6. Save → Project created

## Option B: Link Existing Milestones to Project (After Creation)
1. Open the project board
2. Add automation: "Auto-archive done items after 30 days"
3. Add columns: Backlog, In Progress, In Review, Done, Blocked / Verification
4. Add custom field: Milestone Phase (single select: Phase 1–8)
5. Import all 12 issues into Backlog (auto-linked to milestone numbers)

## Option C: Use `gh project` CLI (if available in newer versions)
```bash
gh project create "SmoothQuant Calibration Project" --public --description "Calibration-efficient post-training quantization"
```

## Template Applied To This Repo
- `.github/PROJECT_TEMPLATE.md` — board structure and automation rules
- `.github/ISSUE_TEMPLATE/` — feature, bug, task templates for consistent issue creation
- `docs/project_management/MILESTONES.md` — 8-phase milestone tracking (Foundation through Final Analysis)
- `CLAUDE.md` — full project documentation with research questions and contributions
- 12 categorized GitHub issues with proper labels (calibration, benchmark, reasoning, quantization, cpu, etc.)
- 15 labels including `priority: high/medium/low`, `needs verification`, `good first issue`
- 8 milestones (created via API where possible) with due dates

## Verification Checklist
- [ ] Project board exists at repo URL
- [ ] Issues #1–#12 visible on board with correct labels
- [ ] Milestones 1–8 linked (due dates visible)
- [ ] Columns Backlog / In Progress / Done / Blocked configured
- [ ] Custom field Milestone Phase added to board
- [ ] Auto-archive / automation rules activated
- [ ] `.github/PROJECT_TEMPLATE.md` kept in sync with actual board config
