---
description: "AI Assistant workflow guidelines: pre-task checking, post-task documentation maintenance, and user-driven testing execution."
trigger: always_on
---

# Workflow Guidelines

To maintain code cleanliness and documentation consistency across the project, all AI assistants must strictly adhere to the following workflow guidelines when performing development, refactoring, debugging, or documentation tasks.

---

## 1. Pre-Task: Documentation Check

* **Retrieve Documentation**: Before modifying any files or starting a new design phase, the AI assistant **MUST** read the **📂 Project Documentation Map (📂 專案文件導覽)** section in the project root's `README.md`.
* **Primary Objective**: Check if the module being modified already has an associated design specification, algorithm analysis report, or configuration guide. This prevents duplicate logic implementation or breaking established architectural invariants.

---

## 2. Post-Task: Documentation Maintenance

* **Update/Create Documents**: Upon implementing major features, refactoring components, or fixing system bugs, the AI assistant **MUST** proactively update existing technical documents or create new ones in the relevant directories (e.g., under `chess_engine/classical/` or `chess_engine/nnue/`).
* **Synchronize the Index**: If any technical document is added, deleted, or renamed, the AI assistant **MUST** update the **📂 Project Documentation Map (📂 專案文件導覽)** index in the root `README.md` to ensure it remains perfectly synchronized with the actual file structure.

---

## 3. Testing Policy: User-Led Execution

* **Strict Ban on Automatic Tests**: AI assistants are **strictly forbidden** from executing any test scripts, benchmarks, or commands in the terminal (such as perft tests, Elo matches, or unit tests) without explicit, real-time authorization from the user.
