---
trigger: glob
description: "Classical HCE eval constraints: SF11 alignment source of truth, scale, specialized endgames."
---

# Classical HCE Evaluation Constraints

**Source of truth for alignment:** `chess_engine/classical/HCE_SF11_GAP_AUDIT.md`  
Reference implementation: **`stockfish_11/`** (not SF 18 NNUE eval).  
Endgame oracle: `chess_engine/classical/ENDGAME_SF_VERIFICATION.md`.

---

## 1. Process (MUST)

* Before changing HCE terms, **read** the gap audit section for that component (mobility, threats, king safety, pawns, passed, space, initiative, scale, special EG).
* Prefer **structural alignment with SF11** at human-readable ~100cp scale — not bit-identical SF internal units unless the audit says otherwise.
* Do **not** reintroduce removed non-SF terms (e.g. discarded passed-pawn **material** scale) without audit + user-approved match testing.

---

## 2. Invariants (MUST)

* Tapered eval uses project phase limits / npm-style phase as documented in the audit and `constants.py`
* Specialized endgames (KXK, KPK bitbase, KRKP, KQKP, KBNK, …) live in `endgame.py` and short-circuit when hit; keep scores on **HCE known-win scale** (~10000), not NNUE display scale
* If eval returns extra data for search (e.g. pinned BBs), preserve the contract used by `search.py`

---

## 3. classical_old

* Old package is a **sync snapshot** for tournaments. Change **`classical/`** first; sync with `tools/sync_classical_old.py` only when intentional.
* **MUST NOT** maintain parallel evaluation markdown under `classical_old/`

---

## 4. After changes (recommend)

```bash
python -m unittest tests.test_phase_a_eval tests.test_endgame_conformance -v
# Elo: only with user OK — tools/tournament.py vs main_old.py
```
* Add the test to verify any new feature.