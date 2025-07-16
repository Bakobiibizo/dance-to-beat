# Audio/Video Processing Library – Refactoring Plan

This document tracks the high-level plan for cleaning up and consolidating the project.  
Each section lists the rationale, actions, and status.

---

## 1. FrameGenerator Consolidation *(IN PROGRESS)*
**Keep**: `src/video/rotation.py::FrameGenerator` – richer features & logging.  
**Remove**: `src/video/frame_generator.py` (simple duplicate).

Planned actions:
1. Confirm no modules import `src.video.frame_generator` (✓ verified via grep).
2. Deprecate or delete `src/video/frame_generator.py`.
3. Update docs & tests to reference the canonical class.
4. (Optional) Move the class into `src/video/frame_generator.py` and import it from `rotation.py` to separate concerns.

---

## 2. Video Generation Consolidation *(Queued)*
**Keep**: `rotation.py::create_rotating_video`  
**Remove**: duplicate in `rotate_to_beat_cli.py`

Actions:
1. Delete duplicate implementation.
2. Point CLI script to canonical function.
3. Extract `FrameState` to its own module if useful.

---

## 3. Audio Processing Refactor *(Queued)*
Consolidate shared onset-envelope logic into new `audio_utils.py`.

---

## 4. Color Management Consolidation *(Queued)*
Merge wheel cache/state logic.

---

## 5. Entry-Point Simplification *(Queued)*
Unify CLI entry point for end-users.

---

## 6. Configuration Improvements *(Queued)*
Organise parameters & add validation.

---

## 7. Documentation & Tests *(Queued)*
Add module docs and unit/integration tests.

---

*Last updated: 2025-07-16*
