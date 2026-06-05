# Model Naming Convention

This document defines the naming conventions used for models developed and evaluated within the WARANEY research project.

---

# Objectives

The naming convention aims to:

- Ensure consistency across experiments.
- Improve reproducibility.
- Simplify model tracking and comparison.
- Support future model evolution.

---

# Model Categories

WARANEY research recognizes two categories of models:

## 1. External Models

Models developed by third parties and evaluated within WARANEY experiments.

Examples:

- followsci/bert-ai-text-detector
- Elia43/distilbert-ai-text-detector
- xnajoan/xna-ai-text-detector

These models retain their original names and should not be renamed.

---

## 2. WARANEY Models

Models developed within the WARANEY research project.

These models follow the WARANEY versioning scheme.

Format:

WARANEY-X.Y.Z

Where:

- X = major version
- Y = minor version
- Z = patch version

Example:

- WARANEY-0.1.0
- WARANEY-0.2.0
- WARANEY-1.0.0
- WARANEY-2.0.0

---

# Versioning Rules

## Major Version

Represents a significant architectural or methodological change.

Examples:

| Version | Description             |
| ------- | ----------------------- |
| 0.x     | Preliminary experiments |
| 1.x     | Initial research model  |
| 2.x     | Linguistic-aware model  |
| 3.x     | Hybrid model            |
| 4.x     | Explainable model       |

---

## Minor Version

Represents a research improvement within the same major version.

Examples:

- New backbone model
- Improved preprocessing
- Better training strategy
- Improved balancing method

Example:

- WARANEY-0.1.0
- WARANEY-0.2.0
- WARANEY-0.3.0

---

## Patch Version

Represents technical corrections that do not substantially change the methodology.

Examples:

- Configuration fixes
- Label mapping corrections
- Reproducibility fixes

Example:

- WARANEY-0.2.0
- WARANEY-0.2.1
- WARANEY-0.2.2

---

# Relationship Between Experiments and Models

Experiments and model versions are separate concepts.

Format:

PEX02-EXP001
PEX02-EXP002
PEX02-EXP003

A single experiment may produce a candidate model version.

Example:

PEX02-EXP001
→ WARANEY-0.1.0

PEX02-EXP002
→ WARANEY-0.2.0

PEX02-EXP003
→ WARANEY-0.3.0

---

# Checkpoint Naming

Training checkpoints should include experiment identifiers.

Format:

<experiment-id>-checkpoint

Example:

PEX02-EXP001-checkpoint

PEX02-EXP002-checkpoint

---

# Recommended Release Naming

Released models should use the official model version only.

Example:

WARANEY-0.1.0

WARANEY-0.2.0

WARANEY-1.0.0

Avoid names such as:

- best_model
- final_model
- final_model_v2
- best_model_fixed

---

# Current Roadmap

## WARANEY v0

Preliminary baseline detector.

Expected releases:

- WARANEY-0.1.0
- WARANEY-0.2.0
- WARANEY-0.3.0

## WARANEY v1

Research-grade detector.

## WARANEY v2

Linguistic-aware detector.

## WARANEY v3

Hybrid detector.
