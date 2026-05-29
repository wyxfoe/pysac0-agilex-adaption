"""NemoDiT — Flow-Matching DiT policy adapted for AgileX / Mobile-Aloha real robots.

This package is intentionally minimal at the top level. The two entry points are
``train_agilex.py`` and ``inference_agilex.py``; both are designed to be invoked
as scripts (``python train_agilex.py ...``) and rely on each other being on the
same import path, so this ``__init__`` deliberately does no eager imports.
"""
