"""Evaluation metrics for the internal benchmark (see ../BENCHMARK.md).

Each metric is an independent, lazily-loaded backend so a missing one (e.g. no
Gemini key) never blocks the others.
"""
