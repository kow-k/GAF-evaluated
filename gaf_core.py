#!/usr/bin/env python3
"""
gaf_core.py --- compatibility alias for gaf_engine.py.

The GAF arithmetic now lives in gaf_engine.py, which exposes BOTH:

  * the high-level gaf_core API --- M, compose, gamma_bigram --- where gamma is
    derived from the pair (geometry or corpus counts), and
  * the matrix-free primitives --- M_with_lambda, compose_simpler,
    gamma_bigram_by_freq --- which take an explicit lambda / raw counts.

This module re-exports everything, so `import gaf_core` keeps working and is
byte-for-byte identical in behaviour to `import gaf_engine`.

Author: Kow Kuroda (Kyorin University) & Claude (Anthropic)
License: MIT
"""
from gaf_engine import *          # noqa: F401,F403
from gaf_engine import __all__    # noqa: F401
