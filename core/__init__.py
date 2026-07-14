"""Trusted verification foundation for TabPVN.

The sound first-order-logic kernel: `FOLKernel` forward-chains a Horn rule base to a fixpoint and emits a
self-contained proof tree; `check_proof` independently re-verifies a proof with no kernel state.
"""

from core.kernel_fol import FOLKernel, check_proof, show

__all__ = ["FOLKernel", "check_proof", "show"]
