"""Semilinear-wave example: an inverse problem with an exact, fixed blind subspace.

A 1D defocusing wave equation is solved from its initial displacement field, recorded by a
few interior receivers over a short time window. Finite speed of propagation makes the
*causal shadow* -- the nodes the signal cannot reach within the window -- exactly invisible
to the forward map at every state, for both the linear and the nonlinear operator. That
gives a genuine kernel of a nonlinear operator, fixed by geometry rather than by the field.

Modules (imported on demand, never here, so a script pays only for what it uses):

``solver``        leapfrog forward map, differentiable end to end.
``blind``         the causal-shadow blind subspace, read from the solver's Jacobian.
``regularizers``  the misspecified smoothness (Tikhonov) legacy regularizer.
``priors``        the rough-field truth prior and truth/data synthesis.
``archive``       discrepancy-principle strength, legacy MAP archive, blind-block freeze.
``coverage``      central-interval coverage per projected direction, plus the oracle gate.
``nets``          the deployed UNet DDPM prior: construction, loading, unconditional draws.
``dps``           diffusion posterior sampling on the wave operator.
``ddpm``          a small MLP DDPM over the field, used by the epsilon control.
``config``        the experiment configuration.
``pipeline``      end-to-end run behind the epsilon control.
"""
