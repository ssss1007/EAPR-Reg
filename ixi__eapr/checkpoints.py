"""Load IXI weights without silently accepting a different architecture."""


def load_registration_weights(model, checkpoint):
    state = checkpoint.get("model_state", checkpoint)
    # Spatial grids are deterministic and can be cached at runtime.
    state = {key: value for key, value in state.items() if not key.endswith(".grid")}
    incompatible = model.load_state_dict(state, strict=False)
    missing = [key for key in incompatible.missing_keys if not key.endswith(".grid")]
    unexpected = [key for key in incompatible.unexpected_keys if not key.endswith(".grid")]
    if missing or unexpected:
        raise RuntimeError(
            "IXI checkpoint architecture mismatch. Use the same APE/fusion/refinement "
            f"flags as training. Missing: {missing}; unexpected: {unexpected}"
        )
