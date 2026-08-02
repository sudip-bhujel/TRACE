from importlib import import_module


_EXPORTS = {
    "PPOGradientBuffer": ("victim.capture.buffers", "PPOGradientBuffer"),
    "capture_a2c_gradients": ("victim.capture.a2c", "capture_a2c_gradients"),
    "capture_federated": ("victim.capture.federated", "capture_federated"),
    "capture_ppo_gradients": ("victim.capture.ppo", "capture_ppo_gradients"),
    "capture_and_save_streaming": (
        "victim.capture.streaming",
        "capture_and_save_streaming",
    ),
    "capture_uniform_per_scene": (
        "victim.capture.streaming",
        "capture_uniform_per_scene",
    ),
    "compute_a2c_gradients": ("victim.capture.a2c", "compute_a2c_gradients"),
    "compute_gradients": ("victim.capture.ppo", "compute_gradients"),
    "compute_ppo_gradients": ("victim.capture.ppo", "compute_ppo_gradients"),
    "create_hdf5_dataset": ("victim.capture.utils", "create_hdf5_dataset"),
    "flatten_gradients": ("victim.capture.utils", "flatten_gradients"),
    "load_model": ("victim.capture.utils", "load_model"),
    "print_file_info": ("victim.capture.utils", "print_file_info"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
