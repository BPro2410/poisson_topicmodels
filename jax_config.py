"""
JAX configuration for Metal GPU support on macOS.
This module handles JAX setup with proper Metal GPU configuration.

Install metal jax: https://developer.apple.com/metal/jax/
"""

import os

# Configure BEFORE importing jax to avoid Metal errors
# Use CPU backend (most reliable with Metal issues)
# If you want to enable Metal GPU, change this to 'metal'
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Also set these for extra compatibility
os.environ["JAX_METAL_DEVICE_SYNC"] = "1"


def setup_jax_metal():
    """
    Setup JAX for Metal GPU support on macOS.

    This function configures JAX to work properly with Metal backend,
    avoiding the 'UNIMPLEMENTED: default_memory_space is not supported' error.

    NOTE: Environment variables should be set BEFORE importing JAX.
    """
    import jax

    try:
        devices = jax.devices()
        print(f"✓ JAX initialized successfully. Available devices: {devices}")

    except Exception as e:
        print(f"⚠ Warning: JAX initialization issue: {e}")


def use_cpu_only():
    """
    Force JAX to use CPU backend only.
    Use this if you experience GPU-related issues.
    NOTE: Call this BEFORE importing jax in your main script.
    """
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    print("✓ JAX will use CPU backend")


def use_metal_gpu():
    """
    Enable Metal GPU backend for JAX.
    NOTE: Call this BEFORE importing jax in your main script.
    """
    os.environ["JAX_PLATFORM_NAME"] = "metal"
    print("✓ JAX will use Metal GPU backend")


def check_jax_devices():
    """
    Print available JAX devices for debugging.
    """
    import jax

    devices = jax.devices()
    print(f"Available JAX devices: {devices}")
    print(f"Default device: {jax.default_device()}")


# Auto-setup on import
setup_jax_metal()
