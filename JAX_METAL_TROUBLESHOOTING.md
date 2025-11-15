# JAX Metal GPU Support Troubleshooting Guide

## Error: `jaxlib._jax.XlaRuntimeError: UNIMPLEMENTED: default_memory_space is not supported`

This error occurs when JAX Metal tries to use memory space features that aren't fully implemented in the Metal backend.

---

## ✅ Solutions (in order of preference)

### **Solution 1: Use the new `jax_config.py` (RECOMMENDED)**

The `jax_config.py` file has been added to your project. It's automatically imported at the start of `run_topicmodels.py`.

**What it does:**
- Configures JAX to work with Metal GPU while avoiding problematic operations
- Includes fallback to CPU if Metal setup fails
- Provides utility functions for debugging

**No action needed** - it's already set up in your code!

---

### **Solution 2: Force CPU Backend (Safest)**

If you still experience issues, edit the `jax_config.py` file and uncomment the CPU line:

```python
def setup_jax_metal():
    # Option 1: Use CPU backend (most reliable but slower)
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # ← Uncomment this line
```

Or add this to the top of your script before any JAX imports:

```python
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
```

---

### **Solution 3: Update JAX to Newer Version**

The version in your `requirements.txt` (0.4.35) is from early 2024. Newer versions have better Metal support.

```bash
pip install --upgrade jax jaxlib
```

For Metal GPU specifically:
```bash
pip install --upgrade jax jaxlib -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

---

### **Solution 4: Use Conditional Device Placement**

If you want to keep GPU but need more control, you can use CPU for specific operations:

```python
import jax
import os

# Try Metal, fall back to CPU if issues occur
try:
    os.environ['JAX_PLATFORMS'] = 'metal'
    _ = jax.numpy.zeros(1)  # Test if Metal works
    print("Using Metal GPU")
except:
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    print("Falling back to CPU")
```

---

## 🔍 Debugging: Check Your Setup

Run this in a Python terminal:

```python
import jax
print(f"Available devices: {jax.devices()}")
print(f"Default device: {jax.default_device()}")
print(f"JAX version: {jax.__version__}")
```

---

## 📊 Performance Comparison

| Backend | Speed | Reliability | JAX Version |
|---------|-------|-------------|-------------|
| Metal (GPU) | Fast | Medium | 0.4.29+ |
| CPU | Slow | Very High | Any |

---

## 🛠 Additional Tips

1. **For development**: Use CPU backend (more stable)
2. **For production**: Use Metal GPU after confirming it works
3. **For debugging**: Add `from jax_config import check_jax_devices` and run `check_jax_devices()` to verify setup
4. **Environment variables**: You can set JAX backend via terminal before running:
   ```bash
   export JAX_PLATFORMS=metal
   python run_topicmodels.py
   ```

---

## 🚀 Next Steps

1. Try running your script with the new `jax_config.py` setup
2. If it still fails, uncomment the CPU line in `jax_config.py`
3. If that works, test upgrading JAX version
4. Report any remaining issues with the error message

Good luck! 🎉
