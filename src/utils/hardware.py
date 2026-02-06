"""Hardware detection utilities for device selection."""

from typing import Dict, List, Optional
import torch


def get_available_devices() -> List[str]:
    """
    Get list of available compute devices.
    
    Returns:
        List of device strings (e.g., ['cpu', 'cuda:0', 'cuda:1'])
    """
    devices = ["cpu"]
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            devices.append(f"cuda:{i}")
    
    return devices


def get_device_info() -> Dict[str, dict]:
    """
    Get detailed information about available devices.
    
    Returns:
        Dictionary with device names as keys and info dicts as values.
    """
    info = {}
    
    # CPU info
    try:
        import os
        cpu_count = os.cpu_count() or 1
        info["cpu"] = {
            "name": "CPU",
            "type": "cpu",
            "cores": cpu_count,
        }
    except Exception:
        info["cpu"] = {"name": "CPU", "type": "cpu"}
    
    # GPU info
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            device_name = f"cuda:{i}"
            try:
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024 ** 3)
                info[device_name] = {
                    "name": props.name,
                    "type": "cuda",
                    "memory_gb": round(memory_gb, 1),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            except Exception:
                info[device_name] = {"name": f"CUDA Device {i}", "type": "cuda"}
    
    return info


def get_default_device() -> str:
    """Get the recommended default device. Prefers CUDA if available."""
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def get_device_display_name(device: str) -> str:
    """Get a human-readable display name for a device."""
    info = get_device_info()
    
    if device == "cpu":
        cpu_info = info.get("cpu", {})
        cores = cpu_info.get("cores", "")
        return f"CPU ({cores} cores)" if cores else "CPU"
    
    if device.startswith("cuda"):
        device_info = info.get(device, {})
        gpu_name = device_info.get("name", "Unknown GPU")
        memory = device_info.get("memory_gb", "")
        gpu_idx = device.split(":")[-1] if ":" in device else "0"
        return f"GPU {gpu_idx}: {gpu_name} ({memory} GB)" if memory else f"GPU {gpu_idx}: {gpu_name}"
    
    return device


def get_hardware_summary() -> Dict:
    """Get a complete summary of available hardware."""
    devices = get_available_devices()
    default = get_default_device()
    details = get_device_info()
    display_names = {dev: get_device_display_name(dev) for dev in devices}
    
    return {
        "devices": devices,
        "default": default,
        "details": details,
        "display_names": display_names,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
