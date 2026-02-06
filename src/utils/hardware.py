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


def get_system_ram_gb() -> float:
    """Get total system RAM in GB."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        # Fallback if psutil not available
        try:
            import os
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        kb = int(line.split()[1])
                        return round(kb / (1024 ** 2), 1)
        except Exception:
            pass
    return 0.0


def get_device_info() -> Dict[str, dict]:
    """
    Get detailed information about available devices.
    
    Returns:
        Dictionary with device names as keys and info dicts as values.
    """
    info = {}
    
    # CPU info with RAM
    try:
        import os
        cpu_count = os.cpu_count() or 1
        ram_gb = get_system_ram_gb()
        info["cpu"] = {
            "name": "CPU",
            "type": "cpu",
            "cores": cpu_count,
            "ram_gb": ram_gb,
        }
    except Exception:
        info["cpu"] = {"name": "CPU", "type": "cpu"}
    
    # GPU info with VRAM
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
                    "vram_gb": round(memory_gb, 1),
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
        ram = cpu_info.get("ram_gb", 0)
        if cores and ram:
            return f"CPU ({cores} cores, {ram} GB RAM)"
        elif cores:
            return f"CPU ({cores} cores)"
        return "CPU"
    
    if device.startswith("cuda"):
        device_info = info.get(device, {})
        gpu_name = device_info.get("name", "Unknown GPU")
        vram = device_info.get("vram_gb", "")
        gpu_idx = device.split(":")[-1] if ":" in device else "0"
        return f"GPU {gpu_idx}: {gpu_name} ({vram} GB VRAM)" if vram else f"GPU {gpu_idx}: {gpu_name}"
    
    return device


def get_hardware_summary() -> Dict:
    """Get a complete summary of available hardware."""
    devices = get_available_devices()
    default = get_default_device()
    details = get_device_info()
    display_names = {dev: get_device_display_name(dev) for dev in devices}
    
    # Extract RAM and VRAM for easy access
    ram_gb = details.get("cpu", {}).get("ram_gb", 0)
    vram_gb = 0
    if "cuda:0" in details:
        vram_gb = details["cuda:0"].get("vram_gb", 0)
    
    return {
        "devices": devices,
        "default": default,
        "details": details,
        "display_names": display_names,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "ram_gb": ram_gb,
        "vram_gb": vram_gb,
    }
