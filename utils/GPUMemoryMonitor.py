import pynvml
import time
import threading
import sys

try:
    _nvml_initialized = False
except:
    _nvml_initialized = False


def get_gpu_memory_mb():
    global _nvml_initialized
    try:
        if not _nvml_initialized:
            pynvml.nvmlInit()
            _nvml_initialized = True
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024 / 1024
    except Exception as e:
        print(f"GPUMemoryMonitor: Failed to get GPU memory: {e}", file=sys.stderr)
        raise


class GPUMemoryMonitor:
    def __init__(self, device_index=0, polling_interval=0.01):
        global _nvml_initialized

        if not isinstance(globals().get('_nvml_initialized', False), bool):
            globals()['_nvml_initialized'] = False
        _nvml_initialized = globals()['_nvml_initialized']

        self.device_index = device_index
        self.polling_interval = polling_interval
        self.running = False
        self.max_memory = 0.0
        self.initial_memory = 0.0
        self.thread = None
        self.handle = None

        try:
            if not _nvml_initialized:
                pynvml.nvmlInit()
                _nvml_initialized = True
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as e:
            msg = (
                f"Failed to initialize GPUMemoryMonitor: {e}\n"
                f"Possible fixes:\n"
                f"  - Install: pip install nvidia-ml-py\n"
                f"  - Check: run 'nvidia-smi' in terminal\n"
                f"  - If in Docker: use --gpus flag"
            )
            print(msg, file=sys.stderr)
            raise RuntimeError(msg) from None

    def start(self):
        if self.running:
            return
        try:
            self.initial_memory = get_gpu_memory_mb()
            self.max_memory = self.initial_memory
            self.running = True
            self.thread = threading.Thread(target=self._monitor, daemon=True)
            self.thread.start()
        except Exception as e:
            self.running = False
            raise RuntimeError(f"Failed to start monitoring: {e}") from None

    def _monitor(self):
        try:
            while self.running:
                curr = get_gpu_memory_mb()
                if curr > self.max_memory:
                    self.max_memory = curr
                time.sleep(self.polling_interval)
        except Exception as e:
            print(f"GPUMemoryMonitor: Monitor thread error: {e}", file=sys.stderr)
            self.running = False

    def stop(self):
        if not self.running:
            return 0.0
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        return self.max_memory - self.initial_memory

    @staticmethod
    def is_available():
        try:
            pynvml.nvmlInit()
            pynvml.nvmlDeviceGetHandleByIndex(0)
            return True
        except:
            return False

    @staticmethod
    def diagnostic():
        print("GPUMemoryMonitor Diagnostic")
        print("-" * 40)
        try:
            pynvml.nvmlInit()
            print("NVML initialized successfully")
        except Exception as e:
            print(f"NVML init failed: {e}")
            return
        try:
            count = pynvml.nvmlDeviceGetCount()
            print(f"Number of GPU devices: {count}")
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                print(f"GPU {i}: {name} | Used memory: {mem.used / 1024**3:.2f} GB")
        except Exception as e:
            print(f"Failed to query GPU info: {e}")