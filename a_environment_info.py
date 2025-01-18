# environment_info.py

import platform
import sys
import subprocess
import psutil

def get_gpu_info():
    """
    Versucht, Infos zur GPU abzurufen 
    """
    try:
        cmd = "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
        output = subprocess.check_output(cmd.split()).decode('utf-8').strip()
        lines = output.split('\n')
        gpus = []
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            name = parts[0]
            mem = parts[1]
            gpus.append((name, mem))
        return gpus
    except:
        return None

if __name__ == "__main__":
    print("=== Hardware- und Software-Umgebung ===\n")

    # CPU, RAM
    print("System/Plattform:", platform.platform())
    print("Prozessor:", platform.processor())

    # Gesammter RAM
    mem_info = psutil.virtual_memory()
    total_ram_gb = mem_info.total / (1024**3)
    print(f"Gesamt-RAM: {total_ram_gb:.2f} GB")

    # GPU-Infos (wenn NVIDIA verfügbar)
    gpu_info = get_gpu_info()
    if gpu_info is not None:
        print("\nNVIDIA GPU(s):")
        for idx, (name, mem) in enumerate(gpu_info):
            print(f"  GPU {idx}: {name}, Gesamtspeicher: {mem} MiB")
    else:
        print("\nKeine NVIDIA GPU erkannt oder nvidia-smi nicht verfügbar.")

    # Python-Version
    print(f"\nPython-Version: {sys.version}")

    # Paketversionen (pythae, sklearn)
    try:
        import sklearn
        print("scikit-learn-Version:", sklearn.__version__)
    except ImportError:
        print("scikit-learn nicht installiert oder Import-Fehler")


    print("\n=== Ende der System- und Umgebungsinformationen ===")
