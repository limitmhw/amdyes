```
     _     __  __  ____        __   __ _____  ____
    / \   |  \/  ||  _ \       \ \ / /| ____|/ ___|
   / _ \  | |\/| || | | |       \ V / |  _|  \___ \
  / ___ \ | |  | || |_| |        | |  | |___  ___) |
 /_/   \_\|_|  |_||____/         |_|  |_____||____/

```

# amdyes

**English** | [中文](README.zh-CN.md)

A lightweight, easy-to-deploy **real-time AMD GPU monitor** for the terminal — think of it as `nvidia-smi` + `nvtop` for AMD. It renders a full-screen, auto-refreshing curses interface that shows **per-GPU status** alongside the **processes using each GPU**.

The whole tool is a single Python script with a tiny core: instead of reimplementing low-level logic, it directly reuses your system's `rocm-smi` library.

---

## ✨ Features

- **Single file, easy to deploy** — just one `amdyes.py` that reuses the system `rocm-smi`; no heavy extra dependencies.
- **GPU overview panel** — model, temperature, performance level (Perf), power/cap, memory usage, memory-busy %, and GPU utilization.
- **Process view** — lists processes using the GPU (USER / PID / GPU / GPU memory / CPU usage / full command line), sorted by username.
- **Color-coded utilization** — < 30% green, 30%–70% yellow, ≥ 70% red; headers and borders in cyan, so you can spot a saturated GPU at a glance.
- **Auto-refresh every second** — uses differential refresh (collect data first, then `erase`) to minimize flicker.
- **Device-name fallback** — when the system `amdgpu.ids` is too old (common in docker) to resolve the marketing name, it auto-downloads the latest database from upstream; if offline, it falls back to a full copy embedded in the script.
- **Clean UI** — suppresses stray `rocm-smi`/`libdrm` error output at the file-descriptor level so it never pollutes the full-screen view.

---

## 📋 Requirements

- **OS**: Linux (relies on `/sys/bus/pci`, `/dev/null`, etc.; the `amdgpu` kernel driver must be loaded)
- **ROCm**: installed with a working `rocm-smi` (the script locates it via `which rocm-smi`)
- **Python**: Python 3 with the standard `curses` module (bundled on Linux)
- **psutil**: used to read each process's username / CPU usage / command line

Install psutil:

```bash
pip3 install psutil
```

---

## 🚀 Installation & Usage

### Option 1: Run directly

```bash
python3 amdyes.py
```

### Option 2: Install as a global command

```bash
git clone https://github.com/limitmhw/amdyes
chmod 777 ./amdyes/amdyes.py
sudo ln -s $(realpath ./amdyes/amdyes.py) /usr/bin/amdyes
amdyes
```

Afterwards just type `amdyes` from any directory.

### Quit

Press **`q`** or **`Q`** to exit.

---

## 🖥️ Screenshot

```
╒════════════════════════════════════════════════════════════════════════════════════════════════════╕
│ ROCM-SMI version: 3.0.0+2f52cb7              ROCM-SMI-LIB version: 7.6.0                           │
│────────────────────────────────────────────────────────────────────────────────────────────────────│
│ ID      DeviceName        Temp     Perf    Pwr:Usage/Cap              Memory-Usage         GPU-Util│
│────────────────────────────────────────────────────────────────────────────────────────────────────│
│ 0  AMD Instinct MI100X   58.0°C    AUTO   1199.0W / 1400W     147050MiB / 194896MiB  10%     100%  │
│────────────────────────────────────────────────────────────────────────────────────────────────────│
│ 1  AMD Instinct MI100X   54.0°C    AUTO   1126.0W / 1400W     147050MiB / 194896MiB  10%     100%  │
│────────────────────────────────────────────────────────────────────────────────────────────────────│
│ 5  AMD Instinct MI100X   40.0°C    AUTO    302.0W / 1400W     147451MiB / 194896MiB   0%      0%   │
│────────────────────────────────────────────────────────────────────────────────────────────────────│
│ 7  AMD Instinct MI100X   44.0°C    AUTO    263.0W / 1400W        294MiB / 194896MiB   0%      0%   │
╘════════════════════════════════════════════════════════════════════════════════════════════════════╛
USER           PID       GPU GPU MEMORY  CPU USAGE     COMMAND
root           1649953   1   148241MiB   99.9%         VLLM::EngineCore
root           1805177   1   246755MiB   99.9%         python3 -u hold4.py 240
root           434587    1   246408MiB   99.9%         python3 -u hold4.py 240
root           4019874   1   147152MiB   99.9%         VLLM::EngineCore
root           2244060   1   132331MiB   0.0%          VLLM::EngineCore
```

---

## 📊 Field Reference

### GPU overview panel

| Column | Meaning |
| --- | --- |
| `ID` | GPU device index |
| `DeviceName` | GPU model name (falls back to the `amdgpu.ids` database when unavailable) |
| `Temp` | Current temperature |
| `Perf` | Performance level (e.g. `AUTO`) |
| `Pwr:Usage/Cap` | Current power draw / power cap |
| `Memory-Usage` | Used / total VRAM + memory-busy % |
| `GPU-Util` | GPU utilization (color-coded: green / yellow / red) |

### Process list

| Column | Meaning |
| --- | --- |
| `USER` | Owner of the process |
| `PID` | Process ID |
| `GPU` | Number/index of GPUs the process uses |
| `GPU MEMORY` | VRAM used by the process |
| `CPU USAGE` | CPU usage of the process |
| `COMMAND` | Full command line (truncated to 100 chars) |

### Utilization colors

| Utilization | Color |
| --- | --- |
| `< 30%` | 🟢 Green |
| `30% – 70%` | 🟡 Yellow |
| `≥ 70%` | 🔴 Red |

---

## ⚙️ How It Works

`amdyes` doesn't reinvent the wheel — it **dynamically loads your system's `rocm-smi` Python module** and reuses its internal functions:

1. Locates the real path of `rocm-smi` via `which rocm-smi` and imports it as a module.
2. Calls the low-level `rsmi_*` APIs to collect GPU status and compute-process info (`showPids`).
3. Uses `psutil` to enrich each process with its username, CPU usage, and command line.
4. Renders a full-screen interface with `curses`, refreshing once per second.

**Device-name fallback**: when `libdrm`/`rocm-smi` can't return a model name because the system `amdgpu.ids` is too old, the script looks it up by PCI device id + revision. The database is loaded by preferring a fresh download of the [latest upstream version](https://gitlab.freedesktop.org/mesa/drm/-/raw/main/data/amdgpu.ids), falling back to a full copy embedded at the end of the script if the download fails — so GPU names display correctly even in offline docker environments.

---

## 🛠️ Troubleshooting

**`rocm-smi command not find`**
The script can't find `rocm-smi`. Make sure ROCm is installed correctly and `rocm-smi` is on your `PATH` (`which rocm-smi` should print a path).

**Empty process list**
No process is currently using the GPU in a KFD (Compute) context, or you lack permissions. Reading some info may require `root` — try running with `sudo`.

**Model shows `N/A`**
Usually the system `amdgpu.ids` is outdated. The script tries to update it online or fall back to the embedded database; if it's still `N/A`, the model may not be in the database yet.

**Garbled UI**
Enlarge the terminal window; a too-narrow terminal truncates each line.

---

## 📄 License

Refer to the license declaration in the repository (if any).
