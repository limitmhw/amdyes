#!/usr/bin/env python
import importlib.util
import sys
import os
import contextlib
from pathlib import Path
import curses
import time
import shutil

rocm_smi_path = shutil.which('rocm-smi')
if rocm_smi_path:
    real_rocm_smi_path = os.path.realpath(rocm_smi_path)
    sys.path.append(os.path.dirname(real_rocm_smi_path))
else:
    print("rocm-smi command not find")
    exit()

spec = importlib.util.spec_from_file_location('rocm_smi_module', Path(real_rocm_smi_path).resolve())
rocm_smi_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rocm_smi_module)
globals().update({name: obj for name, obj in rocm_smi_module.__dict__.items()})

@contextlib.contextmanager
def suppress_output():
    """临时把 stdout/stderr 重定向到 /dev/null。

    rocm-smi 库在 libdrm 调用失败时会直接打印错误（如 "Error when calling
    libdrm"），在 curses 全屏模式下这些打印会穿透并污染界面。在文件描述符层面
    重定向可同时屏蔽 Python 与底层 C 库的输出，渲染在退出本上下文后进行，不受影响。
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(devnull)
        os.close(saved_out)
        os.close(saved_err)

def get_process_info(pid):
    try:
        import psutil
        process = psutil.Process(pid)
        username = process.username()
        cpu_usage = process.cpu_percent(interval=0.1)
        cmdline = process.cmdline()
        return username, cpu_usage, cmdline
    except:
        return "", "", []
    
def myInitializeRsmi():
    initializeRsmi()
    """ initializes rocmsmi if the amdgpu driver is initialized
    """
    global rocmsmi
    # Initialize rsmiBindings
    rocmsmi = initRsmiBindings(silent=PRINT_JSON)
    # Check if amdgpu is initialized before initializing rsmi
    if driverInitialized() is True:
        ret_init = rocmsmi.rsmi_init(0)
        if ret_init != 0:
            logging.error('ROCm SMI returned %s (the expected value is 0)', ret_init)
            exit(ret_init)
    else:
        logging.error('Driver not initialized (amdgpu not found in modules)')
        exit(0)

def showPids():
    """ Show Information for PIDs created in a KFD (Compute) context """
    verbose = "details"
    dataArray = []
    if verbose == "details":
        dataArray.append(['PID', 'PROCESS NAME', 'GPU', 'VRAM USED', 'SDMA USED', 'CU OCCUPANCY'])
    else:
        dataArray.append(['PID', 'PROCESS NAME', 'GPU(s)', 'VRAM USED', 'SDMA USED', 'CU OCCUPANCY'])

    pidList = getPidList()
    if not pidList:
        return []
    dv_indices = c_void_p()
    num_devices = c_uint32()
    proc = rsmi_process_info_t()
    for pid in pidList:
        gpuNumber = 'UNKNOWN'
        vramUsage = 'UNKNOWN'
        sdmaUsage = 'UNKNOWN'
        cuOccupancy = 'UNKNOWN'
        cuOccupancyInvalid = 0xFFFFFFFF
        dv_indices = (c_uint32 * num_devices.value)()
        ret = rocmsmi.rsmi_compute_process_gpus_get(int(pid), None, byref(num_devices))
        if rsmi_ret_ok(ret, metric='get_gpu_compute_process'):
            dv_indices = (c_uint32 * num_devices.value)()
            ret = rocmsmi.rsmi_compute_process_gpus_get(int(pid), dv_indices, byref(num_devices))
            if rsmi_ret_ok(ret, metric='get_gpu_compute_process'):
                gpuNumber = str(num_devices.value)
            else:
                logging.debug('Unable to fetch GPU number by PID')
        if verbose == "details":
            for dv_ind in dv_indices:
                ret = rocmsmi.rsmi_compute_process_info_by_device_get(int(pid), dv_ind, byref(proc))
                if rsmi_ret_ok(ret, metric='get_compute_process_info_by_pid'):
                    vramUsage = proc.vram_usage
                    sdmaUsage = proc.sdma_usage
                    if proc.cu_occupancy != cuOccupancyInvalid:
                        cuOccupancy = proc.cu_occupancy
                else:
                    logging.debug('Unable to fetch process info by PID')
                dataArray.append([pid, getProcessName(pid), str(gpuNumber), str(vramUsage), str(sdmaUsage), str(cuOccupancy)])
        else:
            ret = rocmsmi.rsmi_compute_process_info_by_pid_get(int(pid), byref(proc))
            if rsmi_ret_ok(ret, metric='get_compute_process_info_by_pid'):
                vramUsage = proc.vram_usage
                sdmaUsage = proc.sdma_usage
                if proc.cu_occupancy != cuOccupancyInvalid:
                    cuOccupancy = proc.cu_occupancy
            else:
                logging.debug('Unable to fetch process info by PID')
            dataArray.append([pid, getProcessName(pid), str(gpuNumber), str(vramUsage), str(sdmaUsage), str(cuOccupancy)])
    return dataArray
    

# 设备名数据库：当 libdrm 因系统 amdgpu.ids 过旧而无法解析市场名时作为回退，
# 保证在缺少最新型号数据库的 docker 环境里也能显示 GPU 名称。
# 加载策略：优先临时下载上游最新版，下载失败时回退到本文件末尾内嵌的完整副本。
AMDGPU_IDS_URL = "https://gitlab.freedesktop.org/mesa/drm/-/raw/main/data/amdgpu.ids"
_AMDGPU_IDS = None

def _parse_amdgpu_ids(text):
    """把 amdgpu.ids 文本解析为 {(device_id, revision): name}。"""
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 3:
            continue
        did, rev = parts[0].upper(), parts[1].upper()
        result[(did, rev)] = ','.join(parts[2:]).strip()
    return result

def load_amdgpu_ids():
    """加载设备名数据库（仅一次）：优先临时下载最新版，下载失败则用内嵌副本。"""
    global _AMDGPU_IDS
    if _AMDGPU_IDS is None:
        text = None
        try:
            import urllib.request
            with urllib.request.urlopen(AMDGPU_IDS_URL, timeout=3) as resp:
                text = resp.read().decode('utf-8', errors='ignore')
        except Exception:
            text = None
        _AMDGPU_IDS = _parse_amdgpu_ids(text if text else AMDGPU_IDS_DATA)
    return _AMDGPU_IDS

def device_name_from_ids(device):
    """通过 sysfs 的 PCI id 查设备名数据库取设备名，失败返回 None。"""
    try:
        bdf = getBus(device, silent=True)
        if not bdf:
            return None
        base = '/sys/bus/pci/devices/%s' % bdf.lower()
        with open(base + '/device') as f:
            did = f.read().strip()[2:].upper().zfill(4)
        try:
            with open(base + '/revision') as f:
                rev = f.read().strip()[2:].upper().zfill(2)
        except OSError:
            rev = ''
        ids = load_amdgpu_ids()
        name = ids.get((did, rev))
        if name is None:
            # 退一步：忽略 revision，匹配相同 device id 的任意条目
            for (d, _r), n in ids.items():
                if d == did:
                    name = n
                    break
        return name
    except Exception:
        return None

def get_device_name(device):
    """优先用原始 rsmi/libdrm 方式取名；取不到时回退到脚本自带的 amdgpu.ids。"""
    name = getDeviceName(device, silent=True)
    if name and name != 'N/A':
        return name
    fallback = device_name_from_ids(device)
    return fallback if fallback else (name or 'N/A')

# GPU 利用率配色：低=绿 中=黄 高=红；表头/边框=青
COLOR_LOW, COLOR_MID, COLOR_HIGH, COLOR_HEADER = 1, 2, 3, 4

def util_color(util):
    """根据利用率数值返回对应的 curses 颜色属性。"""
    try:
        value = float(str(util).strip().rstrip('%'))
    except (TypeError, ValueError):
        return curses.color_pair(0)
    if value >= 70:
        return curses.color_pair(COLOR_HIGH)
    if value >= 30:
        return curses.color_pair(COLOR_MID)
    return curses.color_pair(COLOR_LOW)

def get_data(height, width):
    deviceList = listDevices()
    output = []
    
    version = rsmi_version_t()
    status = rocmsmi.rsmi_version_get(byref(version))
    version_string = ""
    if status == 0:
        version_string = "%u.%u.%u" % (version.major, version.minor, version.patch)
        
    degree_sign = u'\N{DEGREE SIGN}'
    gpu_info_length = 0
    for device in deviceList:
        memoryUse = c_uint64()
        rocmsmi.rsmi_dev_memory_busy_percent_get(device, byref(memoryUse))
        for mem in ['vram']:
            mem = mem.upper()
            memInfo = getMemInfo(device, mem)
            temp_val = str(getTemp(device, getTemperatureLabel(deviceList), True))
            if temp_val != 'N/A':
                temp_val += degree_sign + 'C'
            power_dict = getPower(device)
            powerVal = 'N/A'
            if (power_dict['ret'] == rsmi_status_t.RSMI_STATUS_SUCCESS and 
                power_dict['power_type'] != 'INVALID_POWER_TYPE'):
                if power_dict['power'] != 0:
                    powerVal = power_dict['power'] + power_dict['unit']
        gpu_use = getGpuUse(device, True)
        gpu_info =  "│{}{}{}{}{}{}{}│".format(str(device).center(4),
                                            str(get_device_name(device)).center(20),
                                            str(temp_val).center(10),
                                            str(getPerfLevel(device)).center(7),
                                            (str(powerVal) + " / " + str(int(getMaxPower(device, True))) + "W" ).center(20),
                                            ((str(memInfo[0] // 1024 // 1024) + "MiB").rjust(9) + 
                                            " / " +
                                            (str(memInfo[1] // 1024 // 1024) + "MiB ").rjust(9) + 
                                            (str(memoryUse.value) + "%").rjust(4)).center(30), 
                                            (str(gpu_use) + "%").center(9))
        if device == 0:
            gpu_info_length = len(gpu_info)
        output.append((gpu_info, util_color(gpu_use)))
        output.append(("│" + ("─" * (gpu_info_length - 2)) + "│", curses.color_pair(COLOR_HEADER)))

    header_attr = curses.color_pair(COLOR_HEADER)
    head =  [
        ("│ ROCM-SMI version: {}              ROCM-SMI-LIB version: {}".format(__version__, version_string).ljust(gpu_info_length-1) + "│", header_attr),
        ("│" + ("─" * (gpu_info_length - 2)) + "│", header_attr),
        ("│{}{}{}{}{}{}{}│".format("ID".center(4),
                                "DeviceName".center(20),
                                "Temp".center(10),
                                "Perf".center(7),
                                "Pwr:Usage/Cap".center(20),
                                "Memory-Usage".center(31),
                                "GPU-Util".center(8)
                            ), header_attr | curses.A_BOLD),
    ("│" + ("─" * (gpu_info_length - 2)) + "│", header_attr),
    ]
    head.extend(output)
    output = head
    output.insert(0, ("╒" + ("═" * (gpu_info_length - 2)) + "╕", header_attr))
    output = output[:-1]
    output.append(("╘" + ("═" * (gpu_info_length - 2)) + "╛", header_attr))
    pid_info = showPids()
    ps_info = []
    for item in pid_info[1:]:
        pid = int(item[0])
        username, cpu_usage, cmdline = get_process_info(pid)
        item.insert(0, username)
        item.append(str(cpu_usage) + "%")
        item.append(str(" ".join(cmdline)))
        ps_info.append(item)

    output.append(("{}{}{}{}{}{}".format(
                            "USER".ljust(15),
                            "PID".ljust(10),
                            "GPU".ljust(4),
                            "GPU MEMORY".ljust(12),
                            "CPU USAGE".ljust(14),
                            "COMMAND".ljust(10)
                            ), curses.color_pair(COLOR_HEADER) | curses.A_BOLD))

    sorted_ps_info = sorted(ps_info, key=lambda x: x[0])
    for item in sorted_ps_info:
        ps_str = "{}{}{}{}{}{}".format(
            item[0].ljust(15),
            item[1].ljust(10),
            item[3].ljust(4),
            str(str(int(item[4]) // 1024 // 1024) + "MiB").ljust(12),
            item[-2].ljust(14),
            item[-1][:100].ljust(20),
        )
        output.append((ps_str, curses.A_NORMAL))
    return output

def init_colors():
    """初始化颜色对，终端不支持颜色时静默跳过。"""
    try:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(COLOR_LOW, curses.COLOR_GREEN, -1)
        curses.init_pair(COLOR_MID, curses.COLOR_YELLOW, -1)
        curses.init_pair(COLOR_HIGH, curses.COLOR_RED, -1)
        curses.init_pair(COLOR_HEADER, curses.COLOR_CYAN, -1)
    except curses.error:
        pass

def main(stdscr):
    stdscr.clear()
    curses.curs_set(0)
    init_colors()
    height, width = stdscr.getmaxyx()
    update_interval = 1
    stdscr.timeout(update_interval * 1000)
    while True:
        try:
            # 先采集数据再清屏，避免清屏后等待数据期间出现空白闪烁
            # 采集期间屏蔽 rocm-smi/libdrm 的杂散打印，防止污染界面
            with suppress_output():
                output = get_data(height, width)
            # erase 配合差异刷新只更新变化的字符，比 clear 整屏重绘更不易闪烁
            stdscr.erase()
            for i, (line, attr) in enumerate(output):
                if i < height - 1:
                    stdscr.addstr(i + 1, 0, line, attr)
            stdscr.noutrefresh()
            curses.doupdate()
        except:
            pass
        # 在刷新间隔内等待按键，按 Q/q 退出
        if stdscr.getch() in (ord('q'), ord('Q')):
            break

# 内嵌的完整 amdgpu.ids 副本（来自 freedesktop mesa/drm），作为无法联网下载时的回退数据源。
AMDGPU_IDS_DATA = r"""
# List of AMDGPU IDs
#
# Syntax:
# device_id,	revision_id,	product_name        <-- single tab after comma

1.0.0
1114,	C2,	AMD Radeon 860M Graphics
1114,	C3,	AMD Radeon 840M Graphics
1114,	D2,	AMD Radeon 860M Graphics
1114,	D3,	AMD Radeon 840M Graphics
1114,	E2,	AMD Radeon 860M Graphics
1114,	E4,	AMD Radeon 860M Graphics
1114,	E5,	AMD Radeon 840M Graphics
1114,	E9,	AMD Radeon 860M Graphics
1114,	EA,	AMD Radeon 840M Graphics
1114,	ED,	AMD Radeon 860M Graphics
1114,	EE,	AMD Radeon 840M Graphics
1114,	F2,	AMD Radeon 860M Graphics
1114,	F3,	AMD Radeon 840M Graphics
1114,	F9,	AMD Radeon 860M Graphics
1114,	FA,	AMD Radeon 840M Graphics
1114,	FC,	AMD Radeon 860M Graphics
1114,	FD,	AMD Radeon 840M Graphics
1309,	00,	AMD Radeon R7 Graphics
130A,	00,	AMD Radeon R6 Graphics
130B,	00,	AMD Radeon R4 Graphics
130C,	00,	AMD Radeon R7 Graphics
130D,	00,	AMD Radeon R6 Graphics
130E,	00,	AMD Radeon R5 Graphics
130F,	00,	AMD Radeon R7 Graphics
130F,	D4,	AMD Radeon R7 Graphics
130F,	D5,	AMD Radeon R7 Graphics
130F,	D6,	AMD Radeon R7 Graphics
130F,	D7,	AMD Radeon R7 Graphics
1313,	00,	AMD Radeon R7 Graphics
1313,	D4,	AMD Radeon R7 Graphics
1313,	D5,	AMD Radeon R7 Graphics
1313,	D6,	AMD Radeon R7 Graphics
1315,	00,	AMD Radeon R5 Graphics
1315,	D4,	AMD Radeon R5 Graphics
1315,	D5,	AMD Radeon R5 Graphics
1315,	D6,	AMD Radeon R5 Graphics
1315,	D7,	AMD Radeon R5 Graphics
1316,	00,	AMD Radeon R5 Graphics
1318,	00,	AMD Radeon R5 Graphics
131B,	00,	AMD Radeon R4 Graphics
131C,	00,	AMD Radeon R7 Graphics
131D,	00,	AMD Radeon R6 Graphics
1435,	AE,	AMD Custom GPU 0932
1506,	C1,	AMD Radeon 610M
1506,	C2,	AMD Radeon 610M
1506,	C3,	AMD Radeon 610M
1506,	C4,	AMD Radeon 610M
150E,	C1,	AMD Radeon 890M Graphics
150E,	C4,	AMD Radeon 880M Graphics
150E,	C5,	AMD Radeon 890M Graphics
150E,	C6,	AMD Radeon 890M Graphics
150E,	C7,	AMD Radeon 890M Graphics
150E,	D1,	AMD Radeon 890M Graphics
150E,	D2,	AMD Radeon 880M Graphics
150E,	D3,	AMD Radeon 890M Graphics
150E,	E1,	AMD Radeon 890M Graphics
150E,	E3,	AMD Radeon 890M Graphics
150E,	E4,	AMD Radeon 890M Graphics
150E,	F1,	AMD Radeon 890M Graphics
150E,	F3,	AMD Radeon 890M Graphics
1586,	C1,	AMD Radeon 8060S Graphics
1586,	C2,	AMD Radeon 8050S Graphics
1586,	C3,	AMD Radeon 8060S Graphics
1586,	C4,	AMD Radeon 8050S Graphics
1586,	C6,	AMD Radeon 8060S Graphics
1586,	D1,	AMD Radeon 8060S Graphics
1586,	D2,	AMD Radeon 8050S Graphics
1586,	D4,	AMD Radeon 8050S Graphics
1586,	D5,	AMD Radeon 8040S Graphics
15BF,	00,	AMD Radeon 780M Graphics
15BF,	01,	AMD Radeon 760M Graphics
15BF,	02,	AMD Radeon 780M Graphics
15BF,	03,	AMD Radeon 760M Graphics
15BF,	05,	AMD Radeon 760M Graphics
15BF,	06,	AMD Radeon 780M Graphics
15BF,	07,	AMD Radeon 740M Graphics
15BF,	08,	AMD Radeon 740M Graphics
15BF,	C1,	AMD Radeon 780M Graphics
15BF,	C2,	AMD Radeon 780M Graphics
15BF,	C3,	AMD Radeon 760M Graphics
15BF,	C4,	AMD Radeon 780M Graphics
15BF,	C5,	AMD Radeon 740M Graphics
15BF,	C6,	AMD Radeon 780M Graphics
15BF,	C7,	AMD Radeon 780M Graphics
15BF,	C8,	AMD Radeon 760M Graphics
15BF,	C9,	AMD Radeon 780M Graphics
15BF,	CA,	AMD Radeon 740M Graphics
15BF,	CB,	AMD Radeon 760M Graphics
15BF,	CC,	AMD Radeon 740M Graphics
15BF,	CD,	AMD Radeon 760M Graphics
15BF,	CE,	AMD Radeon 740M Graphics
15BF,	CF,	AMD Radeon 780M Graphics
15BF,	D0,	AMD Radeon 780M Graphics
15BF,	D1,	AMD Radeon 780M Graphics
15BF,	D2,	AMD Radeon 780M Graphics
15BF,	D3,	AMD Radeon 780M Graphics
15BF,	D4,	AMD Radeon 780M Graphics
15BF,	D5,	AMD Radeon 760M Graphics
15BF,	D6,	AMD Radeon 760M Graphics
15BF,	D7,	AMD Radeon 780M Graphics
15BF,	D8,	AMD Radeon 740M Graphics
15BF,	D9,	AMD Radeon 780M Graphics
15BF,	DA,	AMD Radeon 780M Graphics
15BF,	DB,	AMD Radeon 760M Graphics
15BF,	DC,	AMD Radeon 760M Graphics
15BF,	DD,	AMD Radeon 780M Graphics
15BF,	DE,	AMD Radeon 740M Graphics
15BF,	DF,	AMD Radeon 760M Graphics
15BF,	F0,	AMD Radeon 760M Graphics
15C8,	C1,	AMD Radeon 740M Graphics
15C8,	C2,	AMD Radeon 740M Graphics
15C8,	C3,	AMD Radeon 740M Graphics
15C8,	C4,	AMD Radeon 740M Graphics
15C8,	C5,	AMD Radeon 740M Graphics
15C8,	C6,	AMD Radeon 740M Graphics
15C8,	C7,	AMD Radeon 740M Graphics
15C8,	C8,	AMD Radeon 740M Graphics
15C8,	D1,	AMD Radeon 740M Graphics
15C8,	D2,	AMD Radeon 740M Graphics
15C8,	D3,	AMD Radeon 740M Graphics
15C8,	D4,	AMD Radeon 740M Graphics
15C8,	D5,	AMD Radeon 740M Graphics
15C8,	D6,	AMD Radeon 740M Graphics
15C8,	D7,	AMD Radeon 740M Graphics
15C8,	D8,	AMD Radeon 740M Graphics
15D8,	00,	AMD Radeon RX Vega 8 Graphics WS
15D8,	91,	AMD Radeon Vega 3 Graphics
15D8,	91,	AMD Ryzen Embedded R1606G with Radeon Vega Gfx
15D8,	92,	AMD Radeon Vega 3 Graphics
15D8,	92,	AMD Ryzen Embedded R1505G with Radeon Vega Gfx
15D8,	93,	AMD Radeon Vega 1 Graphics
15D8,	A1,	AMD Radeon Vega 10 Graphics
15D8,	A2,	AMD Radeon Vega 8 Graphics
15D8,	A3,	AMD Radeon Vega 6 Graphics
15D8,	A4,	AMD Radeon Vega 3 Graphics
15D8,	B1,	AMD Radeon Vega 10 Graphics
15D8,	B2,	AMD Radeon Vega 8 Graphics
15D8,	B3,	AMD Radeon Vega 6 Graphics
15D8,	B4,	AMD Radeon Vega 3 Graphics
15D8,	C1,	AMD Radeon Vega 10 Graphics
15D8,	C2,	AMD Radeon Vega 8 Graphics
15D8,	C3,	AMD Radeon Vega 6 Graphics
15D8,	C4,	AMD Radeon Vega 3 Graphics
15D8,	C5,	AMD Radeon Vega 3 Graphics
15D8,	C8,	AMD Radeon Vega 11 Graphics
15D8,	C9,	AMD Radeon Vega 8 Graphics
15D8,	CA,	AMD Radeon Vega 11 Graphics
15D8,	CB,	AMD Radeon Vega 8 Graphics
15D8,	CC,	AMD Radeon Vega 3 Graphics
15D8,	CE,	AMD Radeon Vega 3 Graphics
15D8,	CF,	AMD Ryzen Embedded R1305G with Radeon Vega Gfx
15D8,	D1,	AMD Radeon Vega 10 Graphics
15D8,	D2,	AMD Radeon Vega 8 Graphics
15D8,	D3,	AMD Radeon Vega 6 Graphics
15D8,	D4,	AMD Radeon Vega 3 Graphics
15D8,	D8,	AMD Radeon Vega 11 Graphics
15D8,	D9,	AMD Radeon Vega 8 Graphics
15D8,	DA,	AMD Radeon Vega 11 Graphics
15D8,	DB,	AMD Radeon Vega 3 Graphics
15D8,	DB,	AMD Radeon Vega 8 Graphics
15D8,	DC,	AMD Radeon Vega 3 Graphics
15D8,	DD,	AMD Radeon Vega 3 Graphics
15D8,	DE,	AMD Radeon Vega 3 Graphics
15D8,	DF,	AMD Radeon Vega 3 Graphics
15D8,	E3,	AMD Radeon Vega 3 Graphics
15D8,	E4,	AMD Ryzen Embedded R1102G with Radeon Vega Gfx
15DD,	81,	AMD Ryzen Embedded V1807B with Radeon Vega Gfx
15DD,	82,	AMD Ryzen Embedded V1756B with Radeon Vega Gfx
15DD,	83,	AMD Ryzen Embedded V1605B with Radeon Vega Gfx
15DD,	84,	AMD Radeon Vega 6 Graphics
15DD,	85,	AMD Ryzen Embedded V1202B with Radeon Vega Gfx
15DD,	86,	AMD Radeon Vega 11 Graphics
15DD,	88,	AMD Radeon Vega 8 Graphics
15DD,	C1,	AMD Radeon Vega 11 Graphics
15DD,	C2,	AMD Radeon Vega 8 Graphics
15DD,	C3,	AMD Radeon Vega 3 / 10 Graphics
15DD,	C4,	AMD Radeon Vega 8 Graphics
15DD,	C5,	AMD Radeon Vega 3 Graphics
15DD,	C6,	AMD Radeon Vega 11 Graphics
15DD,	C8,	AMD Radeon Vega 8 Graphics
15DD,	C9,	AMD Radeon Vega 11 Graphics
15DD,	CA,	AMD Radeon Vega 8 Graphics
15DD,	CB,	AMD Radeon Vega 3 Graphics
15DD,	CC,	AMD Radeon Vega 6 Graphics
15DD,	CE,	AMD Radeon Vega 3 Graphics
15DD,	CF,	AMD Radeon Vega 3 Graphics
15DD,	D0,	AMD Radeon Vega 10 Graphics
15DD,	D1,	AMD Radeon Vega 8 Graphics
15DD,	D3,	AMD Radeon Vega 11 Graphics
15DD,	D5,	AMD Radeon Vega 8 Graphics
15DD,	D6,	AMD Radeon Vega 11 Graphics
15DD,	D7,	AMD Radeon Vega 8 Graphics
15DD,	D8,	AMD Radeon Vega 3 Graphics
15DD,	D9,	AMD Radeon Vega 6 Graphics
15DD,	E1,	AMD Radeon Vega 3 Graphics
15DD,	E2,	AMD Radeon Vega 3 Graphics
163F,	AE,	AMD Custom GPU 0405
163F,	E1,	AMD Custom GPU 0405
164E,	D8,	AMD Radeon 610M
164E,	D9,	AMD Radeon 610M
164E,	DA,	AMD Radeon 610M
164E,	DB,	AMD Radeon 610M
164E,	DC,	AMD Radeon 610M
1681,	06,	AMD Radeon 680M
1681,	07,	AMD Radeon 660M
1681,	0A,	AMD Radeon 680M
1681,	0B,	AMD Radeon 660M
1681,	C7,	AMD Radeon 680M
1681,	C8,	AMD Radeon 680M
1681,	C9,	AMD Radeon 660M
1900,	01,	AMD Radeon 780M Graphics
1900,	02,	AMD Radeon 760M Graphics
1900,	03,	AMD Radeon 780M Graphics
1900,	04,	AMD Radeon 760M Graphics
1900,	05,	AMD Radeon 780M Graphics
1900,	06,	AMD Radeon 780M Graphics
1900,	07,	AMD Radeon 760M Graphics
1900,	B0,	AMD Radeon 780M Graphics
1900,	B1,	AMD Radeon 780M Graphics
1900,	B2,	AMD Radeon 780M Graphics
1900,	B3,	AMD Radeon 780M Graphics
1900,	B4,	AMD Radeon 780M Graphics
1900,	B5,	AMD Radeon 780M Graphics
1900,	B6,	AMD Radeon 780M Graphics
1900,	B7,	AMD Radeon 760M Graphics
1900,	B8,	AMD Radeon 760M Graphics
1900,	B9,	AMD Radeon 780M Graphics
1900,	BA,	AMD Radeon 780M Graphics
1900,	BB,	AMD Radeon 780M Graphics
1900,	C0,	AMD Radeon 780M Graphics
1900,	C1,	AMD Radeon 760M Graphics
1900,	C2,	AMD Radeon 780M Graphics
1900,	C3,	AMD Radeon 760M Graphics
1900,	C4,	AMD Radeon 780M Graphics
1900,	C5,	AMD Radeon 780M Graphics
1900,	C6,	AMD Radeon 760M Graphics
1900,	C7,	AMD Radeon 780M Graphics
1900,	C8,	AMD Radeon 760M Graphics
1900,	C9,	AMD Radeon 780M Graphics
1900,	CA,	AMD Radeon 760M Graphics
1900,	CB,	AMD Radeon 780M Graphics
1900,	CC,	AMD Radeon 780M Graphics
1900,	CD,	AMD Radeon 760M Graphics
1900,	CE,	AMD Radeon 780M Graphics
1900,	CF,	AMD Radeon 760M Graphics
1900,	D0,	AMD Radeon 780M Graphics
1900,	D1,	AMD Radeon 760M Graphics
1900,	D2,	AMD Radeon 780M Graphics
1900,	D3,	AMD Radeon 760M Graphics
1900,	D4,	AMD Radeon 780M Graphics
1900,	D5,	AMD Radeon 780M Graphics
1900,	D6,	AMD Radeon 760M Graphics
1900,	D7,	AMD Radeon 780M Graphics
1900,	D8,	AMD Radeon 760M Graphics
1900,	D9,	AMD Radeon 780M Graphics
1900,	DA,	AMD Radeon 760M Graphics
1900,	DB,	AMD Radeon 780M Graphics
1900,	DC,	AMD Radeon 780M Graphics
1900,	DD,	AMD Radeon 760M Graphics
1900,	DE,	AMD Radeon 780M Graphics
1900,	DF,	AMD Radeon 760M Graphics
1900,	F0,	AMD Radeon 780M Graphics
1900,	F1,	AMD Radeon 780M Graphics
1900,	F2,	AMD Radeon 780M Graphics
1901,	C1,	AMD Radeon 740M Graphics
1901,	C2,	AMD Radeon 740M Graphics
1901,	C3,	AMD Radeon 740M Graphics
1901,	C6,	AMD Radeon 740M Graphics
1901,	C7,	AMD Radeon 740M Graphics
1901,	C8,	AMD Radeon 740M Graphics
1901,	C9,	AMD Radeon 740M Graphics
1901,	CA,	AMD Radeon 740M Graphics
1901,	D1,	AMD Radeon 740M Graphics
1901,	D2,	AMD Radeon 740M Graphics
1901,	D3,	AMD Radeon 740M Graphics
1901,	D4,	AMD Radeon 740M Graphics
1901,	D5,	AMD Radeon 740M Graphics
1901,	D6,	AMD Radeon 740M Graphics
1901,	D7,	AMD Radeon 740M Graphics
1901,	D8,	AMD Radeon 740M Graphics
1902,	C0,	AMD Radeon 840M Graphics
1902,	C1,	AMD Radeon 840M Graphics
1902,	C2,	AMD Radeon 820M Graphics
1902,	C3,	AMD Radeon 840M Graphics
1902,	C6,	AMD Radeon 820M Graphics
1902,	C7,	AMD Radeon 840M Graphics
1902,	C8,	AMD Radeon 840M Graphics
1902,	C9,	AMD Radeon 820M Graphics
1902,	CA,	AMD Radeon 840M Graphics
1902,	D1,	AMD Radeon 840M Graphics
1902,	D3,	AMD Radeon 840M Graphics
1902,	D7,	AMD Radeon 840M Graphics
1902,	D8,	AMD Radeon 840M Graphics
6600,	00,	AMD Radeon HD 8600 / 8700M
6600,	81,	AMD Radeon R7 M370
6601,	00,	AMD Radeon HD 8500M / 8700M
6604,	00,	AMD Radeon R7 M265 Series
6604,	81,	AMD Radeon R7 M350
6605,	00,	AMD Radeon R7 M260 Series
6605,	81,	AMD Radeon R7 M340
6606,	00,	AMD Radeon HD 8790M
6607,	00,	AMD Radeon R5 M240
6608,	00,	AMD FirePro W2100
6610,	00,	AMD Radeon R7 200 Series
6610,	81,	AMD Radeon R7 350
6610,	83,	AMD Radeon R5 340
6610,	87,	AMD Radeon R7 200 Series
6611,	00,	AMD Radeon R7 200 Series
6611,	87,	AMD Radeon R7 200 Series
6613,	00,	AMD Radeon R7 200 Series
6617,	00,	AMD Radeon R7 240 Series
6617,	87,	AMD Radeon R7 200 Series
6617,	C7,	AMD Radeon R7 240 Series
6640,	00,	AMD Radeon HD 8950
6640,	80,	AMD Radeon R9 M380
6646,	00,	AMD Radeon R9 M280X
6646,	80,	AMD Radeon R9 M385
6646,	80,	AMD Radeon R9 M470X
6647,	00,	AMD Radeon R9 M200X Series
6647,	80,	AMD Radeon R9 M380
6649,	00,	AMD FirePro W5100
6658,	00,	AMD Radeon R7 200 Series
665C,	00,	AMD Radeon HD 7700 Series
665D,	00,	AMD Radeon R7 200 Series
665F,	81,	AMD Radeon R7 360 Series
6660,	00,	AMD Radeon HD 8600M Series
6660,	81,	AMD Radeon R5 M335
6660,	83,	AMD Radeon R5 M330
6663,	00,	AMD Radeon HD 8500M Series
6663,	83,	AMD Radeon R5 M320
6664,	00,	AMD Radeon R5 M200 Series
6665,	00,	AMD Radeon R5 M230 Series
6665,	83,	AMD Radeon R5 M320
6665,	C3,	AMD Radeon R5 M435
6666,	00,	AMD Radeon R5 M200 Series
6667,	00,	AMD Radeon R5 M200 Series
666F,	00,	AMD Radeon HD 8500M
66A1,	02,	AMD Instinct MI60 / MI50
66A1,	06,	AMD Radeon Pro VII
66AF,	C1,	AMD Radeon VII
6780,	00,	AMD FirePro W9000
6784,	00,	ATI FirePro V (FireGL V) Graphics Adapter
6788,	00,	ATI FirePro V (FireGL V) Graphics Adapter
678A,	00,	AMD FirePro W8000
6798,	00,	AMD Radeon R9 200 / HD 7900 Series
6799,	00,	AMD Radeon HD 7900 Series
679A,	00,	AMD Radeon HD 7900 Series
679B,	00,	AMD Radeon HD 7900 Series
679E,	00,	AMD Radeon HD 7800 Series
67A0,	00,	AMD Radeon FirePro W9100
67A1,	00,	AMD Radeon FirePro W8100
67B0,	00,	AMD Radeon R9 200 Series
67B0,	80,	AMD Radeon R9 390 Series
67B1,	00,	AMD Radeon R9 200 Series
67B1,	80,	AMD Radeon R9 390 Series
67B9,	00,	AMD Radeon R9 200 Series
67C0,	00,	AMD Radeon Pro WX 7100 Graphics
67C0,	80,	AMD Radeon E9550
67C2,	01,	AMD Radeon Pro V7350x2
67C2,	02,	AMD Radeon Pro V7300X
67C4,	00,	AMD Radeon Pro WX 7100 Graphics
67C4,	80,	AMD Radeon E9560 / E9565 Graphics
67C7,	00,	AMD Radeon Pro WX 5100 Graphics
67C7,	80,	AMD Radeon E9390 Graphics
67D0,	01,	AMD Radeon Pro V7350x2
67D0,	02,	AMD Radeon Pro V7300X
67DF,	C0,	AMD Radeon Pro 580X
67DF,	C1,	AMD Radeon RX 580 Series
67DF,	C2,	AMD Radeon RX 570 Series
67DF,	C3,	AMD Radeon RX 580 Series
67DF,	C4,	AMD Radeon RX 480 Graphics
67DF,	C5,	AMD Radeon RX 470 Graphics
67DF,	C6,	AMD Radeon RX 570 Series
67DF,	C7,	AMD Radeon RX 480 Graphics
67DF,	CF,	AMD Radeon RX 470 Graphics
67DF,	D7,	AMD Radeon RX 470 Graphics
67DF,	E0,	AMD Radeon RX 470 Series
67DF,	E1,	AMD Radeon RX 590 Series
67DF,	E3,	AMD Radeon RX Series
67DF,	E7,	AMD Radeon RX 580 Series
67DF,	EB,	AMD Radeon Pro 580X
67DF,	EF,	AMD Radeon RX 570 Series
67DF,	F7,	AMD Radeon RX P30PH
67DF,	FF,	AMD Radeon RX 470 Series
67E0,	00,	AMD Radeon Pro WX Series
67E3,	00,	AMD Radeon Pro WX 4100
67E8,	00,	AMD Radeon Pro WX Series
67E8,	01,	AMD Radeon Pro WX Series
67E8,	80,	AMD Radeon E9260 Graphics
67EB,	00,	AMD Radeon Pro V5300X
67EF,	C0,	AMD Radeon RX Graphics
67EF,	C1,	AMD Radeon RX 460 Graphics
67EF,	C2,	AMD Radeon Pro Series
67EF,	C3,	AMD Radeon RX Series
67EF,	C5,	AMD Radeon RX 460 Graphics
67EF,	C7,	AMD Radeon RX Graphics
67EF,	CF,	AMD Radeon RX 460 Graphics
67EF,	E0,	AMD Radeon RX 560 Series
67EF,	E1,	AMD Radeon RX Series
67EF,	E2,	AMD Radeon RX 560X
67EF,	E3,	AMD Radeon RX Series
67EF,	E5,	AMD Radeon RX 560 Series
67EF,	E7,	AMD Radeon RX 560 Series
67EF,	EF,	AMD Radeon 550 Series
67EF,	FF,	AMD Radeon RX 460 Graphics
67FF,	C0,	AMD Radeon Pro 465
67FF,	C1,	AMD Radeon RX 560 Series
67FF,	CF,	AMD Radeon RX 560 Series
67FF,	EF,	AMD Radeon RX 560 Series
67FF,	FF,	AMD Radeon RX 550 Series
6800,	00,	AMD Radeon HD 7970M
6801,	00,	AMD Radeon HD 8970M
6806,	00,	AMD Radeon R9 M290X
6808,	00,	AMD FirePro W7000
6808,	00,	ATI FirePro V (FireGL V) Graphics Adapter
6809,	00,	ATI FirePro W5000
6810,	00,	AMD Radeon R9 200 Series
6810,	81,	AMD Radeon R9 370 Series
6811,	00,	AMD Radeon R9 200 Series
6811,	81,	AMD Radeon R7 370 Series
6818,	00,	AMD Radeon HD 7800 Series
6819,	00,	AMD Radeon HD 7800 Series
6820,	00,	AMD Radeon R9 M275X
6820,	81,	AMD Radeon R9 M375
6820,	83,	AMD Radeon R9 M375X
6821,	00,	AMD Radeon R9 M200X Series
6821,	83,	AMD Radeon R9 M370X
6821,	87,	AMD Radeon R7 M380
6822,	00,	AMD Radeon E8860
6823,	00,	AMD Radeon R9 M200X Series
6825,	00,	AMD Radeon HD 7800M Series
6826,	00,	AMD Radeon HD 7700M Series
6827,	00,	AMD Radeon HD 7800M Series
6828,	00,	AMD FirePro W600
682B,	00,	AMD Radeon HD 8800M Series
682B,	87,	AMD Radeon R9 M360
682C,	00,	AMD FirePro W4100
682D,	00,	AMD Radeon HD 7700M Series
682F,	00,	AMD Radeon HD 7700M Series
6830,	00,	AMD Radeon 7800M Series
6831,	00,	AMD Radeon 7700M Series
6835,	00,	AMD Radeon R7 Series / HD 9000 Series
6837,	00,	AMD Radeon HD 7700 Series
683D,	00,	AMD Radeon HD 7700 Series
683F,	00,	AMD Radeon HD 7700 Series
684C,	00,	ATI FirePro V (FireGL V) Graphics Adapter
6860,	00,	AMD Radeon Instinct MI25
6860,	01,	AMD Radeon Instinct MI25
6860,	02,	AMD Radeon Instinct MI25
6860,	03,	AMD Radeon Pro V340
6860,	04,	AMD Radeon Instinct MI25x2
6860,	07,	AMD Radeon Pro V320
6861,	00,	AMD Radeon Pro WX 9100
6862,	00,	AMD Radeon Pro SSG
6863,	00,	AMD Radeon Vega Frontier Edition
6864,	03,	AMD Radeon Pro V340
6864,	04,	AMD Radeon Instinct MI25x2
6864,	05,	AMD Radeon Pro V340
6868,	00,	AMD Radeon Pro WX 8200
686C,	00,	AMD Radeon Instinct MI25 MxGPU
686C,	01,	AMD Radeon Instinct MI25 MxGPU
686C,	02,	AMD Radeon Instinct MI25 MxGPU
686C,	03,	AMD Radeon Pro V340 MxGPU
686C,	04,	AMD Radeon Instinct MI25x2 MxGPU
686C,	05,	AMD Radeon Pro V340L MxGPU
686C,	06,	AMD Radeon Instinct MI25 MxGPU
687F,	01,	AMD Radeon RX Vega
687F,	C0,	AMD Radeon RX Vega
687F,	C1,	AMD Radeon RX Vega
687F,	C3,	AMD Radeon RX Vega
687F,	C7,	AMD Radeon RX Vega
6900,	00,	AMD Radeon R7 M260
6900,	81,	AMD Radeon R7 M360
6900,	83,	AMD Radeon R7 M340
6900,	C1,	AMD Radeon R5 M465 Series
6900,	C3,	AMD Radeon R5 M445 Series
6900,	D1,	AMD Radeon 530 Series
6900,	D3,	AMD Radeon 530 Series
6901,	00,	AMD Radeon R5 M255
6902,	00,	AMD Radeon Series
6907,	00,	AMD Radeon R5 M255
6907,	87,	AMD Radeon R5 M315
6920,	00,	AMD Radeon R9 M395X
6920,	01,	AMD Radeon R9 M390X
6921,	00,	AMD Radeon R9 M390X
6929,	00,	AMD FirePro S7150
6929,	01,	AMD FirePro S7100X
692B,	00,	AMD FirePro W7100
6938,	00,	AMD Radeon R9 200 Series
6938,	F0,	AMD Radeon R9 200 Series
6938,	F1,	AMD Radeon R9 380 Series
6939,	00,	AMD Radeon R9 200 Series
6939,	F0,	AMD Radeon R9 200 Series
6939,	F1,	AMD Radeon R9 380 Series
694C,	C0,	AMD Radeon RX Vega M GH Graphics
694E,	C0,	AMD Radeon RX Vega M GL Graphics
6980,	00,	AMD Radeon Pro WX 3100
6981,	00,	AMD Radeon Pro WX 3200 Series
6981,	01,	AMD Radeon Pro WX 3200 Series
6981,	10,	AMD Radeon Pro WX 3200 Series
6985,	00,	AMD Radeon Pro WX 3100
6986,	00,	AMD Radeon Pro WX 2100
6987,	80,	AMD Embedded Radeon E9171
6987,	C0,	AMD Radeon 550X Series
6987,	C1,	AMD Radeon RX 640
6987,	C3,	AMD Radeon 540X Series
6987,	C7,	AMD Radeon 540
6995,	00,	AMD Radeon Pro WX 2100
6997,	00,	AMD Radeon Pro WX 2100
699F,	81,	AMD Embedded Radeon E9170 Series
699F,	C0,	AMD Radeon 500 Series
699F,	C1,	AMD Radeon 540 Series
699F,	C3,	AMD Radeon 500 Series
699F,	C7,	AMD Radeon RX 550 / 550 Series
699F,	C9,	AMD Radeon 540
6FDF,	E7,	AMD Radeon RX 590 GME
6FDF,	EF,	AMD Radeon RX 580 2048SP
7300,	C1,	AMD FirePro S9300 x2
7300,	C8,	AMD Radeon R9 Fury Series
7300,	C9,	AMD Radeon Pro Duo
7300,	CA,	AMD Radeon R9 Fury Series
7300,	CB,	AMD Radeon R9 Fury Series
7312,	00,	AMD Radeon Pro W5700
731E,	C6,	AMD Radeon RX 5700XTB
731E,	C7,	AMD Radeon RX 5700B
731F,	C0,	AMD Radeon RX 5700 XT 50th Anniversary
731F,	C1,	AMD Radeon RX 5700 XT
731F,	C2,	AMD Radeon RX 5600M
731F,	C3,	AMD Radeon RX 5700M
731F,	C4,	AMD Radeon RX 5700
731F,	C5,	AMD Radeon RX 5700 XT
731F,	CA,	AMD Radeon RX 5600 XT
731F,	CB,	AMD Radeon RX 5600 OEM
7340,	C1,	AMD Radeon RX 5500M
7340,	C3,	AMD Radeon RX 5300M
7340,	C5,	AMD Radeon RX 5500 XT
7340,	C7,	AMD Radeon RX 5500
7340,	C9,	AMD Radeon RX 5500XTB
7340,	CF,	AMD Radeon RX 5300
7341,	00,	AMD Radeon Pro W5500
7347,	00,	AMD Radeon Pro W5500M
7360,	41,	AMD Radeon Pro 5600M
7360,	C3,	AMD Radeon Pro V520
7362,	C1,	AMD Radeon Pro V540
7362,	C3,	AMD Radeon Pro V520
738C,	01,	AMD Instinct MI100
73A1,	00,	AMD Radeon Pro V620
73A3,	00,	AMD Radeon Pro W6800
73A5,	C0,	AMD Radeon RX 6950 XT
73AE,	00,	AMD Radeon Pro V620 MxGPU
73AF,	C0,	AMD Radeon RX 6900 XT
73BF,	C0,	AMD Radeon RX 6900 XT
73BF,	C1,	AMD Radeon RX 6800 XT
73BF,	C3,	AMD Radeon RX 6800
73DF,	C0,	AMD Radeon RX 6750 XT
73DF,	C1,	AMD Radeon RX 6700 XT
73DF,	C2,	AMD Radeon RX 6800M
73DF,	C3,	AMD Radeon RX 6800M
73DF,	C5,	AMD Radeon RX 6700 XT
73DF,	CF,	AMD Radeon RX 6700M
73DF,	D5,	AMD Radeon RX 6750 GRE 12GB
73DF,	D7,	AMD TDC-235
73DF,	DF,	AMD Radeon RX 6700
73DF,	E5,	AMD Radeon RX 6750 GRE 12GB
73DF,	FF,	AMD Radeon RX 6700
73E0,	00,	AMD Radeon RX 6600M
73E1,	00,	AMD Radeon Pro W6600M
73E3,	00,	AMD Radeon Pro W6600
73EF,	C0,	AMD Radeon RX 6800S
73EF,	C1,	AMD Radeon RX 6650 XT
73EF,	C2,	AMD Radeon RX 6700S
73EF,	C3,	AMD Radeon RX 6650M
73EF,	C4,	AMD Radeon RX 6650M XT
73FF,	C1,	AMD Radeon RX 6600 XT
73FF,	C3,	AMD Radeon RX 6600M
73FF,	C7,	AMD Radeon RX 6600
73FF,	CB,	AMD Radeon RX 6600S
73FF,	CF,	AMD Radeon RX 6600 LE
73FF,	DF,	AMD Radeon RX 6750 GRE 10GB
7408,	00,	AMD Instinct MI250X
740C,	01,	AMD Instinct MI250X / MI250
740F,	02,	AMD Instinct MI210
7421,	00,	AMD Radeon Pro W6500M
7422,	00,	AMD Radeon Pro W6400
7423,	00,	AMD Radeon Pro W6300M
7423,	01,	AMD Radeon Pro W6300
7424,	00,	AMD Radeon RX 6300
743F,	C1,	AMD Radeon RX 6500 XT
743F,	C3,	AMD Radeon RX 6500
743F,	C3,	AMD Radeon RX 6500M
743F,	C7,	AMD Radeon RX 6400
743F,	C8,	AMD Radeon RX 6500M
743F,	CC,	AMD Radeon 6550S
743F,	CE,	AMD Radeon RX 6450M
743F,	CF,	AMD Radeon RX 6300M
743F,	D3,	AMD Radeon RX 6550M
743F,	D7,	AMD Radeon RX 6400
7448,	00,	AMD Radeon Pro W7900
7449,	00,	AMD Radeon Pro W7800 48GB
744A,	00,	AMD Radeon Pro W7900 Dual Slot
744B,	00,	AMD Radeon Pro W7900D
744C,	C8,	AMD Radeon RX 7900 XTX
744C,	CC,	AMD Radeon RX 7900 XT
744C,	CE,	AMD Radeon RX 7900 GRE
744C,	CF,	AMD Radeon RX 7900M
745E,	CC,	AMD Radeon Pro W7800
7460,	00,	AMD Radeon Pro V710
7461,	00,	AMD Radeon Pro V710 MxGPU
7470,	00,	AMD Radeon Pro W7700
747E,	C8,	AMD Radeon RX 7800 XT
747E,	D8,	AMD Radeon RX 7800M
747E,	DB,	AMD Radeon RX 7700
747E,	FF,	AMD Radeon RX 7700 XT
7480,	00,	AMD Radeon Pro W7600
7480,	C0,	AMD Radeon RX 7600 XT
7480,	C1,	AMD Radeon RX 7700S
7480,	C2,	AMD Radeon RX 7650 GRE
7480,	C3,	AMD Radeon RX 7600S
7480,	C7,	AMD Radeon RX 7600M XT
7480,	CF,	AMD Radeon RX 7600
7481,   C7,     AMD Steam Machine
7483,	CF,	AMD Radeon RX 7600M
7489,	00,	AMD Radeon Pro W7500
7499,	00,	AMD Radeon Pro W7400
7499,	C0,	AMD Radeon RX 7400
7499,	C1,	AMD Radeon RX 7300
74A0,	00,	AMD Instinct MI300A
74A1,	00,	AMD Instinct MI300X
74A2,	00,	AMD Instinct MI308X
74A5,	00,	AMD Instinct MI325X
74A8,	00,	AMD Instinct MI308X HF
74A9,	00,	AMD Instinct MI300X HF
74B5,	00,	AMD Instinct MI300X VF
74B6,	00,	AMD Instinct MI308X
74BD,	00,	AMD Instinct MI300X HF
7550,	C0,	AMD Radeon RX 9070 XT
7550,	C2,	AMD Radeon RX 9070 GRE
7550,	C3,	AMD Radeon RX 9070
7551,	C0,	AMD Radeon AI PRO R9700
7551,	C8,	AMD Radeon AI PRO R9600D
7590,	C0,	AMD Radeon RX 9060 XT
7590,	C1,	AMD Radeon RX 9060 XT LP
7590,	C7,	AMD Radeon RX 9060
75A0,	00,	AMD Instinct MI350X
75A3,	00,	AMD Instinct MI355X
75B0,	00,	AMD Instinct MI350X VF
75B3,	00,	AMD Instinct MI355X VF
9830,	00,	AMD Radeon HD 8400 / R3 Series
9831,	00,	AMD Radeon HD 8400E
9832,	00,	AMD Radeon HD 8330
9833,	00,	AMD Radeon HD 8330E
9834,	00,	AMD Radeon HD 8210
9835,	00,	AMD Radeon HD 8210E
9836,	00,	AMD Radeon HD 8200 / R3 Series
9837,	00,	AMD Radeon HD 8280E
9838,	00,	AMD Radeon HD 8200 / R3 series
9839,	00,	AMD Radeon HD 8180
983D,	00,	AMD Radeon HD 8250
9850,	00,	AMD Radeon R3 Graphics
9850,	03,	AMD Radeon R3 Graphics
9850,	40,	AMD Radeon R2 Graphics
9850,	45,	AMD Radeon R3 Graphics
9851,	00,	AMD Radeon R4 Graphics
9851,	01,	AMD Radeon R5E Graphics
9851,	05,	AMD Radeon R5 Graphics
9851,	06,	AMD Radeon R5E Graphics
9851,	40,	AMD Radeon R4 Graphics
9851,	45,	AMD Radeon R5 Graphics
9852,	00,	AMD Radeon R2 Graphics
9852,	40,	AMD Radeon E1 Graphics
9853,	00,	AMD Radeon R2 Graphics
9853,	01,	AMD Radeon R4E Graphics
9853,	03,	AMD Radeon R2 Graphics
9853,	05,	AMD Radeon R1E Graphics
9853,	06,	AMD Radeon R1E Graphics
9853,	07,	AMD Radeon R1E Graphics
9853,	08,	AMD Radeon R1E Graphics
9853,	40,	AMD Radeon R2 Graphics
9854,	00,	AMD Radeon R3 Graphics
9854,	01,	AMD Radeon R3E Graphics
9854,	02,	AMD Radeon R3 Graphics
9854,	05,	AMD Radeon R2 Graphics
9854,	06,	AMD Radeon R4 Graphics
9854,	07,	AMD Radeon R3 Graphics
9855,	02,	AMD Radeon R6 Graphics
9855,	05,	AMD Radeon R4 Graphics
9856,	00,	AMD Radeon R2 Graphics
9856,	01,	AMD Radeon R2E Graphics
9856,	02,	AMD Radeon R2 Graphics
9856,	05,	AMD Radeon R1E Graphics
9856,	06,	AMD Radeon R2 Graphics
9856,	07,	AMD Radeon R1E Graphics
9856,	08,	AMD Radeon R1E Graphics
9856,	13,	AMD Radeon R1E Graphics
9874,	81,	AMD Radeon R6 Graphics
9874,	84,	AMD Radeon R7 Graphics
9874,	85,	AMD Radeon R6 Graphics
9874,	87,	AMD Radeon R5 Graphics
9874,	88,	AMD Radeon R7E Graphics
9874,	89,	AMD Radeon R6E Graphics
9874,	C4,	AMD Radeon R7 Graphics
9874,	C5,	AMD Radeon R6 Graphics
9874,	C6,	AMD Radeon R6 Graphics
9874,	C7,	AMD Radeon R5 Graphics
9874,	C8,	AMD Radeon R7 Graphics
9874,	C9,	AMD Radeon R7 Graphics
9874,	CA,	AMD Radeon R5 Graphics
9874,	CB,	AMD Radeon R5 Graphics
9874,	CC,	AMD Radeon R7 Graphics
9874,	CD,	AMD Radeon R7 Graphics
9874,	CE,	AMD Radeon R5 Graphics
9874,	E1,	AMD Radeon R7 Graphics
9874,	E2,	AMD Radeon R7 Graphics
9874,	E3,	AMD Radeon R7 Graphics
9874,	E4,	AMD Radeon R7 Graphics
9874,	E5,	AMD Radeon R5 Graphics
9874,	E6,	AMD Radeon R5 Graphics
98E4,	80,	AMD Radeon R5E Graphics
98E4,	81,	AMD Radeon R4E Graphics
98E4,	83,	AMD Radeon R2E Graphics
98E4,	84,	AMD Radeon R2E Graphics
98E4,	86,	AMD Radeon R1E Graphics
98E4,	C0,	AMD Radeon R4 Graphics
98E4,	C1,	AMD Radeon R5 Graphics
98E4,	C2,	AMD Radeon R4 Graphics
98E4,	C4,	AMD Radeon R5 Graphics
98E4,	C6,	AMD Radeon R5 Graphics
98E4,	C8,	AMD Radeon R4 Graphics
98E4,	C9,	AMD Radeon R4 Graphics
98E4,	CA,	AMD Radeon R5 Graphics
98E4,	D0,	AMD Radeon R2 Graphics
98E4,	D1,	AMD Radeon R2 Graphics
98E4,	D2,	AMD Radeon R2 Graphics
98E4,	D4,	AMD Radeon R2 Graphics
98E4,	D9,	AMD Radeon R5 Graphics
98E4,	DA,	AMD Radeon R5 Graphics
98E4,	DB,	AMD Radeon R3 Graphics
98E4,	E1,	AMD Radeon R3 Graphics
98E4,	E2,	AMD Radeon R3 Graphics
98E4,	E9,	AMD Radeon R4 Graphics
98E4,	EA,	AMD Radeon R4 Graphics
98E4,	EB,	AMD Radeon R3 Graphics
98E4,	EB,	AMD Radeon R4 Graphics
"""

if __name__ == '__main__' or True:
    myInitializeRsmi()
    curses.wrapper(main)


