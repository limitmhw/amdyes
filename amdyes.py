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
                                            str(getDeviceName(device)).center(20),
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

if __name__ == '__main__' or True:
    myInitializeRsmi()
    curses.wrapper(main)


