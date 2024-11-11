#!/usr/bin/env python
import importlib.util
import sys
import os
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
        gpu_info =  "│{}{}{}{}{}{}{}│".format(str(device).center(4),
                                            str(getDeviceName(device)).center(20),
                                            str(temp_val).center(10),
                                            str(getPerfLevel(device)).center(7),
                                            (str(powerVal) + " / " + str(int(getMaxPower(device, True))) + "W" ).center(20),
                                            ((str(memInfo[0] // 1024 // 1024) + "MiB").rjust(9) + 
                                            " / " +
                                            (str(memInfo[1] // 1024 // 1024) + "MiB ").rjust(9) + 
                                            (str(memoryUse.value) + "%").rjust(4)).center(30), 
                                            (str(getGpuUse(device, True)) + "%").center(9))
        if device == 0:
            gpu_info_length = len(gpu_info)
        output.append(gpu_info)
        output.append("│" + ("─" * (gpu_info_length - 2)) + "│")

    head =  [
        "│ ROCM-SMI version: {}              ROCM-SMI-LIB version: {}".format(__version__, version_string).ljust(gpu_info_length-1) + "│",
        "│" + ("─" * (gpu_info_length - 2)) + "│",
        "│{}{}{}{}{}{}{}│".format("ID".center(4),
                                "DeviceName".center(20),
                                "Temp".center(10),
                                "Perf".center(7),
                                "Pwr:Usage/Cap".center(20),
                                "Memory-Usage".center(31),
                                "GPU-Util".center(8)
                            ),
    "│" + ("─" * (gpu_info_length - 2)) + "│",
    ]
    head.extend(output)
    output = head
    output.insert(0, "╒" + ("═" * (gpu_info_length - 2)) + "╕")
    output = output[:-1]
    output.append("╘" + ("═" * (gpu_info_length - 2)) + "╛")
    pid_info = showPids()
    ps_info = []
    for item in pid_info[1:]:
        pid = int(item[0])
        username, cpu_usage, cmdline = get_process_info(pid)
        item.insert(0, username)
        item.append(str(cpu_usage) + "%")
        item.append(str(" ".join(cmdline)))
        ps_info.append(item)

    output.append("{}{}{}{}{}{}".format(
                            "USER".ljust(15),
                            "PID".ljust(10),
                            "GPU".ljust(4),
                            "GPU MEMORY".ljust(12),
                            "CPU USAGE".ljust(14),
                            "COMMAND".ljust(10)
                            ))

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
        output.append(ps_str)
    return output

def main(stdscr):
    stdscr.clear()
    height, width = stdscr.getmaxyx()
    update_interval = 1
    while True:
        stdscr.clear()
        output = get_data(height, width)
        for i, line in enumerate(output):
            if i < height - 1:
                stdscr.addstr(i + 1, 0, line)
        stdscr.refresh()
        time.sleep(update_interval)

if __name__ == '__main__' or True:
    myInitializeRsmi()
    curses.wrapper(main)


