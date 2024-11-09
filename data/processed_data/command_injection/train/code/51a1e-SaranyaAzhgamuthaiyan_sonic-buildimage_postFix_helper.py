import os
import struct
import subprocess
from mmap import *
from sonic_py_common.general import check_output_pipe

<fix/>HOST_CHK_CMD = ["docker"]</fix>
EMPTY_STRING = ""


class APIHelper():

    def __init__(self):
        pass

    def is_host(self):
        <fix/>try:
            subprocess.call(HOST_CHK_CMD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            return False
        return True</fix>

    def pci_get_value(self, resource, offset):
        status = True
        result = ""
        try:
            fd = os.open(resource, os.O_RDWR)
            mm = mmap(fd, 0)
            mm.seek(int(offset))
            read_data_stream = mm.read(4)
            result = struct.unpack('I', read_data_stream)
        except:
            status = False
        return status, result

    <fix/>def run_command(self, cmd1_args, cmd2_args):</fix>
        status = True
        result = ""
        try:
            <fix/>result = check_output_pipe(cmd1_args, cmd2_args)
        except subprocess.CalledProcessError:</fix>
            status = False
        return status, result

    def run_interactive_command(self, cmd):
        try:
            <fix/>subprocess.call(cmd)</fix>
        except:
            return False
        return True

    def read_txt_file(self, file_path):
        try:
            with open(file_path, 'r') as fd:
                data = fd.read()
                return data.strip()
        except IOError:
            pass
        return None

    def ipmi_raw(self, netfn, cmd):
        status = True
        result = ""
        try:
            <fix/>cmd = ["ipmitool", "raw", str(netfn), str(cmd)]</fix>
            p = subprocess.Popen(
                <fix/><fix/>cmd, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)</fix></fix>
            raw_data, err = p.communicate()
            if err == '':
                result = raw_data.strip()
            else:
                status = False
        except:
            status = False
        return status, result

    def ipmi_fru_id(self, id, key=None):
        status = True
        result = ""
        <fix/>cmd1_args = ["ipmitool", "fru", "print", str(id)]
        if not key:
            try:
                p = subprocess.Popen(
                    cmd1_args, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                raw_data, err = p.communicate()
                if err == '':
                    result = raw_data.strip()
                else:
                    status = False
            except:</fix>
                status = False
        <fix/>else:
            cmd2_args = ["grep", str(key)]
            status, result = self.run_command(cmd1_args, cmd2_args)</fix>
        return status, result

    def ipmi_set_ss_thres(self, id, threshold_key, value):
        status = True
        result = ""
        try:
            <fix/>cmd = ["ipmitool", "sensor", "thresh", str(id), str(threshold_key), str(value)]</fix>
            p = subprocess.Popen(
                cmd, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            raw_data, err = p.communicate()
            if err == '':
                result = raw_data.strip()
            else:
                status = False
        except:
            status = False
        return status, result
