#!/usr/bin/python3
import os, sys
import argparse
from datetime import datetime
from statistics import median
import numpy as np
import string
import json
import time
import copy
import threading
import numpy
import cv2
import queue
from openvino.inference_engine import IENetwork, IECore, get_version, StatusCode

model_path = "./"
#openvino_plugin_dir = "/opt/intel/dldt/inference-engine/lib/intel64/"
openvino_plugin_dir = os.environ["INTEL_OPENVINO_DIR"] + "/inference_engine/lib/intel64/"


def testThread(nt, rddata):
    if nt.duration_seconds == 0 and nt.niter == 0:
        return
    if nt.duration_seconds > 0 and nt.niter == -1:
        time.sleep(nt.duration_seconds)
        return
    nt.infer(rddata)
    if not nt.no_del:
        del(nt.exec_network)

class NetworkTest:
    def __init__(self, ie, device: str, duration_seconds, number_iter, \
            number_infer_requests, api_type, scheduler, stream_id, no_del):
        self.device = device.upper()
        self.ie = ie
        self.nireq_set = number_infer_requests
        self.nireq = 0
        self.niter = number_iter
        self.duration_seconds = duration_seconds
        self.api_type = api_type
        self.scheduler = scheduler
        self.stream_id = stream_id
        self.no_del = no_del
        self.device_number_streams = {}
        self.input_info = None
        self.request_queue = None
        self.ie_network = None
        self.exec_network = None
        self.thrd = None

        self.name = ""
        self.fps = 0
        self.latency_ms = 0
        self.total_duration_sec = 0
        self.iteration = 0

    def __del__(self):
        del self.ie

    def load_network(self, path_to_model: str):
        xml_filename = os.path.abspath(path_to_model)
        head, tail = os.path.splitext(xml_filename)
        bin_filename = os.path.abspath(head + ".bin")

        ie_network = self.ie.read_network(xml_filename, bin_filename)

        input_info = ie_network.input_info

        if not input_info:
            raise AttributeError('No inputs info is provided')

        for key in input_info.keys():
            #if is_image(input_info[key]):
            if 0:
                # Set the precision of input data provided by the user
                # Should be called before load of the network to the plugin
                input_info[key].precision = 'U8'

        self.input_info = input_info

        config = {'PERF_COUNT': 'NO'}

        tag = bind = prio = ""
        if self.scheduler == 'bypass':
            st = self.stream_id.split(':')
            tag = st[0]
            bind = st[1] if len(st) > 1 else 'NO'
            prio = st[2] if len(st) > 2 else '0'

        scheduler_config = {'default': None, \
            'squeeze': None, \
            'bypass': {'HDDL_DEVICE_TAG':tag,'HDDL_BIND_DEVICE':bind,'HDDL_RUNTIME_PRIORITY':prio}, \
            'tag': {'HDDL_GRAPH_TAG':self.stream_id}, \
            'stream': {'HDDL_STREAM_ID':self.stream_id}, \
            'sgad': {'HDDL_USE_SGAD':'YES'}, \
            None: None}
        if scheduler_config[self.scheduler] != None:
            config.update(scheduler_config[self.scheduler])
        #print(config)

        exe_network = self.ie.load_network(ie_network,
                                           self.device,
                                           config=config,
                                           num_requests=self.nireq_set or 0)

        print(path_to_model, 'Loaded')

        self.name = path_to_model
        self.exec_network = exe_network
        self.ie_network = ie_network
        self.nireq = len(exe_network.requests)

        return

    def infer(self, requests_input_data):
        # warming up - out of scope
        infer_request_id = self.exec_network.get_idle_request_id()
        exe_network = self.exec_network
        if infer_request_id < 0:
            status = exe_network.wait(num_requests=1)
            if status != StatusCode.OK:
                raise Exception("Wait for idle request failed!")
            infer_request_id = exe_network.get_idle_request_id()
            if infer_request_id < 0:
                        raise Exception("Invalid request id!")
        infer_request = exe_network.requests[infer_request_id]
        if self.api_type == 'sync':
            infer_request.infer(requests_input_data[infer_request_id])
        else:
            infer_request.async_infer(requests_input_data[infer_request_id])

        #request_queue.reset_times()
        exe_network.wait()

        times = []
        in_fly = set()
        start_time = datetime.now()
        exec_time = (datetime.now() - start_time).total_seconds()
        iteration = 0
        
        # Start inference & calculate performance
        # to align number if iterations to guarantee that last infer requests are executed in the same conditions **/
        while (self.niter and iteration < self.niter) or \
              (self.duration_seconds and exec_time < self.duration_seconds) or \
              (self.api_type == 'async' and iteration % self.nireq):
            infer_request_id = exe_network.get_idle_request_id()
            if infer_request_id < 0:
                status = exe_network.wait(num_requests=1)
                if status != StatusCode.OK:
                    raise Exception("Wait for idle request failed!")
                infer_request_id = exe_network.get_idle_request_id()
                if infer_request_id < 0:
                            raise Exception("Invalid request id!")
            infer_request = exe_network.requests[infer_request_id]
            if infer_request_id in in_fly:
                times.append(infer_request.latency)
            else:
                in_fly.add(infer_request_id)
            
            if self.api_type == 'sync':
                infer_request.infer(requests_input_data[infer_request_id])
            else:
                infer_request.async_infer(requests_input_data[infer_request_id])
            times.append(infer_request.latency)
            iteration += 1

            exec_time = (datetime.now() - start_time).total_seconds()

        # wait the latest inference executions
        exe_network.wait()
        exec_time = (datetime.now() - start_time).total_seconds()

        for infer_request_id in in_fly:
            times.append(exe_network.requests[infer_request_id].latency)
        times.sort()
        latency_ms = median(times)

        fps = 1000 / exec_time
        if self.api_type == 'async':
            fps = iteration / exec_time

        self.fps = iteration / exec_time
        self.latency_ms = latency_ms
        self.total_duration_sec = exec_time
        self.iteration = iteration
        return

    def random_inputs(self):
        requests_input_data = []
        input_info = self.input_info
        for request_id in range(0, self.nireq):
            input_data = {}
            keys = list(input_info.keys())
            for key in keys:
                precision = input_info[key].precision
                shape = input_info[key].input_data.shape
                if precision == "FP32":
                    input_data[key] = np.random.rand(*shape).astype(np.float32)
                elif precision == "FP16":
                    input_data[key] = np.random.rand(*shape).astype(np.float16)
                elif precision == "I32":
                    input_data[key] = np.random.rand(*shape).astype(np.int32)
                elif precision == "U8":
                    input_data[key] = np.random.rand(*shape).astype(np.uint8)
                elif precision == "I8":
                    input_data[key] = np.random.rand(*shape).astype(np.int8)
                elif precision == "U16":
                    input_data[key] = np.random.rand(*shape).astype(np.uint16)
                elif precision == "I16":
                    input_data[key] = np.random.rand(*shape).astype(np.int16)
                else:
                    raise Exception("Input precision is not supported: " + precision)

            requests_input_data.append(input_data)

        return requests_input_data

    def print_performance(self):
        print('====================================')
        print('Model:      {}'.format(self.name))
        print('Count:      {} iterations'.format(self.iteration))
        print('Duration:   {:.2f} ms'.format(self.total_duration_sec*1000))
        print('Latency:    {:.2f} ms'.format(self.latency_ms))
        print('Throughput: {:.2f} FPS'.format(self.fps))

    def start(self):
        self.thrd = threading.Thread(target=testThread, args=(self, self.random_inputs()))
        self.thrd.setDaemon(True)
        self.thrd.start()
        
    def wait(self):
        self.thrd.join()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, nargs='+', help='network model list')
    parser.add_argument('-d', '--device', type=str, default='hddl', help='HDDL/MYRIAD/CPU/GPU')
    parser.add_argument('-t', '--time', type=int, nargs='*', default=[], help='test duration')
    parser.add_argument('-n', '--num', type=int, nargs='*', default=[], help='number of infer request')
    parser.add_argument('-i', '--iteration', type=int, nargs='*', default=[], help='number of iterations')
    parser.add_argument('-s', '--hddl_scheduler', type=str, nargs='+', default=[], help='network scheduler')
    parser.add_argument('--stream_id', type=str, nargs='*', default=[], help='network stream id for stream scheduler')
    parser.add_argument('--no_del', action='store_true', default=False, help='disables delete exec_net')
    args = parser.parse_args()
    if args.model == None:
        parser.print_help()
        sys.exit()
    
    device = args.device
    duration_seconds = args.time
    number_infer_requests = args.num
    iteration = args.iteration
    network_list = args.model
    hddl_scheduler = args.hddl_scheduler
    stream_id = args.stream_id
    no_del = args.no_del

    #sys.exit()
    #duration_seconds = 15
    #number_infer_requests = 4
    #network_list = ["resnet-50-pytorch.xml", "mobilenet-v3-small-1.0-224-tf.xml", "yolo-v3-tiny-tf.xml"]
    
    ie = IECore()

    test_list = []

    for i, net in enumerate(network_list):
        duration = 0 if i >= len(duration_seconds) else duration_seconds[i]
        nireq = 0 if i >= len(number_infer_requests) else number_infer_requests[i]
        niter = 0 if i >= len(iteration) else iteration[i]
        scheduler = None if i >= len(hddl_scheduler) else hddl_scheduler[i]
        sid = 0 if i >= len(stream_id) else stream_id[i]
        
        test = NetworkTest(ie, device, duration, niter, nireq, "async", scheduler, sid, no_del)
        test.load_network(net)
        test_list.append(test)

    for t in test_list:
        t.start()

    for t in test_list:
        t.wait()

    for t in test_list:
        t.print_performance()
    

if __name__ == "__main__":
    main()
