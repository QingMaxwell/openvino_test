# openvino_test
This project is created for openvino testing

# test cases

## multi network benchmark

### hddl scheduler test
  
**python3 multi_network_benchmark.py -d HDDL --model resnet-50-pytorch.xml -t 10**  
Model:      resnet-50-pytorch.xml  
Count:      1216 iterations  
Duration:   10193.55 ms  
Latency:    133.02 ms  
Throughput: 119.29 FPS  
  
**python3 multi_network_benchmark.py -d HDDL --model yolo-v3-tiny-tf.xml -t 10**  
Model:      yolo-v3-tiny-tf.xml  
Count:      1872 iterations  
Duration:   10152.97 ms  
Latency:    85.68 ms  
Throughput: 184.38 FPS  
  
**python3 multi_network_benchmark.py -d HDDL --model mobilenet-v3-small-1.0-224-tf.xml -t 10**  
Model:      mobilenet-v3-small-1.0-224-tf.xml  
Count:      3584 iterations  
Duration:   10054.60 ms  
Latency:    44.55 ms  
Throughput: 356.45 FPS  
  
**python3 multi_network_benchmark.py -d HDDL --model resnet-50-pytorch.xml yolo-v3-tiny-tf.xml mobilenet-v3-small-1.0-224-tf.xml -t 30 30 30**  
Model:      resnet-50-pytorch.xml  
Count:      112 iterations  
Duration:   35117.67 ms  
Latency:    136.49 ms  
Throughput: 3.19 FPS  
  
Model:      yolo-v3-tiny-tf.xml  
Count:      744 iterations  
Duration:   30213.12 ms  
Latency:    128.62 ms  
Throughput: 24.63 FPS  

Model:      mobilenet-v3-small-1.0-224-tf.xml  
Count:      4692 iterations  
Duration:   30044.78 ms  
Latency:    31.07 ms  
Throughput: 156.17 FPS  


**"device_schedule_interval":    1000**  
**"max_cycle_switch_out":        3**  
**"max_task_number_switch_out":  10**  
**python3 multi_network_benchmark.py -d HDDL --model resnet-50-pytorch.xml yolo-v3-tiny-tf.xml mobilenet-v3-small-1.0-224-tf.xml -t 30 30 30** 
    
Model:      resnet-50-pytorch.xml  
Count:      1600 iterations  
Duration:   30274.27 ms  
Latency:    252.81 ms  
Throughput: 52.85 FPS  
  
Model:      yolo-v3-tiny-tf.xml  
Count:      1452 iterations  
Duration:   30253.32 ms  
Latency:    241.48 ms  
Throughput: 47.99 FPS  
  
Model:      mobilenet-v3-small-1.0-224-tf.xml  
Count:      3192 iterations  
Duration:   30057.52 ms  
Latency:    58.94 ms  
Throughput: 106.20 FPS  

**python3 multi_network_benchmark.py -d HDDL --model resnet-50-pytorch.xml yolo-v3-tiny-tf.xml mobilenet-v3-small-1.0-224-tf.xml -t 30 30 30 -n 16 32 16**  
Model:      resnet-50-pytorch.xml  
Count:      1760 iterations  
Duration:   30299.84 ms  
Latency:    253.33 ms  
Throughput: 58.09 FPS  
  
Model:      yolo-v3-tiny-tf.xml  
Count:      1536 iterations  
Duration:   30545.72 ms  
Latency:    672.61 ms  
Throughput: 50.29 FPS  
  
Model:      mobilenet-v3-small-1.0-224-tf.xml  
Count:      2848 iterations  
Duration:   30046.98 ms  
Latency:    60.76 ms  
Throughput: 94.78 FPS  

**python3 multi_network_benchmark.py -d HDDL --model resnet-50-pytorch.xml yolo-v3-tiny-tf.xml yolo-v3-tiny-tf.xml mobilenet-v3-small-1.0-224-tf.xml -t 30 30 30 30**  
Model:      resnet-50-pytorch.xml  
Count:      912 iterations  
Duration:   30701.41 ms  
Latency:    529.94 ms  
Throughput: 29.71 FPS  
  
Model:      yolo-v3-tiny-tf.xml  
Count:      1056 iterations  
Duration:   30318.90 ms  
Latency:    71.66 ms  
Throughput: 34.83 FPS  
  
Model:      yolo-v3-tiny-tf.xml  
Count:      1176 iterations  
Duration:   30166.13 ms  
Latency:    66.81 ms  
Throughput: 38.98 FPS  
  
Model:      mobilenet-v3-small-1.0-224-tf.xml  
Count:      1668 iterations  
Duration:   30044.48 ms  
Latency:    31.94 ms  
Throughput: 55.52 FPS  

**"graph_tag_map":{"tagA":2, "tagB":1, "tagC":1}**  
**python3 multi_network_benchmark.py -d HDDL --model resnet-50-pytorch.xml yolo-v3-tiny-tf.xml mobilenet-v3-small-1.0-224-tf.xml -t 30 30 30**  
Model:      resnet-50-pytorch.xml  
Count:      1824 iterations  
Duration:   30457.42 ms  
Latency:    256.73 ms  
Throughput: 59.89 FPS  
  
Model:      yolo-v3-tiny-tf.xml  
Count:      1408 iterations  
Duration:   30480.97 ms  
Latency:    332.02 ms  
Throughput: 46.19 FPS  
  
Model:      mobilenet-v3-small-1.0-224-tf.xml  
Count:      2544 iterations  
Duration:   30078.69 ms  
Latency:    115.34 ms  
Throughput: 84.58 FPS  

**"stream_device_number":4**  
**python3 multi_network_benchmark.py -d HDDL --model resnet-50-pytorch.xml yolo-v3-tiny-tf.xml mobilenet-v3-small-1.0-224-tf.xml -t 30 30 30 -n 16 16 16 -c stream:tagA stream:tagB stream:tagC**  
Model:      resnet-50-pytorch.xml  
Count:      928 iterations  
Duration:   31054.52 ms  
Latency:    532.06 ms  
Throughput: 29.88 FPS  
  
Model:      yolo-v3-tiny-tf.xml  
Count:      992 iterations  
Duration:   30155.52 ms  
Latency:    59.77 ms  
Throughput: 32.90 FPS  
  
Model:      mobilenet-v3-small-1.0-224-tf.xml  
Count:      1360 iterations  
Duration:   30195.17 ms  
Latency:    29.62 ms  
Throughput: 45.04 FPS  

**python3 multi_network_benchmark.py -d HDDL --model resnet-50-pytorch.xml yolo-v3-tiny-tf.xml mobilenet-v3-small-1.0-224-tf.xml resnet-50-pytorch.xml -t 30 30 30 30 -n 16 16 16 16 -c stream:tagA stream:tagB stream:tagC stream:tagD**  
Model:      resnet-50-pytorch.xml  
Count:      928 iterations  
Duration:   30972.32 ms  
Latency:    528.16 ms  
Throughput: 29.96 FPS  
  
Model:      yolo-v3-tiny-tf.xml  
Count:      944 iterations  
Duration:   30432.27 ms  
Latency:    59.48 ms  
Throughput: 31.02 FPS  
  
Model:      mobilenet-v3-small-1.0-224-tf.xml  
Count:      1136 iterations  
Duration:   30067.01 ms  
Latency:    28.14 ms  
Throughput: 37.78 FPS  
  
Model:      resnet-50-pytorch.xml  
Count:      912 iterations  
Duration:   30640.47 ms  
Latency:    525.39 ms  
Throughput: 29.76 FPS  

**"sgad_device_number":4**  
**python3 multi_network_benchmark.py -d HDDL --model resnet-50-pytorch.xml yolo-v3-tiny-tf.xml mobilenet-v3-small-1.0-224-tf.xml -t 30 30 30 -n 16 16 16 -c sgad sgad sgad**  
Model:      resnet-50-pytorch.xml  
Count:      1776 iterations  
Duration:   30368.69 ms  
Latency:    270.23 ms  
Throughput: 58.48 FPS  
  
Model:      yolo-v3-tiny-tf.xml  
Count:      1728 iterations  
Duration:   30205.15 ms  
Latency:    240.55 ms  
Throughput: 57.21 FPS  
  
Model:      mobilenet-v3-small-1.0-224-tf.xml  
Count:      2448 iterations  
Duration:   30049.43 ms  
Latency:    121.11 ms  
Throughput: 81.47 FPS

**"bypass_device_number":4**  
**python3 multi_network_benchmark.py -d HDDL --model resnet-50-pytorch.xml yolo-v3-tiny-tf.xml mobilenet-v3-small-1.0-224-tf.xml resnet-50-pytorch.xml -t 30 30 30 0 -n 16 16 16 0 -c bypass:dev0:NO:0 bypass:dev1:NO:0 bypass:dev2:NO:0 bypass:dev3:NO:0**  
Model:      resnet-50-pytorch.xml  
Count:      1824 iterations  
Duration:   30410.76 ms  
Latency:    254.66 ms  
Throughput: 59.98 FPS  
  
Model:      yolo-v3-tiny-tf.xml  
Count:      1408 iterations  
Duration:   30464.70 ms  
Latency:    333.21 ms  
Throughput: 46.22 FPS  
  
Model:      mobilenet-v3-small-1.0-224-tf.xml  
Count:      2560 iterations  
Duration:   30230.57 ms  
Latency:    104.85 ms  
Throughput: 84.68 FPS  
  
Model:      resnet-50-pytorch.xml  
Count:      0 iterations  
Duration:   0.00 ms  
Latency:    0.00 ms  
Throughput: 0.00 FPS  
