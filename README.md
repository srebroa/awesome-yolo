# awesome-yolo :octocat:


# Object Detection DNN Algorithms 2021

**Most of the DNN object detection algorithm can:**
- classifies object
- localize object

**Features thats counts for the modern Object Detection Algorithms are following:**
- Accuracy (AP - Average Precission)
- Speed (FPS -Frames Per Second)

# YOLO - You Only Looks Once
- **Yolo v1** (2016) Joseph Redmon [‘You Only Look Once: Unified, Real-Time Object Detection’](https://arxiv.org/abs/1506.02640)
- [**Yolo v2**](https://github.com/longcw/yolo2-pytorch) (2017) Joseph Redmon [‘YOLO9000: Better, Faster, Stronger’](https://arxiv.org/abs/1612.08242)
- [**Yolo v3**](https://github.com/ultralytics/yolov3) (2018) Joseph Redmon [‘YOLOv3: An Incremental Improvement’](https://arxiv.org/abs/1804.02767)
- [**Yolo v4**](https://github.com/AlexeyAB/darknet) (2020) Alexey Bochkovskiy [‘YOLOv4: Optimal Speed and Accuracy of Object Detection’](https://arxiv.org/abs/2004.10934). AP increase by 10% and FPS by 12% compared to v3
- [**Yolo v5**](https://github.com/ultralytics/yolov5) (2020) Glen Jocher - PyTorch implementation (v1 to v4 Darknet implementation). The major improvements includes mosaic data augmentation and auto learning bounding box anchors
- [**PP-Yolo**](https://github.com/PaddlePaddle/PaddleDetection) (2020) Xiang Long et al.(Baidu) [‘PP-YOLO: An Effective and Efficient Implementation of Object Detector’](https://arxiv.org/abs/2007.12099). PP-YOLO is based on v3 model with replacement of Darknet 53 backbone of YOLO v3 with a ResNet backbone and increase of training batch size from 64 to 192. Improved mAP to 45.2% (from 43.5 for v4) and FPS from 65 to 73 for Tesla V100 (batch size = 1). Based on PaddlePaddle DL framework
- **Yolo Z** (2021) Aduen Benjumea et al. [‘YOLO-Z: Improving small object detection in YOLOv5 for autonomous vehicles’](https://arxiv.org/abs/2112.11798v2)
- [**Yolo-ReT**](https://github.com/guotao0628/yoloret) (2021) Prakhar Ganesh et al. [‘YOLO-ReT: Towards High Accuracy Real-time Object Detection on Edge GPUs’](https://arxiv.org/abs/2110.13713)
- [**Scaled-Yolo v4**](https://github.com/WongKinYiu/ScaledYOLOv4) (2021) Chien-Yao Wang et al. ['Scaled-YOLOv4: Scaling Cross Stage Partial Network'](https://arxiv.org/abs/2011.08036)
- [**YoloX**](https://github.com/Megvii-BaseDetection/YOLOX) (2021) Zheng Ge at all. [‘YOLOX: Exceeding YOLO Series in 2021’](https://arxiv.org/abs/2107.08430). Good for edge devices. YOLOX-Tiny and YOLOX-Nano outperform YOLOv4-Tiny and NanoDet offering a boost of 10.1% and 1.8% respectively
- [**YoloR**](https://github.com/WongKinYiu/yolor) (You Only Learn One) (2021) Chien-Yao Wang et al. [‘You Only Learn One Representation: Unified Network for Multiple Tasks’](https://arxiv.org/abs/2105.04206)
- [**YoloP**](https://github.com/hustvl/YOLOP) (2022-v7) Dong Wu at all. [‘YOLOP: You Only Look Once for Panoptic Driving Perception’](https://arxiv.org/abs/2108.11250). YoloP was designed to perform three visual perception tasks: traffic object detection, drivable area segmentation and lane detection simultaneously in real-time on an embedded device (Jetson TX2, 23 FPS). It is based on one encoder for feature extraction and three decoders to handle the specific tasks

# Object Detection DNN Algorithms Benchmark

- [Real-Time Object Detection on COCO - **Mean Average Precission (MAP)**](https://paperswithcode.com/sota/real-time-object-detection-on-coco) - YOLOR‑D6
- [Real-Time Object Detection on COCO - **Speed FPS**](https://paperswithcode.com/sota/real-time-object-detection-on-coco?metric=FPS) - YOLOv4-CSP CD53s 640
- [Real-Time Object Detection on COCO - **Inference Time**](https://paperswithcode.com/sota/real-time-object-detection-on-coco?metric=inference%20time%2C%20ms) - YOLOv4-CSP-P6

# Tests and comparisons of models
[![**YoloR vs YoloX**](https://img.youtube.com/vi/Qm3GTj2I_Kk/0.jpg)](https://www.youtube.com/watch?v=Qm3GTj2I_Kk)
[![**Yolo_v5 vs YoloX**](https://img.youtube.com/vi/V6wIxnfOJCs/0.jpg)](https://www.youtube.com/watch?v=V6wIxnfOJCs)
[![**YoloX**](https://img.youtube.com/vi/m7yRGpjiatM/0.jpg)](https://www.youtube.com/watch?v=m7yRGpjiatM)

