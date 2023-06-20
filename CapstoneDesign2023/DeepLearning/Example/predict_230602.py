# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python segment/predict.py --weights yolov5s-seg.pt --source 0                               # webcam
                                                                  img.jpg                         # image
                                                                  vid.mp4                         # video
                                                                  screen                          # screenshot
                                                                  path/                           # directory
                                                                  list.txt                        # list of images
                                                                  list.streams                    # list of streams
                                                                  'path/*.jpg'                    # glob
                                                                  'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python segment/predict.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg_openvino_model     # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                          yolov5s-seg_paddle_model       # PaddlePaddle
"""

import argparse  # 인자를 받아오는 라이브러리 Library that get some options
import os
import platform  # 시스템이 리눅스인지 윈도우인지 Whether system is Linux or Windows
import sys
import time
from pathlib import Path  # 파일 디렉토리 조작 control file directory
import numpy as np
import torch
import math

FILE = Path(__file__).resolve()  # __file__ : 현재 실행 중인 파일 active file, resolve() : 절대경로 absolute directory
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend  # model folder - common.py, 316 line
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, \
    LoadStreams  # var, var, class, class, class
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode

# 여기서부터 추가한 모듈
import easyocr  # OCR 모델
import pyrealsense2 as rs  # realsense
import cv2
from threading import Thread
from utils.augmentations import letterbox  # letterbox
import numpy as np


class LoadStreams_Realsense:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, pipeline_, img_size=640, stride=32, auto=True):
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.imgs, self.fps, self.frames, self.threads = [None], [0], [0], [None]
        self.pipeline = pipeline_

        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()  # depth frame 객체
        color_frame = frames.get_color_frame()  # color frame 객체
        depth_image = np.asanyarray(depth_frame.get_data())  # frame데이터를 행렬화 시켜줌.
        color_image = np.asanyarray(color_frame.get_data())
        h, w, _ = color_image.shape

        self.imgs[0] = color_image
        self.threads[0] = Thread(target=self.update, daemon=True)  # 아하 이것때문에 계속 도는거구나
        self.threads[0].start()
        # check for common shapes
        s = np.stack([letterbox(x, img_size, stride=stride, auto=auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        self.auto = auto and self.rect
        if not self.rect:
            LOGGER.warning('WARNING ⚠️ Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self):  # Read stream `i` frames in daemon thread
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()  # depth frame 객체
                color_frame = frames.get_color_frame()  # color frame 객체
                if not depth_frame or not color_frame:
                    continue
                depth_image = np.asanyarray(depth_frame.get_data())  # frame데이터를 행렬화 시켜줌.
                color_image = np.asanyarray(color_frame.get_data())
                self.imgs[0] = color_image
                '''  # 두 개의 창(컬러, 뎁스)을 합치는 코드
                test = cv2.convertScaleAbs(depth_image, alpha=0.03)
                test2 = np.where(test > 0, 150, 0)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = color_image.shape
                if depth_colormap_dim != color_colormap_dim:
                    resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                                     interpolation=cv2.INTER_AREA)
                    images = np.hstack((resized_color_image, depth_colormap))
                else:
                    images = np.hstack((color_image, depth_colormap))
                '''
        finally:
            self.pipeline.stop()

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        im0 = self.imgs.copy()
        im = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0] for x in im0])  # resize
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        im = np.ascontiguousarray(im)  # contiguous
        return 0, im, im0, None, ''

    def __len__(self):
        return 1


def init_realsense():
    pipeline = rs.pipeline()  # pipeline클래스는 user interaction with the device가 잘 이루어지게 만들어짐.
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)  # pipeline을 랜더링하기위해 알맞은 형태로 변환
    pipeline_profile = config.resolve(pipeline_wrapper)
    device_ = pipeline_profile.get_device()
    print(device_)
    device_product_line = str(device_.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device_.sensors:  # s => pipeline의 device정보 객체임.
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break

    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == "L500":  # 이게 뭔지
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    return pipeline


def visualize_text_bbox(result_, label_masks_, apply=True):
    if not apply:
        return None

    for i in range(len(result_)):
        for j in range(4):
            try:
                cv2.line(label_masks_, tuple(result_[i][0][j]), tuple(result_[i][0][j - 1]), (255, 0, 0), 2)
            except:
                cv2.line(label_masks_, tuple(int(k) for k in result_[i][0][j]),
                         tuple(int(k) for k in result_[i][0][j - 1]),
                         (255, 0, 0), 2)

    cv2.imshow('label_mask', label_masks_)


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s-seg.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/predict-seg',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
):
    # args
    no_bbox_mask = True  # 화면에서 bbox, mask 제거
    # source = '/home/kdh/Downloads/20230509_151835.mp4'
    weights = '/home/kdh/PycharmProjects/condaEnv/yolov5-master/runs/train-seg/exp0505/best.pt'
    nosave = True
    iou_thres = 0.15
    view_img = True  # cv2 imshow 하는 부분
    realsense = True  # Realsense로 영상처리
    pipeline = webcam = screenshot = None  # To solve local variable error

    if realsense:
        source = None
        pipeline = init_realsense()
    else:
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:  # dataset : 웹캠 화면 혹은 저장된 이미지 webcam screen or saved image
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt,
                              vid_stride=vid_stride)  # segment - dataloaders.py
        bs = len(dataset)  # 1이 반환됨. return 1
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    elif realsense:  # for using realsense depth camera
        dataset = LoadStreams_Realsense(pipeline_=pipeline, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference  # 웜업 실행 warmup : check whether the model can do forward and execute forward
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup   # parser : imgsz=(1, 3, 640, 640)
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())  # Profile : Inference time(gpu)
    for path, im, im0s, vid_cap, s in dataset:  # __next__에서 return 한 값들
        with dt[0]:  # with 용법 - Profile을 각각 실행한 효과 execute Profile for each element in dt
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:  # color channels 3 -> initialize?
                im = im[None]  # expand for batch dim
            # 3 dt Thread(dt[0], dt[1], dt[2]) to execute 3 Threads
        # Inference
        with dt[1]:  # print(pred.shape, proto.shape) # torch.Size([1, 18900, 117]) torch.Size([1, 32, 120, 160])
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, proto = model(im, augment=augment, visualize=visualize)[:2]  # proto.shape = [1, 32, 120, 160]
            # pred.shape = torch.Size([1, 18900, 117])
            # model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half) - models-common.py
        # NMS - 일정 이상 겹치는 ROI 제거 remove same labeled ROI that intersect each other over the limit proportion
        with dt[2]:  # nm : number of mask
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Process predictions
        for i, det in enumerate(pred):  # per image # print(det.shape)=[물체 수, 38]
            seen += 1
            if webcam or realsense:  # batch_size >= 1
                if webcam:
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count  # im0s : 원본 이미지
                else:
                    p, im0, frame = 0, im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            if realsense:
                p = '/home'
            else:
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop # imc : image copy 줄임말
            imd = im0.copy()
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):  # print(im.shape) = [1,3,480,640]
                if retina_masks:
                    # scale bbox first the crop masks
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                    masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                else:  # 프린트 해본 결과 여기로 들어감 default - checked by webcam execution
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                    # process_mask location : utils-segment-general.py
                # Segments
                if save_txt:
                    segments = [
                        scale_segments(im0.shape if retina_masks else im.shape[2:], x, im0.shape, normalize=True)
                        for x in reversed(masks2segments(masks))]

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string   # 물체 이름 출력해주는 칸

                # Mask plotting
                if save_img or save_crop or view_img:  # not Add bbox to image
                    annotator.masks(  # utils-plots.py
                        masks,
                        colors=[colors(x, True) for x in det[:, 5]],
                        im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(
                            0).contiguous() /
                               255 if retina_masks else im[i])

                # Write results   # 저장
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    if save_txt:  # Write to file
                        seg = segments[j].reshape(-1)  # (n,2) to (n*2)
                        line = (cls, *seg, conf) if save_conf else (cls, *seg)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image  # 주석0420
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                k_ = cv2.waitKey(1) & 0xFF
                if k_ == ord('q'):
                    im0_ = imd.copy()
                    det_ = det
                    masks_ = masks
                    names_ = names
                    capstone_find_book(im0_, det_, masks_, names_, find_word='6229')
                    cv2.waitKey(0)
                elif k_ == 27:
                    exit()

            # Save results (image with detections)
            if save_img and not realsense:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # capstone_find_book(im0, det, masks, names, find_word='579358') ############# Capstone Design ##############

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img and not realsense:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update and not realsense:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    if dataset.mode == 'image':
        capstone_find_book(imd, det, masks, names, find_word='1318')  # 추가
        # cv2.imshow('img_', im0)
        cv2.waitKey(0)


'''
def capstone_find_book(im0, det, masks, names, find_word='579358'):
    if len(det)<2:  # 물체가 검출되지 않았을 때
        print('not_detected')
        return None
    book_label = cv2.bitwise_or(masks[0].cpu().numpy(), masks[1].cpu().numpy())
    cv2.imshow('img,', book_label)
'''


def capstone_find_book(im0, det, masks, names, find_word='579358'):
    if not len(det):  # 물체가 검출되지 않았을 때
        print('not_detected')
        return None

    label_masks = np.array([])  # to solve local variable error

    visual = False  # 일단 False일 때 동작하도록 해놨습니다

    try:
        ret = [None, np.array([])]  # 나중에 여기다 정리할 것임!!  # ex) [conf mode, mask(book+label or label), ...]
        ret_ = []

        """
        cls_list : 검출결과인 'det'에 있는 5번째 원소(class)를 저장하는 리스트. ex) [0, 1, 0] = [책, 라벨, 책]
        book_len, label_len : cls_list 안에 있는 책(0)의 수, 라벨(1)의 수
        mask_len : 책의 수 + 라벨 수
        
        label_list : 라벨지 관련 정보 저장. 각각의 라벨지에 대해 [마스크 중심점 x좌표, 마스크 중심점 y좌표, mask의 인덱스] 저장
        book_list : 책 관련 정보 저장. 각각의 책에 대해 [마스크 중심점 x좌표, 마스크 중심점 y좌표, mask의 인덱스] 저장
        
        label_idx : 라벨의 인덱스 저장
        book_idx : 책의 인덱스 저장
        
        avg_conf : 전체 물체에 대한 평균적인 confidence 레벨을 저장
        """

        cls_list = [det[i, 5] for i in range(len(det[:, 5]))]  # names = {0: 'Books', 1: 'Labels'}
        origin_h, origin_w, _ = im0.shape  # HWC(height, width, channel)

        img_gray = cv2.cvtColor(im0.copy(), cv2.COLOR_BGR2GRAY)  # make gray scale

        book_len, label_len = cls_list.count(0), cls_list.count(1)  # length of each label
        mask_len = len(masks)  # masks list length

        label_list = np.zeros((label_len, 3), dtype=int)
        book_list = np.zeros((book_len, 3), dtype=int)
        label_idx = np.zeros((mask_len,), dtype=int)
        book_idx = np.zeros((mask_len,), dtype=int)

        book_cnt = 0
        label_cnt = 0

        avg_conf = sum(det[:, 4]) / len(det[:, 4])  # detect_mode = {0:'train', 1:'unseen'}  # mode selection

        ####################### 1번 케이스 : 책에 대한 confidence 레벨이 높음 ############################
        if avg_conf >= 0.8:  # detect_mode == 0:  # 만약 학습된 책이면
            find_book = []
            find_label = []

            book_angles = np.zeros((book_len,), dtype=float)  # book_angle : bbox와 mask로 찾은 책의 orientation angle

            ########################### 1-1 책과 라벨을 각도를 이용해 매칭시키는 과정 ##############################
            # 1-1-1 먼저 책에 대한 마스크 중점을 찾기
            for i in range(mask_len):  # 마스크 각각의 컨투어를 찾는 코드
                mask_0 = masks[i].byte().cpu().numpy()   # 마스크를 cpu로 가져옴
                mask_0 = cv2.resize(mask_0, (origin_w, origin_h))  # 마스크를 원본 이미지 크기로 변환
                contours = cv2.findContours(mask_0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0] # 컨투어 찾기
                contours = sorted(list(contours), key=len, reverse=True)  # 컨투어를 크기(길이) 내림차순 정렬
                mmt = cv2.moments(contours[0]) # 가장 큰 컨투어의 모멘트 찾기
                mask_cx = int(mmt['m10'] / mmt['m00'])  # 마스크 중점 x
                mask_cy = int(mmt['m01'] / mmt['m00'])  # 마스크 중점 y

                if names[int(cls_list[i])] == 'Books': # 만약 책의 마스크라면
                    xy_list = det[i, :4]  # 책의 bbox들을 저장한 리스트
                    bbox_cx = int((xy_list[0] + xy_list[2]) / 2)  # bbox의 중심 x
                    bbox_cy = int((xy_list[1] + xy_list[3]) / 2)  # bbox의 중심 y

                    book_angles[book_cnt] = math.degrees(math.atan2(bbox_cy - mask_cy, bbox_cx - mask_cx)) # 책의 각도 저장
                    book_list[book_cnt] = [mask_cx, mask_cy, book_cnt] # book_list에 [책 중점 x좌표, 책 중점 y좌표, 인덱스] 저장
                    book_idx[book_cnt] = i # 이 책이 mask의 몇 번째 원소인지를 저장(이후 역순으로 책의 마스크 찾아올 때 사용)

                    book_cnt += 1 # 0부터 시작. 책 검출할 때마다 +1
                else:  # 만약 라벨지의 마스크라면
                    label_list[label_cnt] = [mask_cx, mask_cy, label_cnt] # label_list에 [마스크 중점 x좌표, y좌표, 인덱스] 저장
                    label_idx[label_cnt] = i  # 이 책이 mask의 몇 번째 원소인지 저장(이후 역순으로 라벨지 마스크 찾아올 때 사용)

                    # 아래는 라벨지 마스크들을 합쳐서 나중에 글자 읽어올 때 사용
                    if label_cnt:  # 두 번째 이상으로 검출한 라벨지이면 이전 라벨지와 마스크를 합침(bitwise_or)
                        label_masks = cv2.bitwise_or(mask_0, label_masks)
                    else:  # 첫 번째 검출한 라벨지면 그대로 저장
                        label_masks = mask_0.copy()
                    label_cnt += 1  # 0부터 시작. 라벨지를 검출할 때마다 +1

            label_masks = label_masks * img_gray  # 합쳐진 라벨지의 마스크에 원본 이미지를 씌워 저장(나중에 글자 읽을 때 사용)

            # 1-1-2 각도를 이용한 책과 라벨지의 매칭(각도차의 최솟값 찾는 과정)
            label_angle_idx = np.full((book_len,), 100, dtype=float)
            # 인덱스를 100으로 초기화함. 라벨지와 책의 중점의 각도를 저장하여, 나중에 책과 라벨지 매칭에 사용
            min_angles = np.zeros((book_len,), dtype=float)

            # 책과 라벨지 매칭 과정 : 책의 각도를 알고 있기 때문에, 각각의 책 중심점과 라벨지 중심점 간의 각도를 찾고, 그를 이용해 책과 라벨지 매칭
            for i, angle in enumerate(book_angles): # 책의 각도만큼 반복, [라벨지와 책 사이 각도]가 [책 각도]와 가장 비슷한 라벨지 찾기
                temp_label_idx = -1 # 초기화 -> 나중에 디버깅할 때 바꿀 것임
                min_angle = 100  # |[라벨지와 책 사이 각도] - [책 각도]|의 최솟값 찾기 위한 변수 -> 나중에 np.inf로 바꾸기

                for j, label_cp in enumerate(label_list):  # 책과 라벨의 최소 각도 구하는 코드 - 매칭시키기 위함
                    temp_angle_ = np.rad2deg(np.arctan2(label_list[j][1] - book_list[i][1],
                                                        label_list[j][0] - book_list[i][0]))
                    # [라벨지와 책 사이 각도] 저장

                    temp_angle0 = temp_angle_ - angle  # [라벨지와 책 사이 각도] - [책 각도]
                    temp_angle1 = temp_angle_ + angle  # [라벨지와 책 사이 각도] + [책 각도]

                    temp_angle = min(abs(temp_angle0), abs(temp_angle1))  # 위 두 값 중 최솟값
                    if abs(temp_angle) < min_angle:  # temp_angle의 최솟값 찾기
                        min_angle = abs(temp_angle)
                        temp_label_idx = j
                min_angles[i] = min_angle  # 찾은 최솟값을 저장
                label_angle_idx[i] = temp_label_idx  # 책과 매칭된 라벨지의 원본 마스크 리스트에서의 인덱스를 저장 -> 나중에 꺼내오기 위함

            # 1-1-3 디버깅(이상한 책과 매칭되거나, 책의 마스크가 검출되지 않는 경우를 해결)
            for i in range(len(label_angle_idx)):  # 각도 추출한 인덱스의 중복 제거하는 부분 -> 책이 안나오거나 오검출 방지 코드
                temp_list = np.where(label_angle_idx == i)
                if len(temp_list[0]) >= 2:
                    if min_angles[int(temp_list[0][0])] > min_angles[int(temp_list[0][1])]:
                        for j in range(len(label_angle_idx)):
                            if j not in label_angle_idx:
                                label_angle_idx[int(temp_list[0][0])] = j
                                break
                    else:
                        for j in range(len(label_angle_idx)):
                            if j not in label_angle_idx:
                                label_angle_idx[int(temp_list[0][1])] = j
                                break

            # 1-1-4 라벨 -> 책 인덱스 복사하기(매칭)
            # 라벨지 리스트 [중점x, 중점y, 라벨지 인덱스]
            # 책 리스트 [중점x, 중점y, 책 인덱스]
            # 매칭 시 [라벨지 인덱스 -> 책 인덱스]로 복사해오기
            for idx, item in enumerate(label_angle_idx):  # 라벨의 인덱스 label[2]를 책의 인덱스 book[2]에 복사해오기(매칭)
                book_list[idx][2] = item  # 이후에 라벨지의 인덱스를 통한 책의 검출을 위함

            ''' # print for debugging 오류 발생 시 이 리스트를 확인하여 수정할 수 있음
            print(min_angles)
            print(label_angle_idx)
            print(book_list)
            '''

            ########################### 1-2 글자를 읽어서 다시 책의 마스크 찾는 과정 ##############################
            # 1-2-1 글자 읽기
            reader = easyocr.Reader(['en'])  # 리더기 세팅. 한글도 적용하고 싶으면 ['ko', 'en']. 한글은 자음만 따로 인식 안돼서 안씀
            result = reader.readtext(label_masks, slope_ths=0.3)  # 세팅한 리더기로 읽기. slope_ths : 기울어진 글자도 읽기
            temp_idx = 0

            # 1-2-2 읽은 글자 중 find_word(찾는 글자)가 있는지 판단, 있으면 해당 책의 마스크 반환
            for i in range(len(result)):  # 읽은 글자 리스트만큼 반복
                if find_word in result[i][1]:  # 만약 원하는 글자를 찾으면
                    # print('find-detect mode 0')
                    x_ = int((result[i][0][0][0] + result[i][0][2][0]) / 2)
                    y_ = int((result[i][0][0][1] + result[i][0][2][1]) / 2)  # x_, y_ : 찾은 글자 bbox의 중점
                    temp_min = 1E6  # 최솟값 찾기 위한 임시 최솟값 세팅 (1백만)
                    for j, item in enumerate(label_list):  # 글자와 라벨지 간의 최소 거리로 글자에 해당하는 라벨지 찾기
                        dist_ = (item[0] - x_) ** 2 + (item[1] - y_) ** 2
                        if int(dist_) < temp_min:
                            temp_min = int(dist_)
                            temp_idx = j


                    if visual : # 결과창 보여주기 여부 visualize
                        find_label = cv2.resize(masks[label_idx[temp_idx]].byte().cpu().numpy(), (origin_w, origin_h))
                        # cv2.putText(find_label, result[i][1], (x_, y_-50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)
                        cv2.putText(find_label, 'target : ' + str(find_word), (origin_w - 800, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                        cv2.putText(find_label, 'read : ' + result[i][1], (origin_w - 800, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (255, 255, 255), 3)
                        cv2.circle(find_label, (x_, y_), 5, (0, 0, 0), -1)
                    else :
                        find_label = cv2.resize(masks[label_idx[temp_idx]].cpu().numpy(), (origin_w, origin_h))
                    # 찾은 라벨지 저장

                    # cv2.imshow('find_label', find_label)
                    for idx_, item in enumerate(book_list):  # 마스크 사이즈를 원본 사이즈에 맞게 resize
                        if item[2] == temp_idx:  # 라벨지에 매칭되는 책 찾기
                            if visual :
                                find_book = cv2.resize(masks[book_idx[idx_]].byte().cpu().numpy(),
                                                   (origin_w, origin_h))
                            else :
                                find_book = cv2.resize(masks[book_idx[idx_]].cpu().numpy(),
                                                       (origin_w, origin_h))
                            '''
                            find_book = cv2.resize(masks[book_idx[idx_]].byte().cpu().numpy(), (origin_w, origin_h))
                            find_book = find_book * img_gray
                            '''
                            # cv2.imshow('find_book', find_book)
                            break

            book_label_mask = cv2.bitwise_or(find_book, find_label)
            ret[0] = 0
            ret[1] = book_label_mask

            if visual :
                book_label_mask = book_label_mask * img_gray
                cv2.imshow('book+label', book_label_mask)
                visualize_text_bbox(result, label_masks, apply=False)  # apply=True하면 q 눌렀을 때 글자 바운딩 박스 보여줌

            # find_label : 글자 인식을 통해 찾은 라벨지 마스크를 저장하고 있는 변수
            # find_book : 글자 인식을 통해 찾은 책 마스크를 저장하고 있는 변수
            # book_label_mask : 책 마스크 + 라벨 마스크
            else :
#                cv2.imshow('re', ret[1])
                pass
        ####################### 2번 케이스 : 책에 대한 confidence 레벨이 낮음 ############################
        else:  # avg_conf < 0.8 => detect_mode == 1 (unseen) :
            # 2-1 라벨지만 따로 마스크를 떼오는 코드
            label_cnt = 0
            mask_cp = []
            for i in range(mask_len):
                if names[int(cls_list[i])] == 'Books':
                    continue  # 라벨지에 대해서만 for문 진행
                mask_0 = masks[i].byte().cpu().numpy()
                mask_0 = cv2.resize(mask_0, (origin_w, origin_h))
                # mask_0 = mask_0 * img_gray

                contours = cv2.findContours(mask_0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
                contours = sorted(list(contours), key=len, reverse=True)
                mmt = cv2.moments(contours[0])
                mask_cx = int(mmt['m10'] / mmt['m00'])
                mask_cy = int(mmt['m01'] / mmt['m00'])  # 마스크의 중점

                mask_cp.append([mask_cx, mask_cy])

                label_idx[label_cnt] = i  # 나중에 글자 찾은 마스크 찾을 때
                if label_cnt:  # 라벨지 마스크들을 다 합치는 코드
                    label_masks = cv2.bitwise_or(mask_0, label_masks)
                else:
                    label_masks = mask_0.copy()

                label_cnt += 1

            label_masks = label_masks * img_gray  # 라벨지의 마스크들을 저장하고 있는 리스트

            # 2-2. 라벨 마스크에서 글자 찾아서 그 라벨지 자체를 가져오기
            reader = easyocr.Reader(['en'])  # 한글도 적용하고 싶으면 ['ko', 'en']. 다만 여기선 자음만 인식 안돼서 안씀
            result = reader.readtext(label_masks, slope_ths=0.3)
            find_mask = []
            print(result)
            for i in range(len(result)):
                # print(result[i])
                if find_word in result[i][1]:
                    print('find-unseen')
                    x_ = int((result[i][0][0][0] + result[i][0][2][0]) / 2)
                    y_ = int((result[i][0][0][1] + result[i][0][2][1]) / 2)
                    temp_min = 1E6;
                    temp_idx = 1
                    for j, item in enumerate(mask_cp):
                        dist_ = (item[0] - x_) ** 2 + (item[1] - y_) ** 2
                        if int(dist_) < temp_min:
                            temp_min = int(dist_)
                            temp_idx = j

                    if visual :
                        find_mask = cv2.resize(masks[label_idx[temp_idx]].byte().cpu().numpy(), (origin_w, origin_h))
                    else :
                        find_mask = cv2.resize(masks[label_idx[temp_idx]].cpu().numpy(), (origin_w, origin_h))

                    ret[0] = 1
                    ret[1] = find_mask
                    ret_ = find_mask

                    if visual :
                        cv2.circle(find_mask, (x_, y_), 5, (0, 0, 0), -1)
                        # cv2.putText(find_mask, result[i][1], (x_, y_ - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                        cv2.putText(find_mask, 'target : ' + str(find_word), (origin_w - 800, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                        cv2.putText(find_mask, 'read : ' + result[i][1], (origin_w - 800, 150), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                    (255, 255, 255), 3)
                        visualize_text_bbox(result, label_masks, apply=False)  # apply=True하면 q 눌렀을 때 글자 바운딩 박스 보여줌
                        cv2.imshow('find_mask', find_mask * img_gray)

        cv2.imshow('re', ret[1])

    except Exception as e:
        print(e)


def parse_opt():
    parser = argparse.ArgumentParser()  # 인자 받아오는 객체
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
