import sys
import copy

sys.path.insert(0, 'yolov7')
sys.argv = ['']

import argparse
from pathlib import Path

import torch

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(opt, movies, categories):
    source, weights_coco, weights_custom, view_img, save_txt, imgsz, trace = opt.source, opt.weights1, opt.weights2, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights_coco, map_location=device)  # load FP32 model
    model_custom = attempt_load(weights_custom, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Get names
    names_coco = model.module.names if hasattr(model, 'module') else model.names
    names_custom = model.module.names if hasattr(model_custom, 'module') else model_custom.names

    if categories:
        intersection_categories_coco = set(names_coco) & set(categories)
        intersection_categories_custom = set(names_custom) & set(categories)

        classes_coco = [names_coco.index(name) for name in intersection_categories_coco]
        classes_custom = [names_custom.index(name) for name in intersection_categories_custom]
    else:
        intersection_categories_coco = names_coco
        intersection_categories_custom = names_custom

        classes_coco = classes_custom = opt.classes

    if trace:
        if len(intersection_categories_coco):
            model = TracedModel(model, device, opt.img_size)
        if len(intersection_categories_custom):
            model_custom = TracedModel(model_custom, device, opt.img_size)

    if half:
        if len(intersection_categories_coco):
            model.half()  # to FP16
        if len(intersection_categories_custom):
            model_custom.half()

    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        if len(intersection_categories_coco):
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
        if len(intersection_categories_custom):
            model_custom(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model_custom.parameters())))

    old_img_w = old_img_h = imgsz
    old_img_b = 1

    detection = []

    poster_id_index = -1
    for path, img, im0s, vid_cap in dataset:
        poster_id_index += 1
        if img is None:
            continue

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            old_img_w = img.shape[3]
            for i in range(3):
                if len(intersection_categories_coco):
                    model(img, augment=opt.augment)[0]
                if len(intersection_categories_custom):
                    model_custom(img, augment=opt.augment)[0]

        # Inference
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            if len(intersection_categories_coco):
                pred = model(img, augment=opt.augment)[0]
            if len(intersection_categories_custom):
                pred_custom = model_custom(img, augment=opt.augment)[0]

        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
        p = Path(p)  # to Path
        save_path = '/' + p.name  # img.jpg

        det = []

        if len(intersection_categories_coco):
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=classes_coco,
                                       agnostic=opt.agnostic_nms)
            coco_det = process_detection(pred, path, im0s, dataset,
                                         intersection_categories_coco, img,
                                         names_coco, categories is None)
            if not coco_det:
                continue
            det += coco_det

        if len(intersection_categories_custom):
            pred_custom = non_max_suppression(pred_custom, opt.conf_thres, opt.iou_thres,
                                              classes=classes_custom, agnostic=opt.agnostic_nms)

            custom_det = process_detection(pred_custom, path, im0s, dataset,
                                           intersection_categories_custom, img,
                                           names_custom, categories is None)
            if not custom_det:
                continue
            det += custom_det

        if movies is None:
            return det
        movies[poster_id_index]["det"] = det
        detection.append(movies[poster_id_index])

    if categories:
        detection = sorted(detection, key=lambda x: (max(image_det['conf'] for image_det in x['det'])), reverse=True)
    return detection


def process_detection(pred, path, im0s, dataset, categories, img, names, database):
    detections = []
    must_detect_categories = copy.deepcopy(categories) if categories else None
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
        p = Path(p)  # to Path
        save_path = '/' + p.name  # img.jpg

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                if (not database) and must_detect_categories and names[int(cls)] in must_detect_categories:
                    must_detect_categories.remove(names[int(cls)])
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                detections.append({
                    "model": "yolov7",
                    "label": names[int(cls)],
                    "box": xywh,
                    "conf": float(conf)
                })
    if database or len(must_detect_categories) == 0:
        return detections
    return []


def detect_main(data, movie_ids=None, categories=None, confidence=0.25, device='cpu'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights1', nargs='+', type=str, default='yolov7/yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--weights2', nargs='+', type=str, default='yolov7/custom.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str,
                        default=data,
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=confidence, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default=device, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default="false", action='store_true', help='display results')
    parser.add_argument('--save-txt', default="true", action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', default="true", action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                results = detect(opt, movie_ids, categories)
                strip_optimizer(opt.weights, movie_ids)
        else:
            results = detect(opt, movie_ids, categories)

    return results


def find_labels():
    device = select_device()
    model_coco = attempt_load('yolov7/yolov7.pt', map_location=device)
    model_custom = attempt_load('yolov7/custom.pt', map_location=device) or []
    return model_coco.names + model_custom.names


if __name__ == '__main__':
    detect_main(['https://image.tmdb.org/t/p/w300/qFf8anju5f2epI0my8RdwwIXFIP.jpg',
                 'https://image.tmdb.org/t/p/w300/AeyiuQUUs78bPkz18FY3AzNFF8b.jpg',
                 'https://image.tmdb.org/t/p/w300/rFljUdOozFEv6HDHIFpFvcYW0ec.jpg',
                 'https://image.tmdb.org/t/p/w300/6DrHO1jr3qVrViUO6s6kFiAGM7.jpg',
                 'https://image.tmdb.org/t/p/w300/brrgSFtcymZWaXl5a23GJRWdOSY.jpg',
                 'https://image.tmdb.org/t/p/w300/tVxDe01Zy3kZqaZRNiXFGDICdZk.jpg',
                 'https://image.tmdb.org/t/p/w300/1HOYvwGFioUFL58UVvDRG6beEDm.jpg'])
