import asyncio
import datetime
import json
import os
import threading
import threading
import time
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List

import cv2 as cv
import torch
import uvicorn
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, LOGGER, xyxy2xywh, scale_coords
from utils.torch_utils import select_device, time_sync

app = FastAPI()
nc = None
conf_thres = 0.25
iou_thres = 0.45
max_det = 1000
classes = None
agnostic_nms = False
augment = False
visualize = False
save_conf = True

# Load model
device = select_device('cpu')
model = DetectMultiBackend("yolov5s.pt", device=device, dnn=False, data='data/coco128.yaml', fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((640, 640), s=stride)  # check image size


# model = torch.hub.load('./', 'yolov5s', source="local")


def rtsp_frame_handler(tmp_file_path):
    # r = requests.get(img_url, allow_redirects=True)
    # tmp_file_path = f'./tmp/{time.time()}.jpg'
    # open(tmp_file_path, 'wb').write(r.content)

    dataset = LoadImages(tmp_file_path, img_size=imgsz, stride=stride, auto=pt)
    dt, seen = [0.0, 0.0, 0.0], 0

    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # results dict
        inference_results = {}

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # print(f"{n} {det.names[int(c)]}{'s' * (n > 1)}\n")  # add to string

                    inference_results[names[int(c)]] = int(n)

                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    # class x_center y_center width height confident
                    # LOGGER.info(('%g ' * len(line)).rstrip() % line + '\n')

        inference_results = {"inference": inference_results,
                             "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
        LOGGER.info(inference_results)
        asyncio.run(manager.broadcast(str(json.dumps(inference_results))))

    # Print results
    # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # results = model(tmp_file_path)
    # results.print()
    # for i, (im, pred) in enumerate(zip(results.imgs, results.pred)):
    #     if pred.shape[0]:
    #         for c in pred[:, -1].unique():
    #             n = (pred[:, -1] == c).sum()  # detections per class
    #             print(f"{n} {results.names[int(c)]}{'s' * (n > 1)}\n")  # add to string


def cap_and_write(cap, url):
    while True:
        assert cap.isOpened(), f'Failed to open {url}'
        _, im = cap.read()

        file_name = f'{time.time()}.jpg'
        local_path = f'imgs/{file_name}'

        cv.imwrite(local_path, im)
        # print(f'[{threading.current_thread().name}]:Image path {local_path}')
        rtsp_frame_handler(local_path)
        # delete temp file
        os.remove(local_path)
        time.sleep(2)


stream_connections = {}
cap_tasks = ThreadPoolExecutor(max_workers=8)


@app.get("/")
async def root():
    return {"msg": "ok"}


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"You wrote: {data}", websocket)
            await manager.broadcast(f"Client #{client_id} says: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} left the chat")


@app.on_event("startup")
async def startup():
    rtsp_url = ["rtsp://admin:xier123456@192.168.5.66/h264/ch1/sub/av_stream"]
    # create rtsp connections and capture tasks
    for url in rtsp_url:
        print(url)
        cap = cv.VideoCapture(url)
        stream_connections[url] = cap
        task = cap_tasks.submit(cap_and_write, cap=cap, url=url)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
