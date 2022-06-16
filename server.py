import asyncio
import io
import json
import os
import time
import uuid
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List

import cv2 as cv
import hazelcast
import rtsp
import torch
import uvicorn
from fastapi import FastAPI, WebSocket
from hazelcast.core import HazelcastJsonValue
from minio import Minio
from starlette.websockets import WebSocketDisconnect

import minio_helper
from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, LOGGER, xyxy2xywh, scale_coords, check_file
from utils.plots import Annotator, Colors, save_one_box_mem
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
save_crop = True
save_dir = 'imgs'
hide_labels = False
hide_conf = False
line_thickness = 3
name = 'exp'
exist_ok = False
save_txt = False

# Load model
device = select_device('cpu')
model = DetectMultiBackend("yolov5s.pt", device=device, dnn=False, data='data/coco128.yaml', fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((640, 640), s=stride)  # check image size
colors = Colors()

# MinIO Client
minio_host = "127.0.0.1"
minio_port = "9000"
minio_default_bucket = "seeiner-aibox"
minIOClient = Minio(
    f'{minio_host}:{minio_port}',
    access_key="GhHPHJmhk7zDbIaF",
    secret_key="9PR0Ji7DyfqYFRlzcdQpnMPoBE5EON7C",
    secure=False, )
minio_helper.create_bucket(minIOClient)

# Init hazelcast client
hazelcast_client = hazelcast.HazelcastClient()
inference_result_map = hazelcast_client.get_map("inference_result_map")


# model = torch.hub.load('./', 'yolov5s', source="local")


def inference(raw_img_link=None, inference_results=None):
    source = check_file(raw_img_link)
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
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

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # for c in det[:, -1].unique():
                # n = (det[:, -1] == c).sum()  # detections per class
                # print(f"{n} {det.names[int(c)]}{'s' * (n > 1)}\n")  # add to string

                # inference_results[names[int(c)]] = int(n)

                # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    # class x_center y_center width height confident
                    split_res = (('%g ' * len(line)).rstrip() % line).split(" ")
                    element = {"Type": names[int(split_res[0])],
                               "Boxes": [split_res[1], split_res[2], split_res[3], split_res[4]],
                               "Score": split_res[5]}

                    # LOGGER.info(inference_results)
                    if save_crop:
                        c = int(cls)
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                        cropFileName = f'crop_{int(round(time.time() * 1000))}.jpg'
                        crop_in_mem = save_one_box_mem(xyxy, imc, BGR=True)
                        element["CropName"] = cropFileName
                        element["Crop"] = crop_in_mem

                    inference_results["Data"]["Elements"].append(element)

                    # LOGGER.info(('%g ' * len(line)).rstrip() % line + '\n')

    return inference_results

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


def inference_task(client, camera):
    while True:
        try:
            assert client.isOpened(), f'Failed to open {camera["channelUrl"]}'
            # Extract the latest frame from the RTSP stream
            im = client.read(raw=True)

            # Transform to bytes and upload to minio
            img_bytes = cv.imencode('.jpg', im)[1].tobytes()
            file_name = f'raw_{int(round(time.time() * 1000))}.jpg'
            raw_img_link = "http://{minio_host}:{minio_port}/{bucket}/{file}".format(minio_host=minio_host,
                                                                                     minio_port=minio_port,
                                                                                     bucket=minio_default_bucket,
                                                                                     file=file_name)
            minio_helper.put_object(client=minIOClient, fileName=file_name,
                                    data=io.BytesIO(img_bytes))

            cv.imwrite(f'imgs/{file_name}', im)

            # results dict
            inference_id = str(uuid.uuid4())
            inference_results = {"Data": {"Elements": [],
                                          "Width": 640,
                                          "Height": 640,
                                          "Area": camera["geoName"],
                                          "RawImg": raw_img_link,
                                          "Time": int(round(time.time() * 1000))},
                                 "InferenceID": inference_id}

            # inference
            inference_results = inference(f'imgs/{file_name}', inference_results)

            for element in inference_results["Data"]["Elements"]:
                minio_helper.put_object(client=minIOClient, fileName=element["CropName"],
                                        data=io.BytesIO(element["Crop"]))
                element["Crop"] = "http://{minio_host}:{minio_port}/{bucket}/{file}".format(
                    minio_host=minio_host,
                    minio_port=minio_port,
                    bucket=minio_default_bucket,
                    file=element["CropName"])
                element.pop("CropName")

            # to json
            output_json = str(json.dumps(inference_results, ensure_ascii=False))
            LOGGER.info(output_json)

            # put result into queue
            inference_result_map.put(inference_results["InferenceID"], HazelcastJsonValue(output_json), ttl=30)

            # broadcast with websocket
            asyncio.run(manager.broadcast(output_json))

            os.remove(f'imgs/{file_name}')
        except Exception as err:
            print(err)
        time.sleep(10)


stream_connections = {}
cap_tasks = ThreadPoolExecutor(max_workers=8)


@app.get("/")
async def root():
    return {"msg": "ok"}


@app.get("/cv/open/rest/records")
async def records():
    vals = inference_result_map.values().result()
    list = []
    for val in vals:
        list.append(json.loads(str(val)))
    return list


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
    cameras = [{"name": "展厅摄像头-1", "channelUrl": "rtsp://admin:xier123456@192.168.5.66/h264/ch1/sub/av_stream",
                "channelType": "RTSP", "geoName": "展厅区域-1"},
               {"name": "展厅摄像头-2", "channelUrl": "rtsp://admin:xier123456@192.168.5.59/live1.sdp",
                "channelType": "RTSP", "geoName": "展厅区域-2"}
               ]

    # create rtsp connections and capture tasks
    for camera in cameras:
        print(camera["channelUrl"])
        client = rtsp.Client(rtsp_server_uri=camera["channelUrl"], verbose=True)
        task = cap_tasks.submit(inference_task, client=client, camera=camera)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
