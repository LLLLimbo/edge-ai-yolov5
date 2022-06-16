FROM python:3.9
RUN #apt update && apt install -y zip htop screen libgl1-mesa-glx

ADD ./  /edge-ai-yolov5
ADD Arial.ttf /root/.config/Ultralytics/
COPY ./requirements.txt /edge-ai-yolov5/requirements.txt
WORKDIR /edge-ai-yolov5

RUN python -m pip install --upgrade pip
RUN pip uninstall -y torch torchvision torchtext
RUN pip install --no-cache -r requirements.txt
#    torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

#ENV OMP_NUM_THREADS=8

CMD ["python /edge-ai-yolov5/server.py"]
