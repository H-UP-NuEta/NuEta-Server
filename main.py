import io
import zipfile
import torch
import cv2
import numpy as np
from typing import List, Union
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

app = FastAPI()

def load_model_custom(model_name: str):
    model_path = f"./weight/best_{model_name}.pt"
    model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)
    return model

# 서버 시작 시 모델 로드
car_plate_model = load_model_custom("car_plate")

# 모델 초기화
net = cv2.dnn.readNetFromCaffe(
    './weight/deploy.prototxt',
    './weight/res10_300x300_ssd_iter_140000.caffemodel'
)

# 이미지 파일을 불러오는 함수
def load_image(image_path):
    return cv2.imread(image_path)

def read_bboxes(df):
    bboxes = df[["xmin", "ymin", "xmax", "ymax"]].values.tolist()
    return bboxes

# 모자이크 처리 함수
def apply_mosaic(img, bbox, mosaic_size=15):
    x1, y1, x2, y2 = bbox
    roi = img[y1:y2, x1:x2]  # 관심영역(Region of Interest) 추출
    roi = cv2.resize(roi, (mosaic_size, mosaic_size), interpolation=cv2.INTER_LINEAR)
    roi = cv2.resize(roi, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = roi  # 원본 이미지에 모자이크 처리된 영역을 대체
    return img


def apply_mosaic_to_video(video_path):
    cap = cv2.VideoCapture(video_path)
    # 임시 파일 생성으로 결과 동영상 저장 준비
    temp_file = NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_filename = temp_file.name
    temp_file.close()

    # 동영상 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_filename, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 처리
        results = car_plate_model(frame)
        results_data = results.pandas().xyxy[0]
        bboxes = results_data[["xmin", "ymin", "xmax", "ymax"]].values.tolist()

        for bbox in bboxes:
            bbox = list(map(int, bbox))
            frame = apply_mosaic(frame, bbox, 15)

        out.write(frame)

    cap.release()
    out.release()

    return temp_filename

def read_image_from_memory(image_data):
    image_stream = io.BytesIO(image_data)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/car_plate/images/upload")
async def upload_car_plate_images(files: List[UploadFile] = File(...)):
    if len(files) == 1:  # 이미지가 하나인 경우
        file = files[0]
        image_data = await file.read()
        image = read_image_from_memory(image_data)
        results = car_plate_model(image)
        results_data = results.pandas().xyxy[0]
        bboxes = read_bboxes(results_data)
        for bbox in bboxes:
            bbox = list(map(int, bbox))
            image = apply_mosaic(image, bbox, 15)
        _, encoded_image = cv2.imencode(".jpg", image)
        return StreamingResponse(io.BytesIO(encoded_image), media_type="image/jpeg")

    # 이미지가 여러 개인 경우
    temp_file = NamedTemporaryFile(delete=False, suffix='.zip')
    temp_filename = temp_file.name
    temp_file.close()

    with zipfile.ZipFile(temp_filename, 'w') as zf:
        for file in files:
            image_data = await file.read()
            image = read_image_from_memory(image_data)
            results = car_plate_model(image)
            results_data = results.pandas().xyxy[0]
            bboxes = read_bboxes(results_data)
            for bbox in bboxes:
                bbox = list(map(int, bbox))
                image = apply_mosaic(image, bbox, 15)
            _, encoded_image = cv2.imencode(".jpg", image)
            image_filename = file.filename
            zf.writestr(image_filename, encoded_image.tobytes())

    def iterfile():
        with open(temp_filename, mode="rb") as file_like:
            yield from file_like

    return StreamingResponse(iterfile(), media_type="application/zip")


@app.post("/car_plate/video/upload")
async def upload_car_plate_video(file: UploadFile = File(...)):
    # 동영상 파일을 임시 파일로 저장
    with NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    # 모자이크 처리 적용된 동영상 생성
    output_video_path = apply_mosaic_to_video(temp_file_path)

    # 생성된 동영상을 읽어 StreamingResponse로 반환
    def iterfile():
        with open(output_video_path, mode="rb") as file_like:
            yield from file_like

    return StreamingResponse(iterfile(), media_type="video/mp4")

@app.post("/face/images/upload")
async def upload_face_images(files: List[UploadFile] = File(...)):
    if len(files) == 1:
        return await process_single_image(files[0])
    else:
        return await process_multiple_images(files)

async def process_single_image(file: UploadFile):
    image_data = await file.read()
    image = read_image_from_memory(image_data)
    image = detect_and_mosaic_faces(image)
    _, encoded_image = cv2.imencode(".jpg", image)
    return StreamingResponse(io.BytesIO(encoded_image), media_type="image/jpeg")

async def process_multiple_images(files: List[UploadFile]):
    temp_file = NamedTemporaryFile(delete=False, suffix='.zip')
    with zipfile.ZipFile(temp_file.name, 'w') as zf:
        for file in files:
            image_data = await file.read()
            image = read_image_from_memory(image_data)
            image = detect_and_mosaic_faces(image)
            _, encoded_image = cv2.imencode(".jpg", image)
            zf.writestr(file.filename, encoded_image.tobytes())
    return StreamingResponse(open(temp_file.name, 'rb'), media_type="application/zip")

def detect_and_mosaic_faces(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            image = apply_mosaic(image, [startX, startY, endX, endY])
    return image