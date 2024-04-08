from typing import Union

from fastapi import FastAPI, File, UploadFile, status, HTTPException, Query
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import torch
import cv2
import numpy as np

app = FastAPI()

# YOLOv5 모델 로드
model = torch.hub.load(
    "ultralytics/yolov5", "custom", path="./best.pt"
)  # 'best.pt' 파일 경로를 지정


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


# 이미지 파일을 불러오는 함수
def load_image(image_path):
    return cv2.imread(image_path)


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


@app.post("/image/upload")
async def upload_image(file: UploadFile = File(...)):
    # 이미지 파일을 메모리에서 바로 읽기
    image_data = await file.read()
    # image = Image.open(io.BytesIO(image_data))
    image = read_image_from_memory(image_data)

    # 이미지를 모델이 처리할 수 있는 형태로 변환
    results = model(image)

    # 결과 처리 (예: 예측된 객체의 종류와 확률)
    results_data = results.pandas().xyxy[0]  # 감지된 객체 정보

    bboxes = read_bboxes(results_data)

    for idx, bbox in enumerate(bboxes):
        bboxes[idx] = list(map(int, bbox))

    print("results_data:", results_data)
    print("bboxes:", bboxes)

    # 모자이크 처리 적용
    # image = load_image(image_data)

    # 각 바운딩 박스에 대해 모자이크 처리 적용
    for bbox in bboxes:
        image = apply_mosaic(image, bbox, 15)

    # 처리된 이미지를 PIL 이미지로 변환
    # final_image = Image.fromarray(cv_image[:, :, ::-1])  # BGR to RGB

    # # PIL 이미지를 바이트로 변환하여 StreamingResponse로 반환
    # img_byte_arr = io.BytesIO()
    # final_image.save(img_byte_arr, format="JPEG")
    # img_byte_arr = io.BytesIO(img_byte_arr.getvalue())

    # return StreamingResponse(, media_type="image/jpeg")

    cv2.imwrite("car_plate.jpg", image)

    _, encoded_image = cv2.imencode(".jpg", image)

    # 인코딩된 바이트 배열을 io.BytesIO 객체로 변환
    image_stream = io.BytesIO(encoded_image)

    # StreamingResponse 객체 생성 및 반환
    return StreamingResponse(image_stream, media_type="image/jpeg")
