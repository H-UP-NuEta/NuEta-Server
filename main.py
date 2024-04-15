from typing import Union

from fastapi import FastAPI, File, UploadFile, status, HTTPException, Query
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import torch
import cv2
import numpy as np
from tempfile import NamedTemporaryFile

app = FastAPI()

def load_model_custom(model_name: str):
    model_path = f"./best_{model_name}.pt"  # 현재 작업 디렉토리를 기준으로 상대 경로 사용
    model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)
    return model

# 서버 시작 시 모델 로드
car_plate_model = load_model_custom("car_plate")
face_model = load_model_custom("face")

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


@app.post("/car_plate/image/upload")
async def upload_car_plate_image(file: UploadFile = File(...)):
    # 이미지 파일을 메모리에서 바로 읽기
    image_data = await file.read()
    # image = Image.open(io.BytesIO(image_data))
    image = read_image_from_memory(image_data)

    # 이미지를 모델이 처리할 수 있는 형태로 변환
    results = car_plate_model(image)

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

    cv2.imwrite("car_plate.jpg", image)

    _, encoded_image = cv2.imencode(".jpg", image)

    # 인코딩된 바이트 배열을 io.BytesIO 객체로 변환
    image_stream = io.BytesIO(encoded_image)

    # StreamingResponse 객체 생성 및 반환
    return StreamingResponse(image_stream, media_type="image/jpeg")

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

@app.post("/face/image/upload")
async def upload_face_image(file: UploadFile = File(...)):
    # 이미지 파일을 메모리에서 바로 읽기
    image_data = await file.read()
    # image = Image.open(io.BytesIO(image_data))
    image = read_image_from_memory(image_data)

    # 이미지를 모델이 처리할 수 있는 형태로 변환
    results = face_model(image)

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

    cv2.imwrite("car_plate.jpg", image)

    _, encoded_image = cv2.imencode(".jpg", image)

    # 인코딩된 바이트 배열을 io.BytesIO 객체로 변환
    image_stream = io.BytesIO(encoded_image)

    # StreamingResponse 객체 생성 및 반환
    return StreamingResponse(image_stream, media_type="image/jpeg")