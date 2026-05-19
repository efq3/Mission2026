from ultralytics import YOLO

if __name__ == '__main__':
    
    # 모델 불러오기
    model = YOLO("yolo11n-pose.pt")
    
    # workers=0 을 넣어서 멀티프로세싱 충돌 원천 차단
    model.train(
        # 하이퍼 파라미터
        data=".../pose_dataset.yaml", # 데이터셋 경로
        epochs=50, 
        imgsz=640, 
        batch=8, 
        device=0, 
        workers=0 
    )