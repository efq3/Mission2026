from ultralytics import YOLO

# 1. 모델 불러오기 
model_path = "best.pt"
model = YOLO("best.pt")

source = "0"                                        # (3) 노트북 기본 웹캠으로 실시간 테스트할 때?

# 3. 실전 테스트 시작 (GPU 사용, 화면에 띄우기)
print("실전 테스트를 시작합니다 (웹캠 창이 뜨면 'q'를 눌러서 종료하세요
results = model.predict(source=source, show=True, save=False, device=0)