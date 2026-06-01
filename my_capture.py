import cv2
import threading
import os

# 공유 변수 및 안전장치(Lock)
shared_frame = None
lock = threading.Lock()
is_running = True

# 1. 백그라운드에서 카메라 영상만 계속 읽어오는 일꾼 (스레드)
def webcam_thread():
    global shared_frame, is_running
    
    # 💡 리눅스 USB 웹캠 인덱스 (기본 0, 안 나오면 1이나 2로 변경)
    cap = cv2.VideoCapture(2) 
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        is_running = False
        return

    while is_running:
        ret, frame = cap.read()
        if ret:
            with lock:
                shared_frame = frame.copy() # 최신 프레임을 공유 변수에 복사

    cap.release()

# 2. 메인 실행 함수 (화면 표시 및 키보드 입력 처리)
def run_threaded():
    global is_running
    
    # 저장할 폴더 설정
    save_directory = "img_capture_threaded"
    os.makedirs(save_directory, exist_ok=True)
    image_count = 0

    # 카메라 스레드 가동
    is_running = True
    t = threading.Thread(target=webcam_thread)
    t.start()
    print("멀티스레드 카메라 고속 모드 시작... (종료: q, 캡처: c)")

    while is_running:
        # 안전하게 백그라운드에서 최신 프레임 훔쳐오기
        with lock:
            frame = shared_frame.copy() if shared_frame is not None else None

        if frame is not None:
            # 원본 화면 그대로 출력 (기존의 무거운 가우시안 블러 연산 제거)
            cv2.imshow("Threaded Webcam", frame)

        # 키보드 입력 대기
        key = cv2.waitKey(1) & 0xFF
        
        # 'c'를 누르면 이미지 저장
        if key == ord('c') and frame is not None:
            file_name = f"{save_directory}/img_{image_count}.jpg"
            cv2.imwrite(file_name, frame)
            print(f"📷 사진이 저장되었습니다: {file_name}")
            image_count += 1
            
        # 'q'를 누르면 종료
        elif key == ord('q'):
            is_running = False
            break

    # 안전하게 프로그램 종료 및 리소스 해제
    t.join()
    cv2.destroyAllWindows()
    print("프로그램이 정상 종료되었습니다.")

if __name__ == "__main__":
    run_threaded()
