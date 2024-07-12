from django.shortcuts import render
# 요청된 템플릿 렌더링 후 http 응답 반환 사용
from django.http import StreamingHttpResponse
import cv2
import mediapipe as mp
import numpy as np
import math

#포즈인식 담당
mp_pose = mp.solutions.pose
#포즈 추정 수행
pose = mp_pose.Pose()
#포즈 랜드마크를 이미지에 그리는 유틸즈
mp_drawing = mp.solutions.drawing_utils

#기본 웹캠을 열어 비디오 캡처 객체 초기화 0 은 기본 웹캠
cap = cv2.VideoCapture(0)

#a,b,c, 각도 계산 함수 (numpy 배열로 변환)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    #계산식
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

squat_count = 0
squat_status = None

def generate_frames():
    #전역 변수인 스쿼트 횟수와 상태 사용
    global squat_count, squat_status

    while True:
        #비디오 실행(프레임 읽기)
        success, frame = cap.read()
        if not success:
            break
        else:
            #비디오 색상 변환 코드
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                #이미지 좌표 그리기
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                landmarks = results.pose_landmarks.landmark
                #엉덩이 무릎 발목의 각도 얻기
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                #각도 계산
                angle = calculate_angle(hip, knee, ankle)
                
                #각도가 90도 미만이면 내려간걸로 인식
                if angle < 90:
                    squat_status = "down"
                    #각도가 160도 이상이고 스쿼트 상태가  down이면  스쿼트 상태 up으로 바꾸고 개수 증가
                if angle > 160 and squat_status == 'down':
                    squat_status = "up"
                    squat_count += 1
                #캠에 글씨 표시 (스쿼트 횟수)
                cv2.putText(image, f'Squat Count: {squat_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #이미지 인코딩 후 스트리밍
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def index(request):
    return render(request, 'index.html')

