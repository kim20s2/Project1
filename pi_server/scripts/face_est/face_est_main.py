import cv2
import numpy as np
import os
from hailo_platform import VDevice, HailoSchedulingAlgorithm
from .bbox import SCRFDDecoder, debug_bbox_sizes
from .face_logger import PoseDataLogger
from functools import partial

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# output_metrix = ['positive', 'neutral', 'negative']
output_metrix = ['angry', 'disgust', 'happy', 'neutral', 'surprise']
# output_metrix = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
hef_path3 = os.path.join(base_path, 'models', 'scrfd_500m.hef')  # 얼굴

output_names = ['scrfd_500m/conv27','scrfd_500m/conv26', 'scrfd_500m/conv25', 'scrfd_500m/conv33', 
'scrfd_500m/conv32', 'scrfd_500m/conv34', 'scrfd_500m/conv39', 
'scrfd_500m/conv38', 'scrfd_500m/conv40']

extract_list = ['scrfd_500m/conv26','scrfd_500m/conv27', 'scrfd_500m/conv32', 
'scrfd_500m/conv33', 'scrfd_500m/conv38', 
'scrfd_500m/conv39']

extract_name = ['conv26','conv27', 'conv32', 
'conv33', 'conv38', 'conv39']

extract_dic = {'conv26': np.random.randint(0, 255, (80, 80, 2), dtype=np.uint8),
        'conv27': np.random.randint(0, 255, (80, 80, 8), dtype=np.uint8),
        'conv32': np.random.randint(0, 255, (40, 40, 2), dtype=np.uint8),
        'conv33': np.random.randint(0, 255, (40, 40, 8), dtype=np.uint8),
        'conv38': np.random.randint(0, 255, (20, 20, 2), dtype=np.uint8),
        'conv39': np.random.randint(0, 255, (20, 20, 8), dtype=np.uint8),
    }

pos_log = PoseDataLogger(fps=15)

timeout_ms = 1000

def bbox_callback(completion_info, bindings):
    if completion_info.exception:
        # handle exception
        pass
        
    _ = bindings.output().get_buffer()

# 추후 NPU 추론으로 바꿀 수도 있음
def face_detect(frame, vdevice=None):
    pp_frame = preprocess(frame, (640, 640))  # SCRFD 모델 입력 크기

    infer_model = vdevice.create_infer_model(hef_path3)
    # 입출력 정보 확인
    input_shape = infer_model.input().shape
    # print(f"bbox input: {input_shape}")

    output_shape = []
    output_len = len(infer_model.output_names)

    for i in range(output_len):
        output_shape.append(infer_model.output(output_names[i]).shape)
        
    # print(f"bbox output: {output_shape}")
        
    # 모델 설정
    with infer_model.configure() as configured_infer_model:
        bindings = configured_infer_model.create_bindings()

        # 입력 버퍼 설정
        bindings.input().set_buffer(pp_frame)

        # 출력 버퍼 준비
        for i in range(output_len):
            shape = output_shape[i]
            output_buffer = np.empty(shape, dtype=np.uint8 )
            bindings.output(output_names[i]).set_buffer(output_buffer)
        
        configured_infer_model.wait_for_async_ready(timeout_ms)
        # 비동기 추론
        job = configured_infer_model.run_async([bindings], partial(bbox_callback, bindings=bindings))
        
        job.wait(timeout_ms)

        # 결과 가져오기
        for i in range(len(extract_list)):
            extract_dic[extract_name[i]] = bindings.output(extract_list[i]).get_buffer()

    decoder = SCRFDDecoder(
        input_size=640,
        original_size=(640, 480),
        distance_scale=BBOX_SIZE
    )

    bboxes, scores = decoder.detect(
        extract_dic, 
        conf_threshold=0.51,
        nms_threshold=0.4,
    )

    # print(f"face_num: {len(bboxes)}")

    if len(scores) > 0:
        f_index = np.argmax(scores)
    else:
        return None
    # debug_bbox_sizes(bboxes)

    # img = frame.copy()

    # x1, y1, x2, y2 = bboxes[f_index].astype(int) if f_index >= 0 else (0, 0, 0, 0)
    # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.putText(img, f'{scores[f_index]:.2f}', (x1, y1-5), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # cv2.imshow('Detection Result', img)

    # faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return bboxes, f_index

def face_detect_old(image, vdevice=None):
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def extract_face_roi(image, faces):
    if faces is None or len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    face_roi = image[y:y + h, x:x + w]

    return face_roi, (x, y, w, h)

def preprocess(image, target_size):
    # Resize the image to the target size
    resized_image = cv2.resize(image, (target_size[1], target_size[0]))

    # Convert the image to RGB (if needed)
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Expand dimensions to match model input shape (1, height, width, channels)
    input_tensor = np.expand_dims(rgb_image, axis=0).astype(np.uint8)

    return input_tensor

def logging(outputs):
    out = outputs
    probs = np.squeeze(out)
    
    pred_idx = int(np.argmax(probs))

    if output_metrix[pred_idx] == 'neutral':
        is_detected = False
        message = "Neutral expression"
    else:
        is_detected = True
        message = output_metrix[pred_idx]

    analysis = {
        'emotion': (is_detected, message),
    }

    pos_log.log_analysis(analysis)

def postprocess(frame, outputs, bbox):
    x, y, w, h = bbox

    out = outputs
    probs = np.squeeze(out)

    #print(probs)

    if probs.ndim == 2 and probs.shape[0] == 1:
        probs = probs[0]
    
    pred_idx = int(np.argmax(probs))

    # Map to human label for common 2-class mask model, otherwise generic label
    emotion = output_metrix[pred_idx]

    # Draw rectangle around face and label with predicted emotion
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return frame

def bbox_test():
    # 카메라 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다")
        return

    print("camera opened")

    # VDevice 설정
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    
    with VDevice(params) as vdevice:
        # Capture frame-by-frame
        while True:
            ret, frame = cap.read()
            face_infer_model = face_detect(frame, vdevice)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def run_camera_inference(hef_path):
    
    # 카메라 열기
    cap = cv2.VideoCapture(CAMERA)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다")
        return
    
    print("camera opened")

    # VDevice 설정
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    params.group_id = "SHARED"
    params.multi_process_service = True
    
    with VDevice(params) as vdevice:
        # InferModel 생성
        infer_model = vdevice.create_infer_model(hef_path)
        
        # 입출력 정보 확인
        input_shape = infer_model.input().shape
        output_shape = infer_model.output().shape
        
        print(f"input: {input_shape}")
        print(f"output: {output_shape}")
        
        # 모델 설정
        with infer_model.configure() as configured_infer_model:
            # Bindings 생성
            bindings = configured_infer_model.create_bindings()

            try:
                while True:
                    # 프레임 읽기
                    # Capture frame-by-frame
                    ret, frame = cap.read()

                    # Convert frame to grayscale
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Detect faces in the frame
                    faces = face_detect_old(gray_frame, vdevice)

                    # try:
                    #     bboxes, f_index = face_detect(gray_frame, vdevice)
                    # except Exception as e:
                    #     continue

                    # x1, y1, x2, y2 = bboxes[f_index].astype(int)
                    # w, h = x2 - x1, y2 - y1

                    # faces = [(x1, y1, w, h)]

                    if faces is None or len(faces) == 0:
                        # no faces -> continue to next frame
                        if visualize(frame):
                            break
                        continue

                    print(f"faces: {faces}")

                    try: 
                        face_roi, (x, y, w, h) = extract_face_roi(gray_frame, faces)
                    except Exception as e:
                        if visualize(frame):
                            break

                    if face_roi is None:
                        # no faces -> continue to next frame
                        frame_count += 1
                        cv2.imshow('Hailo Camera', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        continue

                    # Preprocess face_roi to match ONNX model input
                    # Determine target size from model input shape if available
                    ih = input_shape[1]
                    iw = input_shape[0]

                    input_tensor = preprocess(face_roi, (iw, ih))

                    print(input_tensor.shape)

                    # NPU 추론

                    # 입력 버퍼 설정
                    bindings.input().set_buffer(input_tensor)

                    # 출력 버퍼 준비
                    output_buffer = np.empty(output_shape, dtype=np.uint8)
                    bindings.output().set_buffer(output_buffer)

                    configured_infer_model.wait_for_async_ready(timeout_ms=1000)

                    # 비동기 추론
                    job = configured_infer_model.run_async([bindings], 
                    partial(
                        bbox_callback, 
                        bindings=bindings))

                    job.wait(timeout_ms=10000)

                    # 결과 가져오기
                    outputs = bindings.output().get_buffer()

                    # NPU 추론 끝
                    logging(outputs)

                    if DEBUG:
                        frame = postprocess(frame, outputs, (x, y, w, h))

                    if visualize(frame):
                        break
            except KeyboardInterrupt:
                print("Interrupted by user.")
                
    
    cap.release()
    cv2.destroyAllWindows()

def visualize(frame):
    # 화면 표시
    if DEBUG is False:
        return False

    cv2.imshow('Hailo Camera', frame)
                
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True

if __name__ == "__main__":
    import argparse

    global BBOX_SIZE, CAMERA, DEBUG

    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_hef = os.path.join(base_path, 'models', 'best_model_float32_5class.hef')
    
    parser = argparse.ArgumentParser(description='Real-time Pose Estimation with Hailo')
    parser.add_argument('--hef', type=str, default=default_hef,
                       help='Path to HEF model file')
    parser.add_argument('--camera', type=str, default='/dev/video0',
                       help='Camera device path (default: /dev/video0)')
    parser.add_argument('--bbox_size', type=float, default=200,
                       help='Bounding box size (default: 200)')
    parser.add_argument('--debug', type=bool, default=False,
                       help='Enable debug mode (default: False)')

    args = parser.parse_args()

    BBOX_SIZE = args.bbox_size
    CAMERA = args.camera
    DEBUG = args.debug

    run_camera_inference(args.hef)
    # run_camera_inference("../../models/resmasking.hef")
    # bbox_test()
