"""
HAR to HEF Converter for Hailo-8 AI Accelerator
================================================
Hailo SDK를 사용하여 HAR 파일을 HEF(Hailo Executable Format)로 변환

주요 기능:
- Calibration Dataset 생성 (FER2013 데이터 기반)
- INT8 양자화 수행
- Model Script 기반 최적화
- Quantized Model 검증
- HEF 컴파일 및 저장

변환 파이프라인:
HAR → Calibration → Quantization → Quantized HAR → HEF Compile → HEF

Requirements:
- hailo_sdk_client
- PIL (Pillow)
- numpy
- tqdm

Author: Your Name
License: MIT
"""

from hailo_sdk_client import ClientRunner, InferenceContext
import numpy as np
from PIL import Image
import os
import random
from tqdm import tqdm


# ============================================
# 1) 전역 설정
# ============================================

# 데이터셋 경로
INPUT_PATH = "./fer2013/"                    # FER2013 데이터셋 루트
TRAIN_DIR = os.path.join(INPUT_PATH, 'train')  # 학습 데이터 (Calibration용)
TEST_DIR = os.path.join(INPUT_PATH, 'test')    # 테스트 데이터 (검증용)

# 클래스 매핑 (7-class → 3-class 감정 분류)
CLASS_MAPPING = {
    'happy': 'positive',
    'neutral': 'neutral',
    'angry': 'negative',
    'disgust': 'negative',
    'fear': 'negative',
    'sad': 'negative',
    'surprise': 'negative'
}

ORIGINAL_CLASSES = list(CLASS_MAPPING.keys())  # 원본 7개 클래스

# 모델 설정
IMG_SIZE = (224, 224)          # 입력 이미지 크기
NUM_CALIB_SAMPLES = 1024       # Calibration 샘플 수 (권장: 256~2048)

# 입출력 파일 경로
HAR_PATH = 'best_model_float32_3class_optimized_test.har'  # 입력 HAR 파일
QUANTIZED_HAR = 'best_model_float32_3class_quantized_test.har'  # 출력 Quantized HAR
HEF_FILENAME = 'best_model_float32_3class_test.hef'  # 출력 HEF 파일

print("="*60)
print("HAR → 양자화 → HEF 변환")
print("="*60)


# ============================================
# 2) Calibration Dataset 생성
# ============================================
def create_calibration_dataset(data_dir, num_samples=1024, img_size=(224, 224)):
    """
    FER2013 데이터셋에서 Calibration Dataset 생성
    
    Calibration Dataset은 양자화 과정에서 활성화 값의 분포를 추정하는 데 사용됩니다.
    모델의 정확도를 유지하면서 INT8로 양자화하기 위해 필요합니다.
    
    프로세스:
    1. 모든 클래스에서 이미지 경로 수집
    2. 랜덤 샘플링 (클래스 균형 고려)
    3. 이미지 로드 및 전처리
    4. NumPy 배열로 변환
    
    Args:
        data_dir (str): 데이터셋 디렉토리 경로
        num_samples (int): 샘플링할 이미지 개수 (권장: 256~2048)
        img_size (tuple): 리사이즈할 이미지 크기 (H, W)
        
    Returns:
        np.ndarray: Calibration 데이터셋 (N, H, W, 3), [0.0, 1.0] 범위
        
    Note:
        - 너무 적은 샘플: 양자화 정확도 저하
        - 너무 많은 샘플: 변환 시간 증가
        - 권장: 1024개 샘플 (적절한 균형)
    """
    print(f"\n=== Calibration Dataset 생성 ===")
    print(f"데이터 경로: {data_dir}")
    print(f"목표 샘플 수: {num_samples}")
    
    # 모든 클래스의 이미지 경로 수집
    all_images = []
    for class_name in ORIGINAL_CLASSES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"  경고: {class_dir} 폴더가 없습니다.")
            continue
        
        # 지원 확장자: jpg, png, jpeg
        img_files = [f for f in os.listdir(class_dir) 
                     if f.endswith(('.jpg', '.png', '.jpeg'))]
        img_paths = [os.path.join(class_dir, f) for f in img_files]
        all_images.extend(img_paths)
        print(f"  {class_name}: {len(img_files)}개 이미지")
    
    print(f"\n총 이미지 수: {len(all_images)}")
    
    # 랜덤 샘플링
    if len(all_images) > num_samples:
        selected_images = random.sample(all_images, num_samples)
        print(f"  → {num_samples}개 랜덤 샘플링")
    else:
        selected_images = all_images
        print(f"  경고: 요청한 샘플 수보다 적은 {len(all_images)}개만 사용")
    
    # 이미지 로드 및 전처리
    calib_data = []
    print(f"\n이미지 로드 중...")
    for img_path in tqdm(selected_images):
        try:
            # 이미지 로드: RGB 변환
            img = Image.open(img_path).convert('RGB')
            
            # 리사이즈: 모델 입력 크기에 맞춤
            img = img.resize(img_size)
            
            # NumPy 배열 변환 및 정규화 [0, 255] → [0.0, 1.0]
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # 배치 차원 추가 (H, W, C) → (1, H, W, C)
            calib_data.append(np.expand_dims(img_array, axis=0))
            
        except Exception as e:
            print(f"  오류 (건너뜀): {img_path} - {e}")
            continue
    
    if len(calib_data) == 0:
        raise ValueError("Calibration dataset이 비어있습니다!")
    
    # 모든 샘플을 하나의 배열로 결합 (N, H, W, C)
    calib_dataset = np.vstack(calib_data)
    
    print(f"\n✓ Calibration dataset 생성 완료")
    print(f"  Shape: {calib_dataset.shape}")
    print(f"  값 범위: [{calib_dataset.min():.3f}, {calib_dataset.max():.3f}]")
    print(f"  Dtype: {calib_dataset.dtype}")
    
    return calib_dataset


# ============================================
# 3) HAR 파일 로드
# ============================================
print(f"\n=== HAR 파일 로드 ===")
print(f"파일: {HAR_PATH}")

# HAR 파일 존재 여부 확인
if not os.path.exists(HAR_PATH):
    raise FileNotFoundError(f"HAR 파일을 찾을 수 없습니다: {HAR_PATH}")

# Hailo SDK ClientRunner 초기화
# hw_arch: 타겟 하드웨어 아키텍처 (hailo8, hailo15 등)
runner = ClientRunner(hw_arch='hailo8', har=HAR_PATH)
print("✓ HAR 로드 완료")


# ============================================
# 4) Model Script 적용 (양자화 최적화 설정)
# ============================================
print("\n=== Model Script 적용 ===")

# Model Script: 양자화 및 최적화 설정을 정의하는 스크립트
# - optimization_level: 모델 최적화 강도 (0~4, 높을수록 공격적)
# - compression_level: 모델 압축 레벨 (0~4, 높을수록 작은 크기)
model_script_content = """
# Flatten + Dense 구조를 위한 최적화 설정

# 전체 모델 최적화 레벨
# - optimization_level=2: 균형잡힌 최적화 (속도 vs 정확도)
# - compression_level=2: 중간 수준 압축
model_optimization_flavor(optimization_level=2, compression_level=2)

# 선택적: 특정 레이어의 비트 폭 설정
# Dense 레이어의 정밀도를 유지하여 정확도 손실 최소화
# set_bit_width(layers=model/flatten_projection/MatMul, bit_width=8)
# set_bit_width(layers=model/dense_1/MatMul, bit_width=8)
"""

# Model Script 파일 저장
model_script_path = 'model_script.alls'
with open(model_script_path, 'w') as f:
    f.write(model_script_content)

# Model Script 로드 시도
try:
    runner.load_model_script(model_script_path)
    print("✓ Model Script 로드 완료")
except Exception as e:
    print(f"경고: Model Script 로드 실패 - {e}")
    print("  기본 설정으로 진행합니다.")


# ============================================
# 5) Calibration Dataset 생성
# ============================================
calib_dataset = create_calibration_dataset(
    data_dir=TRAIN_DIR,
    num_samples=NUM_CALIB_SAMPLES,
    img_size=IMG_SIZE
)


# ============================================
# 6) 양자화 수행 (Float32 → INT8)
# ============================================
print("\n=== 양자화 시작 ===")
print("이 과정은 수 분이 걸릴 수 있습니다...")
print("진행 상황:")

try:
    # 양자화 실행
    # - Calibration 데이터를 사용하여 활성화 값의 범위 추정
    # - Float32 가중치 → INT8 가중치 변환
    # - 정확도 손실 최소화를 위한 스케일 팩터 계산
    runner.optimize(calib_dataset)
    print("\n✓ 양자화 완료!")
    
except Exception as e:
    print(f"\n✗ 양자화 실패: {e}")
    
    # 에러 발생 시 최소 최적화 레벨로 재시도
    print("\n대안: optimization_level=0으로 재시도...")
    
    # 최소 최적화 설정 (가장 안전하지만 성능 저하 가능)
    simple_script = """
model_optimization_flavor(optimization_level=0, compression_level=0)
"""
    with open(model_script_path, 'w') as f:
        f.write(simple_script)
    
    # ClientRunner 재초기화 및 재시도
    runner = ClientRunner(hw_arch='hailo8', har=HAR_PATH)
    runner.load_model_script(model_script_path)
    runner.optimize(calib_dataset)
    print("✓ 양자화 완료 (optimization_level=0)")


# ============================================
# 7) Quantized Model 테스트
# ============================================
print("\n=== Quantized Model 테스트 ===")

def preprocess_single_image(image_path, img_size=(224, 224)):
    """
    단일 이미지 전처리 (추론용)
    
    프로세스:
    1. 이미지 로드 및 RGB 변환
    2. 리사이즈
    3. 정규화 [0, 255] → [0.0, 1.0]
    4. 배치 차원 추가
    
    Args:
        image_path (str): 이미지 파일 경로
        img_size (tuple): 리사이즈 크기 (H, W)
        
    Returns:
        np.ndarray: 전처리된 이미지 (1, H, W, 3)
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


# 테스트 샘플 선택 (각 감정에서 랜덤 1개씩)
test_samples = []
for class_name in ['happy', 'neutral', 'angry']:
    class_dir = os.path.join(TEST_DIR, class_name)
    if os.path.exists(class_dir):
        img_files = [f for f in os.listdir(class_dir) 
                     if f.endswith(('.jpg', '.png', '.jpeg'))]
        if img_files:
            # 랜덤으로 1개 선택
            selected_img = random.choice(img_files)
            test_samples.append((class_name, os.path.join(class_dir, selected_img)))

# 출력 클래스 이름 (3-class)
class_names = ['positive', 'neutral', 'negative']

if test_samples:
    print(f"테스트 샘플 {len(test_samples)}개로 검증 중... (랜덤 선택)")
    
    # SDK_QUANTIZED: 양자화된 모델로 추론
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        for class_name, img_path in test_samples:
            # 이미지 전처리
            test_image = preprocess_single_image(img_path, IMG_SIZE)
            
            # 추론 실행
            result = runner.infer(ctx, test_image)
            
            # 결과 후처리
            result = result.flatten()  # (1, N) → (N,)
            
            # Softmax 적용 (Logits → 확률)
            exp_vals = np.exp(result - np.max(result))  # 수치 안정성을 위한 최댓값 빼기
            probs = exp_vals / exp_vals.sum()
            pred_class = np.argmax(probs)
            
            # 정답 클래스
            expected_class = CLASS_MAPPING[class_name]
            
            # 결과 출력
            print(f"\n이미지: {os.path.basename(img_path)}")
            print(f"  실제 클래스: {class_name} → {expected_class}")
            print(f"  예측 클래스: {class_names[pred_class]}")
            print(f"  확률: positive={probs[0]:.3f}, neutral={probs[1]:.3f}, negative={probs[2]:.3f}")
            
            # 정확도 평가
            if class_names[pred_class] == expected_class:
                print("  ✓ 예측 정확!")
            else:
                print("  ✗ 예측 틀림")
else:
    print("테스트 샘플을 찾을 수 없습니다. 건너뜁니다.")


# ============================================
# 8) Quantized HAR 저장
# ============================================
# Quantized HAR: 양자화가 적용된 중간 형식
# - 디버깅 및 분석 용도
# - HEF 변환 전 백업
runner.save_har(QUANTIZED_HAR)

har_size = os.path.getsize(QUANTIZED_HAR) / (1024*1024)
print(f"\n✓ Quantized HAR 저장 완료")
print(f"  파일: {QUANTIZED_HAR}")
print(f"  크기: {har_size:.2f} MB")


# ============================================
# 9) HEF 컴파일 (Hailo-8 실행 파일 생성)
# ============================================
print("\n=== HEF 컴파일 시작 ===")

try:
    print("HEF 컴파일 중...")
    
    # HEF 컴파일
    # - Hailo-8 하드웨어에서 실행 가능한 바이너리 생성
    # - 추론 최적화 및 메모리 레이아웃 결정
    hef = runner.compile()
    
    # HEF 파일 저장
    with open(HEF_FILENAME, "wb") as f:
        f.write(hef)
    
    print(f"\n✓ HEF 파일 저장 완료: {HEF_FILENAME}")
    
    # 파일 검증
    if os.path.exists(HEF_FILENAME):
        file_size = os.path.getsize(HEF_FILENAME)
        print(f"  크기: {file_size / (1024*1024):.2f} MB")
        
        if file_size > 0:
            print(f"  ✓ HEF 파일이 정상적으로 생성되었습니다")
        else:
            print(f"  ✗ 경고: HEF 파일이 비어있습니다")
    else:
        print(f"  ✗ 경고: HEF 파일을 찾을 수 없습니다")
    
    # 최종 결과 요약
    print("\n" + "="*50)
    print("변환 완료!")
    print(f"최종 출력 파일:")
    print(f"  - Quantized HAR: {QUANTIZED_HAR}")
    print(f"  - HEF: {HEF_FILENAME}")
    print("="*50)
    
except Exception as e:
    print(f"\n✗ HEF 컴파일 실패: {e}")
    print(f"Quantized HAR 파일은 저장되었습니다: {QUANTIZED_HAR}")
    
    # 상세 에러 출력
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("변환 프로세스 완료")
print("="*60)