"""
FER2013 5-Class Emotion Recognition Model
=========================================
MobileNetV3Small 백본을 사용한 감정 인식 모델 학습 스크립트

주요 기능:
- 5가지 감정 분류: angry, disgust, happy, neutral, surprise
- 데이터 불균형 처리: 오버샘플링 + 클래스 가중치
- 데이터 증강: MixUp, Random Augmentation
- 최적화: Focal Loss, Warmup+Cosine LR Scheduling
- 2단계 학습: Backbone Frozen → Fine-tuning

Dataset: FER2013 (Kaggle)
Model: MobileNetV3Small (ImageNet Pretrained)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
import os

# ============================================
# 의존성 확인 및 설치
# ============================================
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'scikit-learn'])
    from sklearn.model_selection import train_test_split

print("TensorFlow:", tf.__version__)
AUTO = tf.data.AUTOTUNE


# ============================================
# 1) 전역 설정 및 하이퍼파라미터
# ============================================

# 경로 설정
INPUT_PATH = "./fer2013/"                      # FER2013 데이터셋 루트 폴더
OUTPUT_DIR = "./fer2013_keras_hailo"           # 모델 출력 디렉토리
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = INPUT_PATH
TRAIN_DIR = os.path.join(DATA_DIR, 'train')    # 학습 데이터 경로
TEST_DIR = os.path.join(DATA_DIR, 'test')      # 테스트 데이터 경로 (선택사항)

# 클래스 설정 (5가지 감정만 사용)
SELECTED_CLASSES = ['angry', 'disgust', 'happy', 'neutral', 'surprise']
NUM_CLASSES = len(SELECTED_CLASSES)

# 이미지 크기 설정
IMG_SRC_SIZE = 48    # FER2013 원본 이미지 크기 (48×48 grayscale)
IMG_SIZE     = 128   # 모델 입력 크기 (128×128 RGB)

# 학습 하이퍼파라미터
BATCH_SIZE   = 32              # 배치 크기
EPOCHS_STAGE1 = 25             # Stage 1: Backbone Frozen 학습 epoch
EPOCHS_STAGE2 = 150            # Stage 2: Fine-tuning 학습 epoch
SEED = 42                      # 랜덤 시드 (재현성 보장)
VALIDATION_SPLIT = 0.2         # 검증 데이터 비율 (20%)
MIXUP_ALPHA = 0.2              # MixUp 증강 강도 (0.0 = 비활성화)

# 캘리브레이션 설정 (Hailo/TFLite 변환용)
CALIB_DIR = '/calib_content/calib_images'   # 캘리브레이션 이미지 저장 경로
CALIB_COUNT = 256                           # 캘리브레이션 샘플 수

# 랜덤 시드 고정 (재현 가능한 결과)
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)


# ============================================
# 2) 데이터 로딩
# ============================================
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

def load_filtered_image_paths(root_dir, selected_classes):
    """
    지정된 클래스의 이미지 경로와 레이블을 로드
    
    Args:
        root_dir (str): 데이터셋 루트 디렉토리 (train 또는 test)
        selected_classes (list): 사용할 클래스 이름 리스트
        
    Returns:
        tuple: (이미지 경로 리스트, 레이블 리스트)
    """
    root_path = Path(root_dir)
    class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}
    all_paths, all_labels = [], []
    
    for cls in selected_classes:
        cls_path = root_path / cls
        if not cls_path.exists():
            continue
        
        # 지원 이미지 확장자
        exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        for ext in exts:
            for img_path in cls_path.glob(ext):
                all_paths.append(str(img_path))
                all_labels.append(class_to_idx[cls])
                
    return all_paths, all_labels

# 학습 데이터 로드
all_train_paths, all_train_labels = load_filtered_image_paths(TRAIN_DIR, SELECTED_CLASSES)


# ============================================
# 2-1) 오버샘플링 (부분 균형화)
# ============================================
def oversample_minority_classes(paths, labels, target_ratio=0.6):
    """
    소수 클래스를 반복하여 데이터 불균형 완화
    
    전략:
    - 최대 클래스 개수의 target_ratio 비율까지 소수 클래스 증강
    - 예: target_ratio=0.6이면 소수 클래스를 최대 클래스의 60%까지 증강
    
    Args:
        paths (list): 이미지 경로 리스트
        labels (list): 레이블 리스트
        target_ratio (float): 목표 비율 (0~1)
        
    Returns:
        tuple: (증강된 경로 리스트, 증강된 레이블 리스트)
    """
    paths = np.array(paths)
    labels = np.array(labels)
    counts = np.bincount(labels)  # 각 클래스별 샘플 수
    max_count = counts.max()      # 최대 클래스의 샘플 수

    oversampled_paths, oversampled_labels = [], []

    for cls_idx in range(len(counts)):
        cls_mask = labels == cls_idx
        cls_paths = paths[cls_mask]

        target_count = int(max_count * target_ratio)  # 목표 샘플 수
        current_count = len(cls_paths)
        
        if current_count == 0:
            print(f"  {SELECTED_CLASSES[cls_idx]:12s}: {0:5d} -> {0:5d} (no samples)")
            continue
            
        if current_count < target_count:
            # 소수 클래스: 반복으로 증강
            n_repeats = target_count // current_count
            remainder = target_count % current_count
            
            # 전체 반복
            repeated_paths = list(cls_paths) * n_repeats
            repeated_labels = [cls_idx] * (current_count * n_repeats)
            
            # 나머지 샘플 랜덤 추가
            if remainder > 0:
                extra_indices = np.random.choice(len(cls_paths), remainder, replace=False)
                extra_paths = cls_paths[extra_indices].tolist()
                extra_labels = [cls_idx] * remainder
            else:
                extra_paths, extra_labels = [], []
                
            oversampled_paths.extend(repeated_paths + extra_paths)
            oversampled_labels.extend(repeated_labels + extra_labels)
            print(f"  {SELECTED_CLASSES[cls_idx]:12s}: {current_count:5d} -> {(len(repeated_paths)+len(extra_paths)):5d}")
        else:
            # 다수 클래스: 그대로 유지
            oversampled_paths.extend(cls_paths.tolist())
            oversampled_labels.extend([cls_idx] * current_count)
            print(f"  {SELECTED_CLASSES[cls_idx]:12s}: {current_count:5d} -> {current_count:5d} (majority)")

    # 데이터 셔플
    idx = np.arange(len(oversampled_paths))
    np.random.default_rng(SEED).shuffle(idx)
    oversampled_paths  = [oversampled_paths[i] for i in idx]
    oversampled_labels = [oversampled_labels[i] for i in idx]
    
    return oversampled_paths, oversampled_labels

print("\nOversampling:")
all_train_paths, all_train_labels = oversample_minority_classes(
    all_train_paths, all_train_labels, target_ratio=0.6
)

# Train/Validation 분할 (Stratified - 클래스 비율 유지)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_train_paths,
    all_train_labels,
    test_size=VALIDATION_SPLIT,
    stratify=all_train_labels,  # 클래스 비율 동일하게 유지
    random_state=SEED
)

print(f"\nTrain: {len(train_paths):,}")
print(f"Val:   {len(val_paths):,}")

# 클래스별 분포 확인
train_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
val_counts   = np.bincount(val_labels,   minlength=NUM_CLASSES)

print("\nTrain:")
for i, cls in enumerate(SELECTED_CLASSES):
    print(f"  {cls:12s}: {train_counts[i]:5d}")
    
print("\nVal:")
for i, cls in enumerate(SELECTED_CLASSES):
    print(f"  {cls:12s}: {val_counts[i]:5d}")

# 테스트 데이터 로드 (선택사항)
test_paths = test_labels = None
if os.path.exists(TEST_DIR):
    test_paths, test_labels = load_filtered_image_paths(TEST_DIR, SELECTED_CLASSES)
    print(f"\nTest:  {len(test_paths):,}")


# ============================================
# 3) 클래스 가중치 계산 (Effective Number)
# ============================================
def compute_class_weights_effective(counts, beta=0.99):
    """
    Effective Number 기반 클래스 가중치 계산
    
    공식: w_i = (1 - β) / (1 - β^n_i)
    - 샘플이 적은 클래스에 더 높은 가중치 부여
    - β=0.99: 불균형이 심할수록 효과적
    
    Args:
        counts (array): 각 클래스의 샘플 수
        beta (float): 감쇠 계수 (0~1, 보통 0.99 사용)
        
    Returns:
        array: 정규화된 클래스 가중치
        
    Reference:
        Cui et al., "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
    """
    counts = np.array(counts, dtype=np.float64)
    effective_num = 1.0 - np.power(beta, np.maximum(counts, 1.0))
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(weights)  # 정규화
    return weights.astype(np.float32)

class_weights = compute_class_weights_effective(train_counts, beta=0.99)
print("\nClass Weights:")
for i, cls in enumerate(SELECTED_CLASSES):
    print(f"  {cls:12s}: {class_weights[i]:.3f}")


# ============================================
# 4) tf.data 파이프라인 구축
# ============================================
def create_tf_dataset(paths, labels):
    """
    이미지 경로와 레이블로부터 tf.data.Dataset 생성
    
    Args:
        paths (list): 이미지 파일 경로 리스트
        labels (list): 레이블 리스트
        
    Returns:
        tf.data.Dataset: 파이프라인 데이터셋
    """
    if paths is None:
        return None

    def load_image(path, label):
        """이미지 파일을 읽고 48×48 크기로 리사이즈"""
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=1, expand_animations=False)
        img.set_shape([None, None, 1])
        img = tf.image.resize(img, [IMG_SRC_SIZE, IMG_SRC_SIZE])
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=AUTO)
    return ds

# 원본 데이터셋 생성 (증강 전)
train_raw = create_tf_dataset(train_paths, train_labels)
val_raw   = create_tf_dataset(val_paths,   val_labels)
test_raw  = create_tf_dataset(test_paths,  test_labels) if test_paths is not None else None


# ============================================
# 4-1) 전처리 함수
# ============================================
def preprocess_image(x, y, training: bool):
    """
    이미지 전처리 및 증강
    
    전처리 단계:
    1. 48×48 → 128×128 리사이즈 (Bicubic)
    2. Grayscale → RGB 변환 (채널 복제)
    3. [0, 255] → [0, 1] 정규화
    4. 데이터 증강 (training=True인 경우만)
    5. [-1, 1] 범위로 최종 정규화
    
    Args:
        x (Tensor): 입력 이미지 (48×48×1)
        y (Tensor): 레이블
        training (bool): 학습 모드 여부 (증강 적용 여부)
        
    Returns:
        tuple: (전처리된 이미지, 레이블)
    """
    x = tf.cast(x, tf.float32)
    
    # 리사이즈: 48×48 → 128×128 (고품질 Bicubic)
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE), method='bicubic', antialias=True)
    
    # Grayscale → RGB 변환 (MobileNetV3는 RGB 입력 필요)
    x = tf.image.grayscale_to_rgb(x)

    # [0, 255] → [0, 1] 정규화
    x = x / 255.0

    # 데이터 증강 (학습 시에만 적용)
    if training:
        # 좌우 반전 (50% 확률)
        x = tf.image.random_flip_left_right(x)
        
        # 밝기 조정 (±20%)
        x = tf.image.random_brightness(x, 0.2)
        
        # 대비 조정 (0.8~1.2배)
        x = tf.image.random_contrast(x, 0.8, 1.2)

        # 색상 증강 (50% 확률)
        if tf.random.uniform([]) > 0.5:
            x = tf.image.random_saturation(x, 0.9, 1.1)  # 채도
        if tf.random.uniform([]) > 0.5:
            x = tf.image.random_hue(x, 0.03)             # 색조

        # 값 범위 클리핑 [0, 1]
        x = tf.clip_by_value(x, 0.0, 1.0)

        # 랜덤 줌 & 크롭 (80% 확률, 85~100% 줌)
        if tf.random.uniform([]) > 0.2:
            zoom = tf.random.uniform([], 0.85, 1.0)
            new_h = tf.maximum(tf.cast(IMG_SIZE * zoom, tf.int32), 32)
            new_w = tf.maximum(tf.cast(IMG_SIZE * zoom, tf.int32), 32)
            x = tf.image.random_crop(x, [new_h, new_w, 3])
            x = tf.image.resize(x, [IMG_SIZE, IMG_SIZE])

    # 최종 정규화: [0, 1] → [-1, 1] (MobileNetV3 입력 범위)
    x = x * 2.0 - 1.0
    x = tf.clip_by_value(x, -1.0, 1.0)

    return x, y


# ============================================
# 4-2) MixUp 증강
# ============================================
def mixup(x, y, alpha=MIXUP_ALPHA):
    """
    MixUp 데이터 증강 기법
    
    두 개의 샘플을 랜덤 비율로 섞어 새로운 샘플 생성:
    - x_mixed = λ * x1 + (1-λ) * x2
    - y_mixed = λ * y1 + (1-λ) * y2
    
    효과:
    - 모델의 일반화 성능 향상
    - 노이즈에 강건한 학습
    - 과적합 방지
    
    Args:
        x (Tensor): 배치 이미지 (B, H, W, C)
        y (Tensor): 배치 레이블 (B, num_classes) - one-hot
        alpha (float): MixUp 강도 (0.0 = 비활성화)
        
    Returns:
        tuple: (섞인 이미지, 섞인 레이블)
        
    Reference:
        Zhang et al., "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)
    """
    if alpha <= 0:
        return x, y

    batch_size = tf.shape(x)[0]

    # 혼합 비율 λ 샘플링 (0.3~0.9 범위로 제한)
    lam = tf.random.uniform([], 0.3, 0.9, dtype=tf.float32)

    # 배치 내 인덱스 셔플
    indices = tf.random.shuffle(tf.range(batch_size))
    x2 = tf.gather(x, indices)
    y2 = tf.gather(y, indices)

    # 이미지 및 레이블 혼합
    x_mixed = lam * x + (1.0 - lam) * x2
    y_mixed = lam * y + (1.0 - lam) * y2
    x_mixed = tf.clip_by_value(x_mixed, -1.0, 1.0)

    return x_mixed, y_mixed


def make_dataset(ds_raw, training: bool):
    """
    전처리, 증강, 배치 처리를 포함한 최종 데이터셋 생성
    
    Pipeline:
    1. 전처리 (리사이즈, 증강)
    2. 셔플 (training=True)
    3. 배치 생성
    4. One-hot 인코딩
    5. MixUp (training=True)
    6. Prefetch (성능 최적화)
    
    Args:
        ds_raw (tf.data.Dataset): 원본 데이터셋
        training (bool): 학습 모드 여부
        
    Returns:
        tf.data.Dataset: 전처리된 데이터셋
    """
    if ds_raw is None:
        return None

    # 전처리 및 증강
    ds = ds_raw.map(
        lambda x, y: preprocess_image(x, y, training),
        num_parallel_calls=AUTO
    )

    # 셔플 (학습 시에만)
    if training:
        ds = ds.shuffle(16384, seed=SEED, reshuffle_each_iteration=True)

    # 배치 생성 (학습 시 마지막 불완전한 배치 버림)
    ds = ds.batch(BATCH_SIZE, drop_remainder=training)
    
    # One-hot 인코딩
    ds = ds.map(
        lambda x, y: (x, tf.one_hot(y, NUM_CLASSES)),
        num_parallel_calls=AUTO
    )

    # MixUp 증강 (학습 시에만)
    if training and MIXUP_ALPHA > 0:
        ds = ds.map(
            lambda x, y: mixup(x, y, MIXUP_ALPHA),
            num_parallel_calls=AUTO
        )

    # Prefetch (GPU/CPU 병렬 처리 최적화)
    ds = ds.prefetch(AUTO)
    return ds

# 최종 데이터셋 생성
train_ds = make_dataset(train_raw, training=True)
val_ds   = make_dataset(val_raw,   training=False)
test_ds  = make_dataset(test_raw,  training=False) if test_raw is not None else None


# ============================================
# 5) Focal Loss 정의
# ============================================
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="custom")
class FocalLoss(keras.losses.Loss):
    """
    Focal Loss - 불균형 데이터셋을 위한 손실 함수
    
    공식: FL = -α_t * (1 - p_t)^γ * log(p_t)
    - α: 클래스 가중치 (소수 클래스에 더 큰 가중치)
    - γ: Focusing parameter (어려운 샘플에 집중)
    - p_t: 정답 클래스의 예측 확률
    
    특징:
    - 쉬운 샘플의 기여도 감소 (잘 분류되는 샘플)
    - 어려운 샘플에 집중 학습 (경계가 모호한 샘플)
    - 클래스 불균형 문제 완화
    
    Args:
        alpha (list): 클래스별 가중치 [w1, w2, ..., wn]
        gamma (float): Focusing parameter (보통 0.5~2.0)
        label_smoothing (float): 레이블 스무딩 강도 (0.0~0.2)
        from_logits (bool): 입력이 logits인지 여부
        
    Reference:
        Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    def __init__(self, alpha=None, gamma=0.5, label_smoothing=0.0, from_logits=True, name='focal_loss'):
        super().__init__(name=name)
        self._alpha_list = None if alpha is None else list(np.asarray(alpha, np.float32))
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.from_logits = bool(from_logits)

    def call(self, y_true, y_pred):
        """손실 계산"""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # 클래스 가중치 α
        alpha = tf.constant(self._alpha_list, dtype=tf.float32) if self._alpha_list else None

        # Label Smoothing (과적합 방지)
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes

        # Logits → 확률 변환
        if self.from_logits:
            log_probs = tf.nn.log_softmax(y_pred, axis=-1)
            probs = tf.nn.softmax(y_pred, axis=-1)
        else:
            probs = y_pred
            log_probs = tf.math.log(tf.clip_by_value(probs, 1e-7, 1.0))

        probs = tf.clip_by_value(probs, 1e-7, 1.0)

        # p_t: 정답 클래스의 예측 확률
        pt = tf.reduce_sum(y_true * probs, axis=-1, keepdims=True)
        pt = tf.clip_by_value(pt, 1e-7, 1.0)

        # Focal Weight: (1 - p_t)^γ
        focal_weight = tf.pow(1.0 - pt, self.gamma)

        # Cross Entropy
        ce = -tf.reduce_sum(y_true * log_probs, axis=-1, keepdims=True)

        # Focal Loss
        focal_loss = focal_weight * ce

        # 클래스 가중치 적용
        if alpha is not None:
            alpha_t = tf.reduce_sum(y_true * alpha, axis=-1, keepdims=True)
            focal_loss = alpha_t * focal_loss

        return tf.reduce_mean(focal_loss)

    def get_config(self):
        """설정 저장 (모델 저장/로드 시 필요)"""
        return {
            "alpha": self._alpha_list,
            "gamma": self.gamma,
            "label_smoothing": self.label_smoothing,
            "from_logits": self.from_logits,
            "name": self.name,
        }


# ============================================
# 6) 학습률 스케줄 (Warmup + Cosine Annealing)
# ============================================
@register_keras_serializable(package="custom")
class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    """
    Warmup + Cosine Annealing 학습률 스케줄러
    
    전략:
    1. Warmup: 초기 학습률을 서서히 증가 (안정적 학습 시작)
    2. Cosine Decay: Cosine 함수로 학습률 감소 (부드러운 수렴)
    
    수식:
    - Warmup (step < warmup_steps):
        lr = base_lr * (step / warmup_steps)
    - Cosine (step >= warmup_steps):
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))
    
    Args:
        base_lr (float): 최대 학습률
        total_steps (int): 전체 학습 스텝 수
        warmup_steps (int): Warmup 스텝 수
        min_lr (float): 최소 학습률
        
    Reference:
        Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (ICLR 2017)
    """
    def __init__(self, base_lr, total_steps, warmup_steps, min_lr=1e-6):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        """현재 스텝의 학습률 계산"""
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps  = tf.cast(self.total_steps,  tf.float32)
        
        # Warmup: 선형 증가
        warm = self.base_lr * (step / tf.maximum(1.0, warmup_steps))
        
        # Cosine Decay: 부드러운 감소
        cosine = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + tf.cos(np.pi * (step - warmup_steps) / tf.maximum(1.0, total_steps - warmup_steps))
        )
        
        return tf.where(step < warmup_steps, warm, cosine)

    def get_config(self):
        """설정 저장"""
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr
        }


# ============================================
# 7) 모델 구축
# ============================================
def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES, stage=1):
    """
    MobileNetV3Small 기반 감정 인식 모델 생성 아키텍처:
    - Backbone: MobileNetV3Small (ImageNet Pretrained)
    - Head: Dense(512) → Dense(256) → Dense(128) → Dense(num_classes)
    - 정규화: BatchNorm, Dropout, L2 Regularization

    2단계 학습 전략:
    - Stage 1: Backbone 동결, Head만 학습
    - Stage 2: 상위 60개 레이어 언프리즈, Fine-tuning

    Args:
        input_shape (tuple): 입력 이미지 크기 (H, W, C)
        num_classes (int): 출력 클래스 수
        stage (int): 학습 단계 (1=Frozen, 2=Fine-tuning)
        
    Returns:
        tuple: (model, backbone) - 전체 모델과 백본 객체
    """
    inp = layers.Input(shape=input_shape, name="input_image")

    # MobileNetV3Small 백본 (ImageNet 가중치 사용)
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,           # 분류기 제외 (특징 추출만)
        weights="imagenet",          # ImageNet Pretrained
        include_preprocessing=False, # 전처리는 수동으로 수행
        minimalistic=False           # SE-Block 포함 (성능 우선)
    )

    # Stage 1: Backbone 완전 동결
    if stage == 1:
        base.trainable = False
        training_mode = False
        print("Stage 1: Backbone FROZEN")

    # Stage 2: 상위 레이어만 Fine-tuning
    else:
        base.trainable = True
        freeze_until = len(base.layers) - 60  # 하위 레이어는 동결 유지
        trainable_count = 0
        
        for i, layer in enumerate(base.layers):
            if i >= freeze_until:
                # BatchNorm은 항상 동결 (분포 안정성 유지)
                if isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False
                else:
                    layer.trainable = True
                    trainable_count += 1
            else:
                layer.trainable = False
        
        training_mode = True
        print(f"Stage 2: Unfrozen {trainable_count} layers (BN excluded)")

    # Backbone 통과 (특징 추출)
    x = base(inp, training=training_mode)

    # ============================================
    # Custom Classification Head
    # ============================================

    # Global Average Pooling: (H, W, C) → (C,)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.BatchNormalization(name='bn_initial')(x)
    x = layers.Dropout(0.5, name='dropout_1')(x)

    # Dense Block 1: 512 units
    x = layers.Dense(
        512, 
        use_bias=False,  # BN 사용 시 bias 불필요
        kernel_regularizer=keras.regularizers.l2(3e-5),  # L2 정규화
        name='dense_1'
    )(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Activation('relu', name='relu_1')(x)
    x = layers.Dropout(0.4, name='dropout_2')(x)

    # Dense Block 2: 256 units
    x = layers.Dense(
        256, 
        use_bias=False, 
        kernel_regularizer=keras.regularizers.l2(3e-5), 
        name='dense_2'
    )(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Activation('relu', name='relu_2')(x)
    x = layers.Dropout(0.3, name='dropout_3')(x)

    # Dense Block 3: 128 units
    x = layers.Dense(
        128, 
        use_bias=False, 
        kernel_regularizer=keras.regularizers.l2(3e-5), 
        name='dense_3'
    )(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.Activation('relu', name='relu_3')(x)
    x = layers.Dropout(0.2, name='dropout_4')(x)

    # Output Layer: Logits (활성화 함수 없음)
    logits = layers.Dense(
        num_classes,
        use_bias=True,
        name="outputs",  # 추론 시 출력 레이어 이름
        kernel_initializer='glorot_uniform',
        activation=None  # Logits 출력 (Focal Loss가 softmax 처리)
    )(x)

    # 최종 모델 생성
    model = keras.Model(inp, logits, name=f"MobileNetV3_S{stage}")

    # 파라미터 수 출력
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    total_params     = sum([tf.size(w).numpy() for w in model.weights])
    print(f"\nTrainable Parameters: {trainable_params:,}")
    print(f"Total Parameters:     {total_params:,}")

    return model, base
    
# ============================================
# 8) Optimizer 생성
# ============================================
def create_optimizer(learning_rate, stage=1):
    """
    Adam Optimizer 생성
    설정:
    - Adam: 적응적 학습률 (Momentum + RMSProp)
    - Gradient Clipping: 기울기 폭발 방지

    Args:
        learning_rate (float or LRSchedule): 학습률 또는 스케줄러
        stage (int): 학습 단계 (로깅용)
        
    Returns:
        keras.optimizers.Optimizer: 설정된 Optimizer
    """
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,      # 1차 모멘텀 계수
        beta_2=0.999,    # 2차 모멘텀 계수
        epsilon=1e-7,    # 수치 안정성
        clipnorm=1.0     # Gradient Clipping (L2 norm)
    )
    return optimizer

# ============================================
# 9) 평가 함수
# ============================================
def evaluate_model(model, val_ds, test_ds=None):
    """
    모델 성능 평가 및 분석
    출력:
    - Validation/Test Accuracy
    - Confusion Matrix
    - 클래스별 정확도 (Per-Class Accuracy)
    - Macro Average Accuracy

    Args:
        model (keras.Model): 평가할 모델
        val_ds (tf.data.Dataset): 검증 데이터셋
        test_ds (tf.data.Dataset): 테스트 데이터셋 (선택사항)
        
    Returns:
        tuple: (val_accuracy, per_class_accuracy)
    """
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    def _eval(ds):
        """데이터셋의 정확도 계산"""
        losses, accs = [], []
        for xb, yb in ds:
            # 예측: Logits → Softmax
            preds = tf.nn.softmax(model(xb, training=False)).numpy()
            y_true = np.argmax(yb.numpy(), axis=1)
            y_pred = np.argmax(preds, axis=1)
            accs.append((y_true == y_pred).mean())
        return float(np.mean(accs)) if accs else float("nan")

    # Validation 정확도
    val_acc = _eval(val_ds)
    print(f"\nValidation Accuracy: {val_acc:.4f}")

    # Test 정확도 (선택사항)
    if test_ds is not None:
        test_acc = _eval(test_ds)
        print(f"Test Accuracy:       {test_acc:.4f}")

    # Confusion Matrix 및 클래스별 분석
    y_true, y_pred = [], []
    for xb, yb in val_ds:
        probs = tf.nn.softmax(model(xb, training=False)).numpy()
        y_pred.append(np.argmax(probs, axis=1))
        y_true.append(np.argmax(yb.numpy(), axis=1))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Confusion Matrix
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES).numpy()

    # 클래스별 정확도 계산
    per_class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nPer-Class Accuracy:")
    all_above_80 = True
    for i, cls in enumerate(SELECTED_CLASSES):
        count = int(cm.sum(axis=1)[i])
        acc = per_class_acc[i]
        
        # 성능 상태 표시
        status = "OK" if acc >= 0.80 else ("WARN" if acc >= 0.70 else "LOW")
        print(f"  {status} {cls:12s}: {acc:.4f} ({count:5d} samples)")
        
        if acc < 0.80:
            all_above_80 = False

    # Macro Average (클래스 불균형 고려)
    print(f"\nMacro Average: {per_class_acc.mean():.4f}")

    # 성공 기준 평가
    if all_above_80:
        print("✓ Success: ALL CLASSES >= 80%!")
    elif per_class_acc.mean() >= 0.80:
        print("✓ Success: MACRO AVG >= 80%!")

    return val_acc, per_class_acc

# ============================================
# 10) Stage 1: Backbone Frozen 학습
# ============================================
print("\n" + "="*60)
print("BACKBONE FROZEN")
print("="*60)

# 모델 생성 (Stage 1)
model, backbone = build_model(num_classes=NUM_CLASSES, stage=1)

# Optimizer 설정 (고정 학습률)
optimizer_s1 = create_optimizer(learning_rate=5e-4, stage=1)

# Focal Loss 설정
alpha_list = class_weights.tolist()
loss_fn = FocalLoss(
    alpha=alpha_list,
    gamma=0.5,
    label_smoothing=0.0,
    from_logits=True
)

# 모델 컴파일
model.compile(
    optimizer=optimizer_s1,
    loss=loss_fn,
    metrics=[keras.metrics.CategoricalAccuracy(name='accuracy')]
)

# Callbacks 설정
callbacks_s1 = [
    keras.callbacks.TerminateOnNaN(),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
]

print("\nStarting Training...")
history_s1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    callbacks=callbacks_s1,
    verbose=1
)

# ===================================================
# 11) Stage 2: Fine-tuning (Partial Unfreeze)
# ===================================================
print("\n" + "="*60)
print("PARTIAL UNFREEZE")
print("="*60)

# Backbone 일부 언프리즈
backbone.trainable = True
freeze_until = len(backbone.layers) - 60 # 하위 레이어는 동결 유지
trainable_count = 0

for i, layer in enumerate(backbone.layers):
    if i >= freeze_until:
        # BatchNorm은 동결 유지 (분포 안정성)
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            trainable_count += 1
    else:
        layer.trainable = False

print(f"Unfrozen {trainable_count} layers (BN excluded)")

# Warmup + Cosine 학습률 스케줄
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
total_steps = steps_per_epoch * EPOCHS_STAGE2
warmup_steps = steps_per_epoch * 5

lr_schedule = WarmupCosine(
    base_lr=1e-4,  # 최대 학습률
    total_steps=total_steps,
    warmup_steps=warmup_steps,
    min_lr=1e-7    # 최소 학습률
)

# Optimizer 재생성 (스케줄러 적용)
optimizer_s2 = create_optimizer(learning_rate=lr_schedule, stage=2)

# 모델 재컴파일
model.compile(
    optimizer=optimizer_s2,
    loss=loss_fn,
    metrics=[keras.metrics.CategoricalAccuracy(name='accuracy')]
)

# Callbacks 설정
callbacks_s2 = [
    # NaN 발생 시 학습 중단
    keras.callbacks.TerminateOnNaN(),
    # 최고 성능 모델 저장
    keras.callbacks.ModelCheckpoint(
        "best_model_float32.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    # Early Stopping (35 epoch 동안 개선 없으면 중단)
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=35,
        restore_best_weights=True,
        mode="max",
        verbose=1
    )
]
print("\nStarting Training...")
history_s2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    callbacks=callbacks_s2,
    verbose=1
)

# ============================================
# 12) 최종 평가 및 결과 요약
# ============================================
# 최종 모델 평가
val_acc_s2, per_class_acc = evaluate_model(model, val_ds, test_ds)

# 학습 이력에서 최고 성능 추출
best_val_s1 = max(history_s1.history.get('val_accuracy', [float('nan')]))
best_val_s2 = max(history_s2.history.get('val_accuracy', [float('nan')]))

# 결과 요약 출력
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Stage 1: {best_val_s1:.4f}")
print(f"Stage 2: {best_val_s2:.4f}")
print(f"Macro: {per_class_acc.mean():.4f}")
print("\nTraining Complete!")
print("="*70)
