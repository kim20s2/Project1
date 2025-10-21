import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ============================================================
# 1. GPU 설정 (Mixed Precision 제거)
# ============================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)

# TensorFlow 메모리 최적화
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# ============================================================
# 2. 클래스 정의
# ============================================================
FER_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
FER_CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(FER_CLASSES)}

FANE_CLASSES = ['angry', 'confused', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'shy', 'surprise']
FANE_CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(FANE_CLASSES)}

print("FER2013 Classes (7):", FER_CLASSES)
print("FANE Classes (9):", FANE_CLASSES)

# ============================================================
# 3. 데이터 로딩
# ============================================================

def load_image_paths_and_labels(root_dir, class_mapping, domain_label):
    """이미지 경로, 레이블, 도메인 레이블 수집"""
    paths = []
    labels = []
    domains = []

    for class_name, class_idx in class_mapping.items():
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.exists(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(class_dir, img_name))
                labels.append(np.int32(class_idx))
                domains.append(np.int32(domain_label))

    return paths, labels, domains

# FER2013 Train 로딩
fer_train_paths, fer_train_labels, fer_train_domains = load_image_paths_and_labels(
    'fer2013/train', FER_CLASS_TO_IDX, domain_label=0
)

# FER2013 Test 로딩
fer_test_paths, fer_test_labels, fer_test_domains = load_image_paths_and_labels(
    'fer2013/test', FER_CLASS_TO_IDX, domain_label=0
)

# FANE 로딩
fane_paths, fane_labels, fane_domains = load_image_paths_and_labels(
    'fane_data', FANE_CLASS_TO_IDX, domain_label=1
)

# FANE train/val 분할
fane_train_paths, fane_val_paths, fane_train_labels, fane_val_labels = train_test_split(
    fane_paths, fane_labels, test_size=0.2, random_state=42, stratify=fane_labels
)
fane_train_domains = [np.int32(1)] * len(fane_train_paths)
fane_val_domains = [np.int32(1)] * len(fane_val_paths)

print(f"\nFER Train: {len(fer_train_paths)}, FER Test: {len(fer_test_paths)}")
print(f"FANE Train: {len(fane_train_paths)}, FANE Val: {len(fane_val_paths)}")

# ============================================================
# 4. tf.data 파이프라인
# ============================================================

IMG_SIZE = 224

def create_dataset(paths, labels, domains, is_fer=False, augment=False,
                   batch_size=16, shuffle=True):
    """tf.data.Dataset 생성"""

    def parse_function(path, label, domain):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)

        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

        # FER 흑백 → RGB 복제
        if is_fer:
            img = tf.cond(
                tf.equal(tf.shape(img)[2], 1),
                lambda: tf.repeat(img, 3, axis=2),
                lambda: img
            )

        # Normalize to [0, 1] (EfficientNetV2 전처리)
        img = img / 255.0

        # Augmentation
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.9, 1.1)
            img = tf.clip_by_value(img, 0.0, 1.0)

        label = tf.cast(label, tf.int32)
        domain = tf.cast(domain, tf.int32)
        
        return img, label, domain

    paths = np.array(paths, dtype=str)
    labels = np.array(labels, dtype=np.int32)
    domains = np.array(domains, dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels, domains))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(paths))

    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.map(lambda img, lbl, dom: (
        tf.ensure_shape(img, [IMG_SIZE, IMG_SIZE, 3]),
        tf.ensure_shape(lbl, []),
        tf.ensure_shape(dom, [])
    ))

    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    return dataset.prefetch(tf.data.AUTOTUNE)

# ============================================================
# 5. 배치 샘플링
# ============================================================

def create_balanced_dataset(fer_ds, fane_ds, batch_size=16, fer_weight=0.3):
    n_fer = max(1, int(batch_size * fer_weight))
    n_fane = batch_size - n_fer

    fer_batched = fer_ds.batch(n_fer)
    fane_batched = fane_ds.batch(n_fane)

    combined = tf.data.Dataset.zip((fer_batched, fane_batched))

    def merge_batches(fer_batch, fane_batch):
        imgs = tf.concat([fer_batch[0], fane_batch[0]], axis=0)
        labels = tf.concat([fer_batch[1], fane_batch[1]], axis=0)
        domains = tf.concat([fer_batch[2], fane_batch[2]], axis=0)
        return imgs, labels, domains

    combined = combined.map(merge_batches)
    return combined.prefetch(tf.data.AUTOTUNE)

BATCH_SIZE = 16
FER_WEIGHT = 0.3

fer_train_ds = create_dataset(fer_train_paths, fer_train_labels, fer_train_domains,
                              is_fer=True, augment=True, batch_size=None, shuffle=True)
fane_train_ds = create_dataset(fane_train_paths, fane_train_labels, fane_train_domains,
                               is_fer=False, augment=True, batch_size=None, shuffle=True)

train_dataset = create_balanced_dataset(fer_train_ds, fane_train_ds,
                                       batch_size=BATCH_SIZE, fer_weight=FER_WEIGHT)

fer_val_ds = create_dataset(fer_test_paths, fer_test_labels, fer_test_domains,
                            is_fer=True, augment=False, batch_size=BATCH_SIZE, shuffle=False)
fane_val_ds = create_dataset(fane_val_paths, fane_val_labels, fane_val_domains,
                             is_fer=False, augment=False, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nBatch size: {BATCH_SIZE}, FER weight: {FER_WEIGHT}")

# ============================================================
# 6. Hailo 친화적 모델 아키텍처
# ============================================================

def build_hailo_friendly_model(input_shape=(224, 224, 3)):

    inputs = layers.Input(shape=input_shape, name='input_image')

    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    backbone.trainable = True

    x = backbone(inputs)

    # Shared head
    x = layers.Dense(512, activation='relu', name='shared_dense')(x)
    x = layers.BatchNormalization(name='shared_bn')(x)  # ✅ training 제거
    shared_features = layers.Dropout(0.2, name='shared_dropout')(x)

    # FER Head (7 classes)
    fer_x = layers.Dense(256, activation='relu', name='fer_dense1')(shared_features)
    fer_x = layers.BatchNormalization(name='fer_bn')(fer_x)
    fer_x = layers.Dropout(0.1, name='fer_dropout')(fer_x)
    fer_logits = layers.Dense(7, name='fer_logits')(fer_x)
    fer_output = layers.Softmax(dtype='float32', name='fer_output')(fer_logits)  # ✅ dtype 명시

    # FANE Head (9 classes)
    fane_x = layers.Dense(256, activation='relu', name='fane_dense1')(shared_features)
    fane_x = layers.BatchNormalization(name='fane_bn')(fane_x)
    fane_x = layers.Dropout(0.1, name='fane_dropout')(fane_x)
    fane_logits = layers.Dense(9, name='fane_logits')(fane_x)
    fane_output = layers.Softmax(dtype='float32', name='fane_output')(fane_logits)

    # Domain Classifier (2 domains)
    dom_x = layers.Dense(128, activation='relu', name='domain_dense1')(shared_features)
    dom_x = layers.BatchNormalization(name='domain_bn')(dom_x)
    dom_logits = layers.Dense(2, name='domain_logits')(dom_x)
    domain_output = layers.Softmax(dtype='float32', name='domain_output')(dom_logits)

    model = Model(
        inputs=inputs,
        outputs=[fer_output, fane_output, domain_output],
        name='hailo_multidomain_fer'
    )

    return model, backbone

model, backbone = build_hailo_friendly_model()
model.summary()

# ============================================================
# 7. 커스텀 학습 루프
# ============================================================

class MultitaskTrainer:
    def __init__(self, model, backbone):
        self.model = model
        self.backbone = backbone
        self.optimizer = None
        self.loss_weights = {'fer': 2.0, 'fane': 2.0, 'domain': 0.1}
        self.label_smoothing = 0.0

        self.history = {
            'loss': [], 'fer_loss': [], 'fane_loss': [], 'domain_loss': [],
            'val_fer_f1': [], 'val_fane_f1': [], 'val_domain_acc': []
        }

    def set_stage(self, stage, learning_rate, loss_weights, label_smoothing):
        self.loss_weights = loss_weights
        self.label_smoothing = label_smoothing

        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-5,
            clipnorm=1.0,
            epsilon=1e-7
        )

        if stage == 1:
            # Freeze 50% of backbone
            total_blocks = len(self.backbone.layers)
            freeze_until = total_blocks // 2

            for i, layer in enumerate(self.backbone.layers):
                layer.trainable = (i >= freeze_until)

        elif stage == 2:
            # Unfreeze top 70%
            total_blocks = len(self.backbone.layers)
            freeze_until = int(total_blocks * 0.3)

            for i, layer in enumerate(self.backbone.layers):
                layer.trainable = (i >= freeze_until)

        elif stage == 3:
            # Full fine-tune
            self.backbone.trainable = True

        trainable_count = sum([tf.size(v).numpy() for v in self.model.trainable_variables])
        total_count = sum([tf.size(v).numpy() for v in self.model.variables])

        print(f"\n{'='*60}")
        print(f"Stage {stage} Configuration:")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Loss Weights: {loss_weights}")
        print(f"  Trainable params: {trainable_count:,} / {total_count:,}")
        print(f"{'='*60}\n")

    @tf.function
    def train_step(self, images, labels, domains):
        with tf.GradientTape() as tape:
            fer_pred, fane_pred, domain_pred = self.model(images, training=True)

            fer_labels_oh = tf.one_hot(labels, depth=7)
            fane_labels_oh = tf.one_hot(labels, depth=9)
            domain_labels_oh = tf.one_hot(domains, depth=2)

            mask_fer = tf.cast(tf.equal(domains, 0), tf.float32)
            mask_fane = tf.cast(tf.equal(domains, 1), tf.float32)

            cce = tf.keras.losses.CategoricalCrossentropy(
                reduction='none',
                label_smoothing=self.label_smoothing
            )

            losses_fer = cce(fer_labels_oh, fer_pred)
            losses_fane = cce(fane_labels_oh, fane_pred)
            losses_domain = cce(domain_labels_oh, domain_pred)

            loss_fer = tf.reduce_sum(losses_fer * mask_fer) / (tf.reduce_sum(mask_fer) + 1e-8)
            loss_fane = tf.reduce_sum(losses_fane * mask_fane) / (tf.reduce_sum(mask_fane) + 1e-8)
            loss_domain = tf.reduce_mean(losses_domain)

            total_loss = (
                self.loss_weights['fer'] * loss_fer +
                self.loss_weights['fane'] * loss_fane +
                self.loss_weights['domain'] * loss_domain
            )

            total_loss = tf.where(tf.math.is_nan(total_loss), tf.constant(0.0), total_loss)

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        gradients = [
            tf.where(tf.math.is_nan(g), tf.zeros_like(g), g) if g is not None else g
            for g in gradients
        ]
        
        grad_norm = tf.sqrt(sum([tf.reduce_sum(tf.square(g)) for g in gradients if g is not None]))
        
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return total_loss, loss_fer, loss_fane, loss_domain, grad_norm

    def train_epoch(self, dataset, steps_per_epoch):
        epoch_loss = 0.0
        epoch_fer_loss = 0.0
        epoch_fane_loss = 0.0
        epoch_domain_loss = 0.0
        grad_norms = []

        for step, (images, labels, domains) in enumerate(dataset.take(steps_per_epoch)):
            total_loss, fer_loss, fane_loss, domain_loss, grad_norm = self.train_step(
                images, labels, domains
            )

            epoch_loss += total_loss.numpy()
            epoch_fer_loss += fer_loss.numpy()
            epoch_fane_loss += fane_loss.numpy()
            epoch_domain_loss += domain_loss.numpy()
            grad_norms.append(grad_norm.numpy())

            if (step + 1) % 50 == 0:
                print(f"  Step {step+1}/{steps_per_epoch} - "
                      f"Loss: {total_loss:.4f}, FER: {fer_loss:.4f}, "
                      f"FANE: {fane_loss:.4f}, Domain: {domain_loss:.4f}")

        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0

        return {
            'loss': epoch_loss / steps_per_epoch,
            'fer_loss': epoch_fer_loss / steps_per_epoch,
            'fane_loss': epoch_fane_loss / steps_per_epoch,
            'domain_loss': epoch_domain_loss / steps_per_epoch,
            'avg_grad_norm': avg_grad_norm
        }

    def evaluate(self, fer_dataset, fane_dataset):
        fer_y_true, fer_y_pred, fer_domain_pred = [], [], []

        for images, labels, domains in fer_dataset:
            fer_pred, _, domain_pred = self.model(images, training=False)

            fer_y_true.extend(labels.numpy())
            fer_y_pred.extend(tf.argmax(fer_pred, axis=1).numpy())
            fer_domain_pred.extend(tf.argmax(domain_pred, axis=1).numpy())

        fer_f1 = f1_score(fer_y_true, fer_y_pred, average='macro')
        fer_domain_acc = np.mean(np.array(fer_domain_pred) == 0)

        fane_y_true, fane_y_pred, fane_domain_pred = [], [], []

        for images, labels, domains in fane_dataset:
            _, fane_pred, domain_pred = self.model(images, training=False)

            fane_y_true.extend(labels.numpy())
            fane_y_pred.extend(tf.argmax(fane_pred, axis=1).numpy())
            fane_domain_pred.extend(tf.argmax(domain_pred, axis=1).numpy())

        fane_f1 = f1_score(fane_y_true, fane_y_pred, average='macro')
        fane_domain_acc = np.mean(np.array(fane_domain_pred) == 1)

        domain_acc = (fer_domain_acc * len(fer_y_true) + fane_domain_acc * len(fane_y_true)) / \
                     (len(fer_y_true) + len(fane_y_true))

        return {
            'fer_f1': fer_f1,
            'fane_f1': fane_f1,
            'domain_acc': domain_acc,
            'fer_preds': (fer_y_true, fer_y_pred),
            'fane_preds': (fane_y_true, fane_y_pred)
        }

    def fit(self, train_dataset, fer_val_ds, fane_val_ds, epochs, steps_per_epoch):
        best_f1 = 0.0
        patience = 5
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 60)

            train_metrics = self.train_epoch(train_dataset, steps_per_epoch)
            val_metrics = self.evaluate(fer_val_ds, fane_val_ds)

            self.history['loss'].append(train_metrics['loss'])
            self.history['fer_loss'].append(train_metrics['fer_loss'])
            self.history['fane_loss'].append(train_metrics['fane_loss'])
            self.history['domain_loss'].append(train_metrics['domain_loss'])
            self.history['val_fer_f1'].append(val_metrics['fer_f1'])
            self.history['val_fane_f1'].append(val_metrics['fane_f1'])
            self.history['val_domain_acc'].append(val_metrics['domain_acc'])

            print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
                f"FER: {train_metrics['fer_loss']:.4f}, "
                f"FANE: {train_metrics['fane_loss']:.4f}")
            print(f"Val   - FER F1: {val_metrics['fer_f1']:.4f}, "
                f"FANE F1: {val_metrics['fane_f1']:.4f}")

            avg_f1 = (val_metrics['fer_f1'] + val_metrics['fane_f1']) / 2
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                patience_counter = 0
                self.model.save_weights('best_model_hailo.weights.h5')
                print(f"✓ Best model saved (F1: {best_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping (patience={patience})")
                    break

# ============================================================
# 8. 학습 실행
# ============================================================

trainer = MultitaskTrainer(model, backbone)
steps_per_epoch = (len(fer_train_paths) + len(fane_train_paths)) // BATCH_SIZE

# Stage 1
print("\n" + "="*60)
print("STAGE 1: HEAD INITIALIZATION")
print("="*60)
trainer.set_stage(
    stage=1,
    learning_rate=1e-3,
    loss_weights={'fer': 2.0, 'fane': 2.0, 'domain': 0.1},
    label_smoothing=0.0
)
trainer.fit(train_dataset, fer_val_ds, fane_val_ds, epochs=15, steps_per_epoch=steps_per_epoch)

# Stage 2
print("\n" + "="*60)
print("STAGE 2: DOMAIN ADAPTATION")
print("="*60)
trainer.set_stage(
    stage=2,
    learning_rate=1e-4,
    loss_weights={'fer': 2.0, 'fane': 2.0, 'domain': 0.2},
    label_smoothing=0.03
)
trainer.fit(train_dataset, fer_val_ds, fane_val_ds, epochs=15, steps_per_epoch=steps_per_epoch)

# Stage 3
print("\n" + "="*60)
print("STAGE 3: FULL FINE-TUNE")
print("="*60)
trainer.set_stage(
    stage=3,
    learning_rate=5e-5,
    loss_weights={'fer': 2.5, 'fane': 2.5, 'domain': 0.1},
    label_smoothing=0.02
)
trainer.fit(train_dataset, fer_val_ds, fane_val_ds, epochs=20, steps_per_epoch=steps_per_epoch)

model.load_weights('best_model_hailo.weights.h5')
print("\n✓ Best model weights loaded")

# ============================================================
# 9. 최종 평가
# ============================================================

print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

final_metrics = trainer.evaluate(fer_val_ds, fane_val_ds)

print(f"\nFER Test - Macro F1: {final_metrics['fer_f1']:.4f}")
print(f"FANE Val - Macro F1: {final_metrics['fane_f1']:.4f}")
print(f"Domain Acc: {final_metrics['domain_acc']:.4f}")

# Confusion matrices
fer_y_true, fer_y_pred = final_metrics['fer_preds']
fane_y_true, fane_y_pred = final_metrics['fane_preds']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cm_fer = confusion_matrix(fer_y_true, fer_y_pred)
sns.heatmap(cm_fer, annot=True, fmt='d', cmap='Blues',
            xticklabels=FER_CLASSES, yticklabels=FER_CLASSES, ax=axes[0])
axes[0].set_title('FER2013 Confusion Matrix')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

cm_fane = confusion_matrix(fane_y_true, fane_y_pred)
sns.heatmap(cm_fane, annot=True, fmt='d', cmap='Greens',
            xticklabels=FANE_CLASSES, yticklabels=FANE_CLASSES, ax=axes[1])
axes[1].set_title('FANE Confusion Matrix')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('confusion_matrices_hailo.png', dpi=300, bbox_inches='tight')
print("\n✓ Confusion matrices saved")

print("\n" + "="*60)
print("FER2013 Classification Report:")
print("="*60)
print(classification_report(fer_y_true, fer_y_pred, target_names=FER_CLASSES))

print("\n" + "="*60)
print("FANE Classification Report:")
print("="*60)
print(classification_report(fane_y_true, fane_y_pred, target_names=FANE_CLASSES))

# ============================================================
# 10. Hailo-8 Export (.keras → .tflite 직접 변환)
# ============================================================

print("\n" + "="*60)
print("EXPORTING FOR HAILO-8")
print("="*60)

# Step 1: Inference 전용 Keras 모델 생성
print("\n[Step 1: Creating Inference Model]")

input_layer = layers.Input(shape=(224, 224, 3), batch_size=1, name='input_image')
fer_out, fane_out, domain_out = model(input_layer, training=False)

inference_model = Model(
    inputs=input_layer,
    outputs={
        'fer_output': fer_out,
        'fane_output': fane_out,
        'domain_output': domain_out
    },
    name='hailo_inference'
)

# 모든 레이어 non-trainable 설정
inference_model.trainable = False
for layer in inference_model.layers:
    layer.trainable = False

print("✓ Inference model created")
inference_model.summary()

# Step 2: .keras 파일로 저장
keras_path = 'hailo_model.keras'
print(f"\n[Step 2: Saving to {keras_path}]")

try:
    inference_model.save(keras_path, save_format='keras')
    file_size = os.path.getsize(keras_path) / 1024 / 1024
    print(f"✓ Keras model saved: {file_size:.2f} MB")
except Exception as e:
    print(f"✗ Failed to save Keras model: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 11. .keras → .tflite 변환 (옵션 최소화)
# ============================================================

print("\n" + "="*60)
print("CONVERTING .keras TO .tflite (MINIMAL OPTIONS)")
print("="*60)

#  변환 1: Float32 TFLite (옵션 없음)
print("\n[Conversion 1: Float32 TFLite - No Optimization]")

try:
    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    
    #  모든 최적화 옵션 제거
    # converter.optimizations = []  # 기본값
    # converter.target_spec = None  # 기본값
    
    print("Converting to TFLite (Float32, no optimization)...")
    tflite_model_float32 = converter.convert()
    
    # 저장
    float32_path = 'hailo_model_float32.tflite'
    with open(float32_path, 'wb') as f:
        f.write(tflite_model_float32)
    
    float32_size = len(tflite_model_float32) / 1024 / 1024
    print(f"✓ Float32 TFLite saved: {float32_path}")
    print(f"  Size: {float32_size:.2f} MB")
    
except Exception as e:
    print(f"✗ Float32 conversion failed: {e}")
    import traceback
    traceback.print_exc()

#  변환 2: Float32 TFLite (기본 최적화만)
print("\n[Conversion 2: Float32 TFLite - Default Optimization]")

try:
    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    
    # 기본 최적화만 (가장 안전)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    print("Converting to TFLite (Float32, default optimization)...")
    tflite_model_optimized = converter.convert()
    
    # 저장
    optimized_path = 'hailo_model_float32_optimized.tflite'
    with open(optimized_path, 'wb') as f:
        f.write(tflite_model_optimized)
    
    optimized_size = len(tflite_model_optimized) / 1024 / 1024
    print(f"✓ Optimized Float32 TFLite saved: {optimized_path}")
    print(f"  Size: {optimized_size:.2f} MB")
    
except Exception as e:
    print(f"✗ Optimized conversion failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 12. TFLite 모델 검증
# ============================================================

print("\n" + "="*60)
print("VALIDATING TFLITE MODELS")
print("="*60)

def validate_tflite_model(tflite_path):
    """TFLite 모델 검증"""
    if not os.path.exists(tflite_path):
        print(f"\n✗ File not found: {tflite_path}")
        return False
    
    print(f"\n[Validating: {tflite_path}]")
    
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # 입력 정보
        print("\n Input Details:")
        for detail in input_details:
            print(f"  Name:  {detail['name']}")
            print(f"  Shape: {detail['shape']}")
            print(f"  Type:  {detail['dtype'].__name__}")
        
        # 출력 정보
        print("\n Output Details:")
        for i, detail in enumerate(output_details):
            print(f"\n  Output {i}:")
            print(f"    Name:  {detail['name']}")
            print(f"    Shape: {detail['shape']}")
            print(f"    Type:  {detail['dtype'].__name__}")
        
        # 테스트 추론
        print("\n Test Inference:")
        test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        
        print("  Status: Inference successful")
        
        # 출력 통계
        for i, detail in enumerate(output_details):
            output_data = interpreter.get_tensor(detail['index'])
            print(f"\n  Output {i} Statistics:")
            print(f"    Shape: {output_data.shape}")
            print(f"    Type:  {output_data.dtype}")
            print(f"    Range: [{output_data.min():.4f}, {output_data.max():.4f}]")
            print(f"    Mean:  {output_data.mean():.4f}")
            
            # Softmax 출력 확인
            if output_data.shape[-1] in [7, 9, 2]:  
                prob_sum = output_data.sum()
                print(f"    Sum:   {prob_sum:.6f} {'✓' if 0.99 < prob_sum < 1.01 else '⚠️'}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# 검증 실행
validate_tflite_model('hailo_model_float32.tflite')
validate_tflite_model('hailo_model_float32_optimized.tflite')