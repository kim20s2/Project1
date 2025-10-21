import tensorflow as tf

# .keras 모델 로드
model = tf.keras.models.load_model('multidomain_inference_simple.keras', compile=False)

# TFLite 변환기 생성
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# TFLite 모델로 변환
tflite_model = converter.convert()

# .tflite 파일로 저장
with open('multidomain_inference_simple.tflite', 'wb') as f:
    f.write(tflite_model)

print("변환 완료: multidomain_inference_simple.tflite")


# import tensorflow as tf

# print("=== TFLite 변환 (최적화된 모델) ===")

# # 최적화된 .keras 모델 로드
# model = tf.keras.models.load_model('best_model_float32_3class_optimized.keras', compile=False)

# # TFLite 변환기 생성
# converter = tf.lite.TFLiteConverter.from_keras_model(model)

# # 최적화 옵션 (선택사항)
# # converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 양자화는 Hailo에서 수행

# # TFLite 모델로 변환
# tflite_model = converter.convert()

# # .tflite 파일로 저장
# output_path = 'best_model_float32_3class_optimized.tflite'
# with open(output_path, 'wb') as f:
#     f.write(tflite_model)

# print(f"✓ 변환 완료: {output_path}")
# print(f"  파일 크기: {len(tflite_model) / (1024*1024):.2f} MB")

# # ============================================
# # TFLite 모델 검증
# # ============================================
# import numpy as np

# interpreter = tf.lite.Interpreter(model_path=output_path)
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# print(f"\n=== TFLite 모델 정보 ===")
# print(f"입력 이름: {input_details[0]['name']}")
# print(f"입력 shape: {input_details[0]['shape']}")
# print(f"출력 이름: {output_details[0]['name']}")
# print(f"출력 shape: {output_details[0]['shape']}")

# # 테스트
# test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
# interpreter.set_tensor(input_details[0]['index'], test_input)
# interpreter.invoke()
# tflite_output = interpreter.get_tensor(output_details[0]['index'])

# print(f"\nTFLite 출력: {tflite_output.flatten()}")
# print(f"예측 클래스: {np.argmax(tflite_output)}")

# # import tensorflow as tf

# # # .keras 모델 로드
# # model = tf.keras.models.load_model('best_model_float32_3class.keras', compile=False)

# # # TFLite 변환기 생성
# # converter = tf.lite.TFLiteConverter.from_keras_model(model)

# # # TFLite 모델로 변환
# # tflite_model = converter.convert()

# # # .tflite 파일로 저장
# # with open('best_model_float32_3class.tflite', 'wb') as f:
# #     f.write(tflite_model)

# # print("변환 완료: best_model_float32_3class.tflite")