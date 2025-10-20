import numpy as np
import cv2

class SCRFDDecoder:
    def __init__(self, input_size=640, original_size=None, distance_scale=16.0):
        # Feature pyramid strides
        self.strides = [8, 16, 32]
        self.feature_maps = [(80, 80), (40, 40), (20, 20)]
        
        # UINT8 quantization parameters (모델에 따라 조정 필요)
        self.score_scale = 1.0 / 255.0
        self.bbox_scale = 1.0 / 255.0
        
        # Distance scaling factor (중요!)
        # 기본값 16.0, bbox가 너무 작으면 증가, 너무 크면 감소
        self.distance_scale = distance_scale
        
        # Aspect ratio correction
        self.input_size = input_size
        self.original_size = original_size  # (width, height)
        if original_size:
            self.scale_x = original_size[0] / input_size
            self.scale_y = original_size[1] / input_size
        else:
            self.scale_x = 1.0
            self.scale_y = 1.0
        
    def dequantize_uint8(self, data, scale=1.0/255.0, zero_point=0):
        """UINT8 데이터를 float으로 변환"""
        return (data.astype(np.float32) - zero_point) * scale
    
    def distance2bbox(self, points, distance, stride):
        """Distance format을 bbox로 변환
        
        Args:
            points: anchor points (N, 2) - [x, y]
            distance: distances (N, 4) - [left, top, right, bottom]
            stride: feature stride
        
        Returns:
            bboxes (N, 4) - [x1, y1, x2, y2]
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)
    
    def generate_anchors(self, height, width, stride):
        """Anchor points 생성"""
        shift_x = np.arange(0, width) * stride
        shift_y = np.arange(0, height) * stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        
        # Center points
        anchor_centers = np.stack([shift_x, shift_y], axis=-1).reshape(-1, 2)
        anchor_centers = anchor_centers + stride // 2  # center offset
        
        return anchor_centers
    
    def decode_single_level(self, score_map, bbox_map, height, width, stride, 
                           conf_threshold=0.5):
        """단일 feature level에서 detection 추출
        
        Args:
            score_map: (H, W, 2) - UINT8
            bbox_map: (H, W, 8) - UINT8, 4 distances × 2
            height, width: feature map 크기
            stride: feature stride
            conf_threshold: confidence threshold
        
        Returns:
            bboxes, scores
        """
        # Dequantize
        scores = self.dequantize_uint8(score_map, self.score_scale)
        bboxes_raw = self.dequantize_uint8(bbox_map, self.bbox_scale)
        
        # Softmax for scores (배경 vs 얼굴)
        scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        scores = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
        face_scores = scores[:, :, 1]  # 얼굴 confidence
        
        # Threshold filtering
        mask = face_scores > conf_threshold
        if not np.any(mask):
            return np.array([]), np.array([])
        
        # Get valid positions
        indices = np.argwhere(mask)  # (N, 2) - [y, x]
        valid_scores = face_scores[mask]
        
        # Generate anchor centers for valid positions
        anchor_centers = (indices[:, [1, 0]] * stride + stride // 2).astype(np.float32)
        
        # Extract bbox distances (첫 4채널 사용, 나머지 4채널은 refinement)
        bbox_distances = bboxes_raw[indices[:, 0], indices[:, 1], :4]
        
        # Distance format을 실제 거리로 변환
        # stride * distance_scale이 핵심 파라미터
        # UINT8 (0-255)를 픽셀 거리로 변환
        bbox_distances = bbox_distances * self.distance_scale
        
        # Convert to bbox format [x1, y1, x2, y2]
        bboxes = self.distance2bbox(anchor_centers, bbox_distances, stride)
        
        return bboxes, valid_scores
    
    def nms(self, bboxes, scores, iou_threshold=0.4):
        """Non-Maximum Suppression"""
        if len(bboxes) == 0:
            return np.array([], dtype=np.int32)
        
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep, dtype=np.int32)
    
    def detect(self, outputs, conf_threshold=0.5, nms_threshold=0.4):
        """전체 detection 수행
        
        Args:
            outputs: 모델 출력 딕셔너리
                {
                    'conv26': (80, 80, 2),  # stride 8 score
                    'conv27': (80, 80, 8),  # stride 8 bbox
                    'conv32': (40, 40, 2),  # stride 16 score
                    'conv33': (40, 40, 8),  # stride 16 bbox
                    'conv38': (20, 20, 2),  # stride 32 score
                    'conv39': (20, 20, 8),  # stride 32 bbox
                }
        
        Returns:
            final_bboxes: (N, 4) - [x1, y1, x2, y2]
            final_scores: (N,)
        """
        all_bboxes = []
        all_scores = []
        
        # Level 0: stride 8 (80x80)
        bboxes, scores = self.decode_single_level(
            outputs['conv26'], outputs['conv27'], 
            80, 80, 8, conf_threshold
        )
        if len(bboxes) > 0:
            all_bboxes.append(bboxes)
            all_scores.append(scores)
        
        # Level 1: stride 16 (40x40)
        bboxes, scores = self.decode_single_level(
            outputs['conv32'], outputs['conv33'],
            40, 40, 16, conf_threshold
        )
        if len(bboxes) > 0:
            all_bboxes.append(bboxes)
            all_scores.append(scores)
        
        # Level 2: stride 32 (20x20)
        bboxes, scores = self.decode_single_level(
            outputs['conv38'], outputs['conv39'],
            20, 20, 32, conf_threshold
        )
        if len(bboxes) > 0:
            all_bboxes.append(bboxes)
            all_scores.append(scores)
        
        if len(all_bboxes) == 0:
            return np.array([]), np.array([])
        
        # Concatenate all detections
        all_bboxes = np.concatenate(all_bboxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        
        # NMS (640x640 좌표계에서 수행)
        keep_indices = self.nms(all_bboxes, all_scores, nms_threshold)
        
        final_bboxes = all_bboxes[keep_indices]
        final_scores = all_scores[keep_indices]
        
        # ===== 후처리: Aspect ratio 보정 =====
        if self.original_size:
            final_bboxes = self.correct_aspect_ratio(final_bboxes)
        
        # Clip to image boundaries (원본 이미지 크기 기준)
        img_w = self.original_size[0] if self.original_size else 640
        img_h = self.original_size[1] if self.original_size else 640
        final_bboxes[:, [0, 2]] = np.clip(final_bboxes[:, [0, 2]], 0, img_w)
        final_bboxes[:, [1, 3]] = np.clip(final_bboxes[:, [1, 3]], 0, img_h)
        
        return final_bboxes, final_scores
    
    def correct_aspect_ratio(self, bboxes):
        """
        Stretch된 이미지(640x640)의 bbox를 원본 비율로 보정
        
        예: 640x480 이미지가 640x640으로 stretch됨
        - X축: 640 → 640 (변화 없음, scale=1.0)
        - Y축: 480 → 640 (늘어남, scale=0.75)
        
        Args:
            bboxes: (N, 4) - [x1, y1, x2, y2] in 640x640 coordinate
        
        Returns:
            bboxes_corrected: (N, 4) - [x1, y1, x2, y2] in original coordinate
        """
        if len(bboxes) == 0:
            return bboxes
        
        bboxes_corrected = bboxes.copy()
        
        # X축 보정 (width)
        bboxes_corrected[:, [0, 2]] *= self.scale_x
        
        # Y축 보정 (height)
        bboxes_corrected[:, [1, 3]] *= self.scale_y
        
        return bboxes_corrected


# 사용 예제
def example_usage():
    """모델 출력으로부터 bbox 추출 예제"""
    
    # 모델 출력 (UINT8 형태로 가정)
    outputs = {
        'conv26': np.random.randint(0, 255, (80, 80, 2), dtype=np.uint8),
        'conv27': np.random.randint(0, 255, (80, 80, 8), dtype=np.uint8),
        'conv32': np.random.randint(0, 255, (40, 40, 2), dtype=np.uint8),
        'conv33': np.random.randint(0, 255, (40, 40, 8), dtype=np.uint8),
        'conv38': np.random.randint(0, 255, (20, 20, 2), dtype=np.uint8),
        'conv39': np.random.randint(0, 255, (20, 20, 8), dtype=np.uint8),
    }
    
    # ===== Distance Scale 조정이 핵심 =====
    # distance_scale 값에 따른 bbox 크기 변화:
    # - 4.0: 매우 작은 bbox (코/입만)
    # - 8.0: 작은 bbox
    # - 16.0: 보통 bbox (기본값)
    # - 32.0: 큰 bbox
    # - 64.0: 매우 큰 bbox
    
    decoder = SCRFDDecoder(
        input_size=640,
        original_size=(640, 480),  # (width, height)
        distance_scale=16.0  # ⭐ 이 값을 조정하세요!
    )
    
    bboxes, scores = decoder.detect(
        outputs, 
        conf_threshold=0.5,
        nms_threshold=0.4
    )
    
    print(f"검출된 얼굴 수: {len(bboxes)}")
    debug_bbox_sizes(bboxes)
    
    for i, (bbox, score) in enumerate(zip(bboxes, scores)):
        x1, y1, x2, y2 = bbox #y2 -> y, x1 -> x
        w, h = x2 - x1, y2 - y1
        print(f"Face {i}: pos=({x1:.1f}, {y1:.1f}), size=({w:.1f}x{h:.1f}), score={score:.3f}")
    
    # 시각화 (원본 640x480 이미지에 그리기)
    # img = cv2.imread('your_640x480_image.jpg')
    # for bbox, score in zip(bboxes, scores):
    #     x1, y1, x2, y2 = bbox.astype(int)
    #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText(img, f'{score:.2f}', (x1, y1-5), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # cv2.imshow('Detection Result', img)
    # cv2.waitKey(0)


def find_optimal_distance_scale(outputs, target_bbox_size=(80, 120)):
    """
    최적의 distance_scale 값을 찾기 위한 테스트 함수
    
    Args:
        outputs: 모델 출력
        target_bbox_size: 목표 bbox 크기 (width, height)
    """
    scales_to_test = [4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0]
    
    print("Testing different distance_scale values:")
    print("=" * 60)
    
    for scale in scales_to_test:
        decoder = SCRFDDecoder(
            input_size=640,
            original_size=(640, 480),
            distance_scale=scale
        )
        
        bboxes, scores = decoder.detect(outputs, conf_threshold=0.3)
        
        if len(bboxes) > 0:
            widths = bboxes[:, 2] - bboxes[:, 0]
            heights = bboxes[:, 3] - bboxes[:, 1]
            print(f"scale={scale:5.1f} → size: {widths.mean():.1f}x{heights.mean():.1f} "
                  f"({len(bboxes)} detections)")
        else:
            print(f"scale={scale:5.1f} → No detections")
    
    print("=" * 60)
    print(f"Target size: {target_bbox_size[0]}x{target_bbox_size[1]}")
    print("Choose the scale that produces bbox size closest to your target.")


def debug_bbox_sizes(bboxes):
    """Bbox 크기 분포 확인 (디버깅용)"""
    if len(bboxes) == 0:
        print("No bboxes detected")
        return
    
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]
    areas = widths * heights
    
    print("\n=== Bbox Size Statistics ===")
    print(f"Width  - min: {widths.min():.1f}, max: {widths.max():.1f}, mean: {widths.mean():.1f}")
    print(f"Height - min: {heights.min():.1f}, max: {heights.max():.1f}, mean: {heights.mean():.1f}")
    print(f"Area   - min: {areas.min():.1f}, max: {areas.max():.1f}, mean: {areas.mean():.1f}")
    print(f"Aspect ratio - mean: {(widths/heights).mean():.2f}")
    
    # 크기 평가
    avg_width = widths.mean()
    avg_height = heights.mean()
    
    print("\n=== Size Evaluation ===")
    if avg_width < 60 or avg_height < 80:
        print("❌ Bboxes are TOO SMALL (catching only nose/mouth)")
        print(f"   → Increase distance_scale (try {16.0 * 80 / avg_height:.1f})")
    elif avg_width > 150 or avg_height > 200:
        print("❌ Bboxes are TOO LARGE (including too much background)")
        print(f"   → Decrease distance_scale (try {16.0 * 120 / avg_height:.1f})")
    else:
        print("✅ Bbox sizes look reasonable!")
        print(f"   Current distance_scale seems appropriate")

if __name__ == "__main__":
    example_usage()