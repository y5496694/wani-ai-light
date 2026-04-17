"""
와니 AI — 카메라 모듈
OpenCV 기반 사진 촬영 기능
"""

import logging
import time
import cv2
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CAMERA_INDEX, TMP_DIR

logger = logging.getLogger(__name__)

class CameraManager:
    """사진 촬영 및 관리 엔진"""

    def __init__(self, device_index: int = CAMERA_INDEX):
        self.device_index = device_index
        self.cap = None

    def capture_photo(self, filename: str = "captured_photo.jpg") -> str | None:
        """
        카메라로 사진을 찍고 파일로 저장.
        
        Args:
            filename: 저장할 파일 이름
            
        Returns:
            저장된 파일의 절대 경로. 실패 시 None.
        """
        output_path = str(TMP_DIR / filename)
        
        try:
            # 카메라 초기화
            self.cap = cv2.VideoCapture(self.device_index)
            if not self.cap.isOpened():
                logger.error(f"카메라({self.device_index})를 열 수 없습니다.")
                return None

            # 조도 조절을 위해 몇 프레임 버림
            for _ in range(5):
                self.cap.read()

            # 사진 촬영
            ret, frame = self.cap.read()
            if not ret:
                logger.error("프레임을 캡처할 수 없습니다.")
                return None

            # 이미지 리사이징 (분석 속도 향상을 위해 최대 512px로 축소)
            max_size = 512
            h, w = frame.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 이미지 저장
            cv2.imwrite(output_path, frame)
            logger.info(f"📸 사진 촬영 완료: {output_path}")
            
            return output_path

        except Exception as e:
            logger.error(f"카메라 촬영 중 오류 발생: {e}")
            return None
        finally:
            if self.cap:
                self.cap.release()
                self.cap = None

    def test_camera(self) -> bool:
        """카메라 작동 여부 테스트"""
        try:
            cap = cv2.VideoCapture(self.device_index)
            is_opened = cap.isOpened()
            cap.release()
            return is_opened
        except Exception:
            return False

if __name__ == "__main__":
    # 간단한 테스트 실행
    logging.basicConfig(level=logging.INFO)
    cam = CameraManager()
    path = cam.capture_photo("test.jpg")
    if path:
        print(f"테스트 성공: {path}")
    else:
        print("테스트 실패")
