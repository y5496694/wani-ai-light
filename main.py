"""
와니 AI Light — 메인 어시스턴트 (경량 버전)
버츄얼 캐릭터 없이 음성 트리거로 사진 촬영 및 분석 수행

실행: python wani_light.py
"""

import logging
import sys
import time
import signal
import io
from pathlib import Path

from config import CHARACTER_NAME, WAKE_WORD, VISION_PROMPT
from modules.audio import AudioManager
from modules.stt import STTEngine
from modules.llm import LLMEngine
from modules.tts import TTSEngine, TTSEngineDummy
from modules.camera import CameraManager

# ──────────────────────────────────────────────
# 로깅 설정
# ──────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("wani-light")

class WaniLight:
    """와니 AI Light 메인 클래스"""

    def __init__(self):
        logger.info(f"🚀 {CHARACTER_NAME} AI Light 시작")
        self._shutdown = False

        # 모듈 초기화
        self.audio = AudioManager()
        self.stt = STTEngine()
        self.llm = LLMEngine()
        self.camera = CameraManager()

        try:
            self.tts = TTSEngine()
        except Exception:
            logger.warning("TTS 초기화 실패, 더미 엔진 사용")
            self.tts = TTSEngineDummy()

        # 시그널 핸들러
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        logger.info("종료 시그널 수신")
        self._shutdown = True

    def run(self):
        """메인 루프: 항상 듣기 -> 호출어 감지 -> 사진 촬영 -> 분석"""
        logger.info(f"🎧 '{WAKE_WORD}' 호출을 기다리는 중...")

        while not self._shutdown:
            try:
                # 1. 음성 녹음 (묵음 시까지)
                audio_file = self.audio.record_until_silence()
                if not audio_file:
                    continue

                # 2. STT 변환
                user_text = self.stt.transcribe(audio_file)
                if not user_text.strip():
                    continue

                logger.info(f"인식됨: '{user_text}'")

                # 3. 호출어 체크 ("와니야" 포함 여부)
                if WAKE_WORD in user_text:
                    logger.info(f"🔔 호출어 '{WAKE_WORD}' 감지! 사진 분석을 시작합니다.")
                    
                    # 알림음 재생
                    ping_file = Path(__file__).parent / "assets" / "ping.wav"
                    if ping_file.exists():
                        self.audio.play_audio(str(ping_file))
                    
                    # 4. 사진 촬영
                    photo_path = self.camera.capture_photo()
                    if not photo_path:
                        self._speak("미안, 카메라를 사용할 수 없어.")
                        continue

                    # 5. 비전 LLM 분석
                    logger.info("🧐 사진 분석 중...")
                    emotion, description = self.llm.analyze_image(photo_path, VISION_PROMPT)
                    
                    logger.info(f"🐊 분석 결과: {description}")

                    # 6. 결과 출력 (TTS)
                    self._speak(description)

            except Exception as e:
                logger.error(f"루프 오류: {e}")
                time.sleep(1)

        self._cleanup()

    def _speak(self, text: str):
        """텍스트를 음성으로 합성 및 재생"""
        try:
            tts_file = self.tts.synthesize(text)
            if tts_file:
                self.audio.play_audio(tts_file)
        except Exception as e:
            logger.error(f"말하기 실패: {e}")

    def _cleanup(self):
        logger.info("리소스 정리 중...")
        self.audio.cleanup()
        self.tts.cleanup_temp_files()
        logger.info("종료되었습니다.")

if __name__ == "__main__":
    app = WaniLight()
    app.run()
