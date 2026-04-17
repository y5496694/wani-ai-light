"""
와니 AI 어시스턴트 — 설정 파일
Raspberry Pi 5 (8GB) 기반 버츄얼 로컬 AI
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
LIVE2D_MODEL_DIR = MODELS_DIR / "wani"
WHISPER_DIR = BASE_DIR / "whisper.cpp"
WHISPER_MODEL = WHISPER_DIR / "models" / "ggml-tiny.bin"
WHISPER_BIN = WHISPER_DIR / "build" / "bin" / "whisper-cli"
TMP_DIR = Path("/tmp/wani-ai")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# 디스플레이 설정
# ──────────────────────────────────────────────
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 480
TARGET_FPS = 30
WINDOW_TITLE = "와니 AI"
BACKGROUND_COLOR = (30, 30, 40)  # 어두운 남색 배경

# ──────────────────────────────────────────────
# LLM 설정 (Gemma4 E2B via Ollama)
# ──────────────────────────────────────────────
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:e2b"     # 일반 대화용 모델
VISION_MODEL = "moondream"      # 사진 분석용 경량 모델 (속도 향상)
LLM_CONTEXT_LENGTH = 2048  # RAM 절약을 위해 2048 권장 (최대 4096)
LLM_TEMPERATURE = 0.7
LLM_MAX_HISTORY = 20  # 대화 기록 최대 유지 수 (메모리 관리)

# ──────────────────────────────────────────────
# 카메라 및 비전 설정
# ──────────────────────────────────────────────
CAMERA_INDEX = 0               # 연결된 카메라 장치 인덱스
VISION_PROMPT = "이 사진에서 무엇이 보여? 한국어로 짧고 친절하게 설명해줘."

# ──────────────────────────────────────────────
# 캐릭터 시스템 프롬프트
# ──────────────────────────────────────────────
CHARACTER_NAME = "와니"

SYSTEM_PROMPT = f"""너는 "{CHARACTER_NAME}"라는 이름의 유능하고 친절한 AI 어시스턴트야.
사진을 분석할 때는 있는 그대로 객관적이고 정확하게 설명해야 해.
항상 한국어로 대화하며, 정중하고 친절한 존댓말을 사용해줘.
응답은 간결하면서도 핵심적인 정보를 포함해야 해.

감정을 표현할 때는 응답 맨 앞에 [감정] 태그를 반드시 넣어줘.
사용 가능한 감정: [기쁨], [슬픔], [놀람], [분노], [평온], [부끄러움]
항상 하나의 감정 태그로 시작해야 해.

예시:
- [평온] 현재 사진에는 책상 위에 노트북과 머그컵이 놓여 있는 것으로 보입니다.
- [기쁨] 요청하신 사진 분석을 완료했습니다. 도움이 필요하시면 언제든 말씀해 주세요.
- [놀람] 아, 조명이 조금 어두워서 분석에 시간이 더 걸릴 수 있습니다.
"""

# ──────────────────────────────────────────────
# STT 설정 (Whisper.cpp)
# ──────────────────────────────────────────────
STT_LANGUAGE = "ko"
STT_THREADS = 2  # CPU 스레드 (LLM 추론과 겹치지 않게 2개)
STT_SAMPLE_RATE = 16000

# ──────────────────────────────────────────────
# TTS 설정 (Supertone Supertonic)
# ──────────────────────────────────────────────
TTS_ENGINE_TYPE = "supertonic" # 'melo' 또는 'supertonic'

# Supertonic 설정
SUPERTONIC_ASSETS_DIR = MODELS_DIR / "supertonic" / "assets"
SUPERTONIC_VOICE_STYLE = "F2"  # 사용자가 요청한 F2 여성 음성
TTS_SPEED = 1.0                # 재생 속도
TTS_OUTPUT_FILE = TMP_DIR / "wani_response.wav"

# ──────────────────────────────────────────────
# 오디오 설정
# ──────────────────────────────────────────────
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CHUNK_SIZE = 1024
AUDIO_FORMAT_WIDTH = 2  # 16-bit (2 bytes)

# VAD (Voice Activity Detection) 설정
VAD_SILENCE_THRESHOLD = 500  # RMS 기준 묵음 판단 임계값
VAD_SILENCE_DURATION = 1.5   # 이 시간(초) 동안 묵음이면 녹음 종료
VAD_MIN_SPEECH_DURATION = 0.5  # 최소 음성 길이 (짧은 노이즈 무시)
VAD_MAX_RECORD_DURATION = 15.0  # 최대 녹음 시간

# ──────────────────────────────────────────────
# Wake Word 설정 (선택)
# ──────────────────────────────────────────────
USE_WAKE_WORD = True  # True면 호출어 방식, False면 항상 듣기
WAKE_WORDS = ["와니야", "원이야", "와미야", "왔냐", "완이야", "와니", "와 미야"]
WAKE_WORD = WAKE_WORDS[0]  # 하위 호환성 유지

# ──────────────────────────────────────────────
# Live2D 감정 매핑
# ──────────────────────────────────────────────
EMOTION_MAP = {
    "기쁨": {
        "expression": "happy",
        "motion_group": "happy",
        "param_overrides": {
            "ParamEyeLSmile": 1.0,
            "ParamEyeRSmile": 1.0,
            "ParamMouthForm": 1.0,
        }
    },
    "슬픔": {
        "expression": "sad",
        "motion_group": "sad",
        "param_overrides": {
            "ParamEyeLOpen": 0.5,
            "ParamEyeROpen": 0.5,
            "ParamBrowLY": -0.5,
            "ParamBrowRY": -0.5,
        }
    },
    "놀람": {
        "expression": "surprise",
        "motion_group": "surprise",
        "param_overrides": {
            "ParamEyeLOpen": 1.3,
            "ParamEyeROpen": 1.3,
            "ParamMouthOpenY": 0.5,
        }
    },
    "분노": {
        "expression": "angry",
        "motion_group": "angry",
        "param_overrides": {
            "ParamBrowLAngle": -1.0,
            "ParamBrowRAngle": -1.0,
            "ParamMouthForm": -0.5,
        }
    },
    "평온": {
        "expression": "neutral",
        "motion_group": "idle",
        "param_overrides": {}
    },
    "부끄러움": {
        "expression": "shy",
        "motion_group": "shy",
        "param_overrides": {
            "ParamEyeLOpen": 0.6,
            "ParamEyeROpen": 0.6,
            "ParamEyeBallX": -0.3,
            "ParamCheek": 1.0,
        }
    },
}

# ──────────────────────────────────────────────
# 상태 정의
# ──────────────────────────────────────────────
class AppState:
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"
