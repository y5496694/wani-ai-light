# Wani-AI Light (와니 AI 라이트) 🐊✨

Raspberry Pi 5 및 기타 로컬 환경을 위한 초경량 음성 트리거 비전 AI 어시스턴트입니다.

## 주요 기능
- **음성 호출어 감지**: "와니야"라고 부르면 알림음과 함께 작동합니다.
- **사진 촬영 및 분석**: USB 카메라를 통해 사진을 찍고 Gemma 4 비전 모델로 분석합니다.
- **객관적이고 친절한 설명**: 촬영된 결과를 정중한 한국어로 설명합니다.
- **로컬 실행**: Ollama, Whisper.cpp, Supertonic TTS를 사용하여 모든 과정을 로컬에서 처리합니다.

## 설치 및 실행 방법

1. **저장소 클론**:
   ```bash
   git clone https://github.com/y5496694/wani-ai-light.git
   cd wani-ai-light
   ```

2. **의존성 설치**:
   ```bash
   pip install -r requirements.txt
   ```

3. **로컬 AI 모델 준비**:
   - Ollama를 설치하고 비전 모델을 다운로드합니다:
     ```bash
     ollama pull gemma4:e2b
     ```

4. **실행**:
   ```bash
   python main.py
   ```

## 설정
`config.py` 파일에서 카메라 인덱스, 호출어, 모델 설정 등을 변경할 수 있습니다.
