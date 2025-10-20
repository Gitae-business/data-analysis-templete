# 데이터 분석 및 머신러닝 템플릿

이 프로젝트는 데이터 분석과 머신러닝 작업을 위한 표준화된 템플릿입니다. 재사용 가능한 코드 구조를 제공하여 새로운 프로젝트를 빠르고 효율적으로 시작할 수 있도록 돕습니다.

## 프로젝트 구조

```
C:\Workspace\Python\data_analysis_templete\
├───.gitignore           # Git이 추적하지 않을 파일 및 폴더 목록
├───config.py            # 프로젝트의 주요 설정을 관리하는 파일
├───main.py              # 프로젝트의 메인 실행 파일
├───pyproject.toml       # 프로젝트 메타데이터 및 의존성 관리
├───README.md            # 프로젝트 설명 파일
├───checkpoints/         # 훈련된 모델의 가중치(체크포인트) 저장
├───data/                # 원본 데이터 (train.csv, test.csv)
├───logs/                # 실행 로그 저장
├───notebooks/           # 탐색적 데이터 분석(EDA)을 위한 Jupyter Notebook
├───output/              # 모델 예측 결과 등 최종 출력물 저장
└───src/                 # 핵심 소스 코드
    ├───analysis/        # 데이터 분석 (EDA 등)
    ├───inference/       # 훈련된 모델을 사용한 추론
    ├───loader/          # 데이터 로딩
    ├───models/          # 모델 아키텍처, 손실 함수, 옵티마이저 등
    ├───postprocessing/  # 추론 결과 후처리
    ├───preprocessing/   # 데이터 전처리
    └───training/        # 모델 훈련 로직
```

## 시작하기

### 1. 의존성 설치

이 프로젝트는 `uv`와 같은 최신 Python 패키지 관리 도구를 사용하는 것을 권장합니다. `pyproject.toml` 파일에 필요한 라이브러리를 명시하고 다음 명령어로 설치할 수 있습니다.

```bash
# pyproject.toml 파일의 [project].dependencies에 pandas, scikit-learn, torch 등을 추가하세요.
# 예: dependencies = ["pandas", "scikit-learn", "torch"]

uv pip install -r requirements.txt 
# 또는 pyproject.toml을 직접 사용하는 경우
uv pip install .
```

### 2. 데이터 준비

`data/` 디렉터리에 `train.csv`와 `test.csv` 파일을 위치시킵니다.

## 사용 방법

### 1. 설정 관리

`config.py` 파일에서 프로젝트의 주요 변수들을 설정할 수 있습니다.

- **경로 설정**: 데이터, 모델, 체크포인트 디렉터리 경로
- **모델 설정**: 사용할 모델의 이름 (`MODEL_NAME`)
- **하이퍼파라미터**: `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE` 등
- **시드 고정**: 재현성을 위한 `SEED` 값

### 2. 프로젝트 실행

프로젝트의 모든 과정(데이터 로딩, 훈련, 추론)은 `main.py`를 통해 실행됩니다.

```bash
python main.py
```

`main.py`의 실행 흐름은 다음과 같습니다.
1.  `load_data()`: `data/` 폴더에서 학습 및 테스트 데이터를 로드합니다.
2.  `train_test_split`: 검증을 위해 학습 데이터를 훈련/검증 세트로 분리합니다.
3.  `Trainer`: `MLP`와 `Linear` 모델을 각각 5-fold 교차 검증으로 훈련합니다.
4.  `HillClimbEnsembler`: 훈련된 모든 모델의 예측 결과를 앙상블하여 최적의 가중치를 찾습니다.
5.  `Predictor`: 최종적으로 테스트 데이터에 대한 예측을 수행하고 결과를 출력합니다.

## 커스터마이징

이 템플릿을 자신의 프로젝트에 맞게 수정하려면 다음 부분을 확인하세요.

- **`config.py`**: 자신의 환경에 맞게 경로와 하이퍼파라미터를 수정합니다.
- **`src/loader/loader.py`**: 다른 형식의 데이터를 로드해야 할 경우 이 파일을 수정합니다.
- **`src/preprocessing/preprocessor.py`**: 데이터에 맞는 전처리 로직을 추가합니다.
- **`src/models/model_factory.py`**: 새로운 모델을 추가하고 싶을 때 이 파일을 수정하여 모델을 등록합니다.
- **`main.py`**: 전체적인 실행 흐름을 원하는 대로 변경합니다.
