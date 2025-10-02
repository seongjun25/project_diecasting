# project_diecasting


이 Shiny 앱의 README에 포함할 내용을 정리해드리겠습니다.
markdown# 다이캐스팅 공정 품질관리 대시보드

다이캐스팅 공정 데이터를 기반으로 불량을 예측하고 품질을 모니터링하는 Shiny 웹 애플리케이션입니다.

## 주요 기능

### 1. 성과 모니터링
- **불량률 현황**: 전체 기간 및 일별 불량률 추적
- **일간 변화 분석**: 전일 대비 불량률 증감 모니터링
- **시간대별 분석**: 불량 집중 시간대 파악
- **공정 능력 분석**: Cp/Cpk를 활용한 공정 능력 평가
- **시계열 추이**: 선택 기간 및 변수에 대한 Plotly 인터랙티브 그래프

### 2. 불량 원인 예측
- **금형별 분석**: 각 금형코드별 불량률 비교
- **SHAP 분석**: 금형별 변수 영향도 TOP10 시각화
- **PDP(Partial Dependence Plot)**: 특정 변수의 불량 예측 영향 분석

### 3. 예측 & 개선
- **실시간 불량 예측**: 공정 변수 입력 시 불량 확률 예측
- **SHAP 해석**: 개별 예측에 대한 상위 5개 영향 변수 분석
- **개선 권장사항**: PDP 기반 최적 변수 범위 제시
- **피드백 시스템**: 예측 결과와 실제 불량 여부 기록 및 누적

### 4. 데이터 분석 (EDA)
- **전처리 전후 비교**: 각 변수의 분포 및 통계량 비교
- **변수별 처리 요약**: 이상치 제거, 결측치 보간 등 전처리 내역
- **모델 성능 평가**: 혼동행렬, 분류 리포트, ROC-AUC 표시

## 기술 스택

- **Python 3.x**
- **Shiny for Python**: 웹 애플리케이션 프레임워크
- **주요 라이브러리**:
  - pandas: 데이터 처리
  - scikit-learn: 머신러닝 모델 및 평가
  - matplotlib, seaborn, plotly: 시각화
  - shap: 모델 해석
  - joblib: 모델 저장/로드

## 필요 파일

애플리케이션 실행을 위해 다음 파일들이 필요합니다:
project/
├── app.py                 # 메인 애플리케이션 파일
├── train.csv             # 원본 학습 데이터
├── train_drop.csv        # 시계열 분석용 데이터
├── train_df.csv          # 전처리된 학습 데이터
└── final_model.pkl       # 학습된 LightGBM 모델

## 설치 및 실행

### 1. 필요한 패키지 설치
```bash
pip install shiny pandas scikit-learn matplotlib seaborn plotly shap joblib
2. 애플리케이션 실행
bashshiny run app.py
3. 브라우저 접속
기본적으로 http://127.0.0.1:8000에서 실행됩니다.
데이터 전처리
주요 전처리 내용

이상치 제거: 센서 오류값(65535, 1449 등) 제거
결측치 보간: KNN Imputer를 활용한 결측치 처리
스케일링: RobustScaler 적용
인코딩: 범주형 변수 Ordinal Encoding
파생 변수: registration_time에서 hour 추출

주요 변수별 처리

low_section_speed: 65535 이상치 제거, KNN 보간
molten_temp: 80도 이하 센서 오류 결측 처리
physical_strength: 5 이하 값 결측 처리
Coolant_temperature: 1449 이상치 행 제거
tryshot_signal: 결측치를 'A'로 대체
heating_furnace: molten_volume 기준 'C'로 보간

모델 정보

알고리즘: LightGBM (Light Gradient Boosting Machine)
선정 이유: 불량 케이스 탐지(Recall) 성능 우수
성능 지표:

ROC-AUC: 0.9889
Threshold: 0.8346
Recall (불량 클래스): 0.96
F1-Score: 0.96



주요 분석 변수
온도 관련

molten_temp (용탕온도)
upper_mold_temp1/2 (상부금형온도)
lower_mold_temp1/2 (하부금형온도)
sleeve_temperature (슬리브온도)
Coolant_temperature (냉각수온도)

압력 및 속도

cast_pressure (주조압력)
low_section_speed (저속구간속도)
high_section_speed (고속구간속도)
physical_strength (물리적강도)

계량 및 시간

molten_volume (용탕량)
biscuit_thickness (비스킷두께)
facility_operation_cycleTime (설비작동사이클시간)
production_cycletime (생산사이클시간)
hour (시간대)

설비 상태

mold_code (금형코드)
working (가동여부)
heating_furnace (가열로)
tryshot_signal (트라이샷신호)
EMS_operation_time (EMS작동시간)

사용 가이드
성과 모니터링 탭

날짜 선택기로 분석 일자 선택
전체 기간 불량률과 선택일 불량률 비교
불량 집중 시간대 확인
변수 선택 후 Cp/Cpk 공정 능력 분석
기간 범위와 변수를 선택하여 시계열 추이 확인

불량 원인 예측 탭

금형 코드 선택 (TOP 5 중 선택)
금형별 불량률과 SHAP 변수 영향도 확인
PDP 변수 선택 후 영향도 분석

예측 & 개선 탭

각 공정 변수 값 입력 (슬라이더 또는 숫자 입력)
"예측하기" 버튼 클릭
예측 결과 확인 (양품/불량 확률)
SHAP Bar Plot으로 영향 변수 파악
PDP로 변수 조정 권장사항 확인
실제 결과 피드백 저장 (선택사항)

데이터 분석 탭

분석할 변수 선택
전처리 전후 분포 비교
기술통계량 및 처리 요약 확인
모델 성능 지표 확인
