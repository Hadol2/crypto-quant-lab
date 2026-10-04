# Risk_calc

암호화폐 선물 매매용 **리스크 기반 손절가 계산기** (Streamlit).
총 자산 대비 감당할 손실 비율을 정하면, 포지션 크기·레버리지·진입가로부터 손절 가격을 계산한다.

```
손절폭 = (총자산 × 리스크 비율) / (포지션 수량 × 레버리지)
LONG  손절가 = 진입가 − 손절폭
SHORT 손절가 = 진입가 + 손절폭
```

> **상태: 개발 중단 (2025).** 이후 리스크 관리·전략 연구는 별도의 비공개 트레이딩 리서치 프로젝트에서 이어가고 있다.

## 앱 구성

| 파일 | 설명 |
|---|---|
| `app.py` | 실거래용. Binance 선물 열린 포지션을 불러와 손절가 계산 (`BINANCE_API_KEY/SECRET` 필요) |
| `app_manual.py` | 수동 입력 버전 (API 키 불필요) |
| `paper_trade_app.py` | 가상매매(paper trading): 계약 저장·청산·손익 기록 |
| `public_dashboard.py` | `risk_log.csv` 기반 수익률 대시보드 |
| `paper_dashboard.py` | GitHub에 백업된 가상매매 기록 분석 (Streamlit secrets 필요) |
| `calculator.py` | 손절가 계산 로직 |
| `asset_manager.py`, `logger.py` | 자산 잔고(JSON)·매매 로그(CSV) 관리 |
| `binance_client.py`, `github_uploader.py` | Binance API / GitHub 백업 연동 |

## 실행

```sh
pip install -r requirements.txt
streamlit run app_manual.py      # 키 없이 바로 실행
```

실거래 버전은 `.env`에 `BINANCE_API_KEY`, `BINANCE_API_SECRET`을 넣고 `streamlit run app.py`.
`paper_dashboard.py`·GitHub 백업 기능은 `.streamlit/secrets.toml`에 `GITHUB_TOKEN`, `GITHUB_USERNAME`, `GITHUB_REPO`, `GITHUB_BRANCH`, `TARGET_FILE`(업로더는 `GITHUB_FILE_PATH`)이 필요하다 (현재 미운영).
