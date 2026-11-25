# 연구 재현성 검증 기록

## 1. 프로젝트 개요

- **프로젝트명:** T2D 인간 췌도 전사체 데이터의 2차 분석
- **데이터 출처:** NCBI Gene Expression Omnibus (GEO)
- **GEO accession:** GSE159984
- **분석 목적:** 제2형 당뇨병(T2D)과 대조군의 차등 발현 유전자를 확인하고, 포도당 및 팔미테이트 대사 스트레스 조건의 발현 변화와 비교하기 위한 분석 환경 구축

데이터 출처: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE159984

## 2. 개발 환경 구성

- 운영체제: Windows 11
- Python: 3.14.7
- 가상환경: `.venv`
- 패키지 설치: `requirements.txt` 기반 설치
- 주요 패키지:
  - pandas 2.3.3
  - numpy 2.5.5
  - scipy 1.18.1
  - requests 2.34.2
  - pytest 9.2.0
  - matplotlib 3.11.2
  - seaborn 0.13.2
  - gseapy 1.3.2

## 3. 다운로드 함수 테스트

실제 GEO 서버에 연결하기 전, 다운로드 함수의 기본 동작을 자동 테스트하였다.

테스트 파일: `tests/test_download.py`

검증 항목:

1. 정상적인 gzip 데이터가 다운로드되어 저장되는지 확인
2. 다운로드 파일의 SHA-256 체크섬이 계산되는지 확인
3. 다운로드 응답이 정상적으로 닫히는지 확인
4. 잘못된 gzip 파일이 최종 파일로 저장되지 않는지 확인
5. 임시 파일이 오류 발생 후 정리되는지 확인

실행 명령:

```bash
python -m pytest tests/ -v
```

실행 결과:

```text
tests/test_download.py::test_download_valid_gzip PASSED
tests/test_download.py::test_invalid_gzip_is_not_saved PASSED
```

**결과: 작성한 다운로드 관련 테스트 2개 통과.**

## 4. 실제 GEO 데이터 다운로드

실행 명령:

```bash
python -m src.download_data
```

다운로드 결과:

| 분석 조건 | 파일명 | 크기 |
|---|---|---:|
| T2D | GSE159984_de-t2d.tsv.gz | 1,939,179 bytes |
| Combined stress | GSE159984_de-d4pg.tsv.gz | 2,054,479 bytes |
| Palmitate | GSE159984_de-d4p.tsv.gz | 1,995,647 bytes |
| Glucose | GSE159984_de-d4g.tsv.gz | 2,327,179 bytes |

네 파일 모두 다운로드 성공 메시지를 확인하였다.

또한 `data/raw/download_manifest.json` 파일이 생성되었다. 이 파일에는 다운로드 URL, 파일 크기, SHA-256 체크섬 및 상태 정보가 기록된다.

## 5. 실제 데이터 구조 확인

pandas를 이용하여 gzip 압축 TSV 파일을 읽고 컬럼 이름과 첫 3개 행을 확인하였다.

확인 명령:

```bash
python -c "import pandas as pd; from pathlib import Path; [print('\nFILE:', p.name, '\nCOLUMNS:', pd.read_csv(p, sep='\t', compression='gzip', nrows=0).columns.tolist(), '\nPREVIEW:\n', pd.read_csv(p, sep='\t', compression='gzip', nrows=3).to_string(index=False)) for p in sorted(Path('data/raw').glob('*.tsv.gz'))]"
```

네 파일에서 동일한 23개 컬럼을 확인하였다.

주요 컬럼:

- `baseMean`: 평균 발현량 관련 값
- `log2FoldChange`: 로그 변환된 발현 변화량
- `lfcSE`: 로그 폴드 변화량의 표준오차
- `stat`: 통계 검정 통계량
- `pvalue`: 원시 p-value
- `padj`: 다중검정 보정 p-value
- `gene_id`: 유전자 식별자
- `ensembl_id`: Ensembl 유전자 ID
- `short_id`: 유전자 심볼
- `description`: 유전자 설명
- `biotype`: 유전자 유형

실제 데이터 미리보기에서 `log2FoldChange`, `padj`, `ensembl_id`, `short_id` 등의 컬럼이 존재함을 확인하였다.

## 6. 현재까지의 검증 결과

| 검증 항목 | 결과 |
|---|---|
| Python 가상환경 구성 | 완료 |
| 필수 패키지 설치 | 완료 |
| 정상 gzip 다운로드 테스트 | 통과 |
| 손상 gzip 파일 처리 테스트 | 통과 |
| GEO 데이터 4개 다운로드 | 성공 |
| 다운로드 manifest 생성 | 성공 |
| 데이터 컬럼 확인 | 완료 |
| 데이터 미리보기 확인 | 완료 |

## 7. 다음 단계

1. 네 데이터 파일의 행 수, 결측치 및 중복 유전자 확인
2. `padj < 0.05` 및 `abs(log2FoldChange) > 0.58` 기준으로 DEG 분류
3. 상향 및 하향 조절 유전자 수 계산
4. T2D와 복합 대사 스트레스 조건의 DEG 중복 비교
5. 포도당 단독 및 팔미테이트 단독 조건 비교
6. 유전자 집합 기반 경로 풍부도 분석
7. 결과 표와 시각화 자료 생성

## 8. 분석 시 주의사항

- 본 프로젝트는 공개된 사전 계산 DEG 테이블을 이용한 2차 분석이다. 원시 FASTQ 파일부터 직접 재분석하는 과정은 포함하지 않는다.
- 벌크 췌도 데이터만으로 특정 유전자 변화가 베타세포에서만 발생했다고 단정할 수 없다.
- DEG 중복은 두 조건 사이의 유전자 발현 변화가 겹친다는 의미이며, 인과관계를 입증하지 않는다.
- 경로 풍부도 분석 결과는 실제 경로 활성도를 직접 측정한 결과가 아니다.
- 포도당과 팔미테이트의 상승작용을 주장하려면 적절한 상호작용 분석이 별도로 필요하다.

## 9. 데이터 품질 검사(QC)

**검사 대상:** GSE159984의 DEG 결과 파일 4개

| 데이터셋 | 행 수 | 열 수 | 중복 유전자 ID | Ensembl ID 불일치 |
|---|---:|---:|---:|---:|
| T2D | 17,499 | 24 | 0 | 0 |
| 복합 스트레스 | 18,235 | 24 | 0 | 0 |
| 팔미테이트 | 17,918 | 24 | 0 | 0 |
| 포도당 | 21,485 | 24 | 0 | 0 |

**검사 결과**
- 4개 파일 모두 정상적으로 불러와 검사를 완료하였다.
- 헤더와 실제 데이터의 필드 수 차이를 확인하고 24개 열 구조로 정리하였다.
- 중복 유전자 ID는 발견되지 않았다.
- 주요 수치형 열(`baseMean`, `log2FoldChange`, `pvalue`, `padj`)에서 결측값은 발견되지 않았다.
- 비교 가능한 레코드에서 원본 Ensembl ID와 주석 Ensembl ID의 불일치는 발견되지 않았다.
- 일부 행에는 유전자 주석 정보가 없었으며, 해당 행은 임의로 삭제하지 않고 유지하였다.

**검사 결과 파일**
- `results/qc/data_quality_report.json`

**진행 상태:** 초기 데이터 품질 검사 완료
---

**기록 기준:** 2026년 11월 25일  
**현재 진행 상태:** 데이터 다운로드 및 기본 구조 검증 완료  
**다음 진행 단계:** 실제 데이터 품질 검사 및 DEG 분석 준비
