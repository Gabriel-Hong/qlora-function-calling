# GEN NX OpenAPI 스펙 분석 결과

> 소스: `openapi_gen_965.json` (OpenAPI 3.1.0)
> 분석일: 2026-02-27

---

## 1. 전체 요약

| 항목 | 수 |
|------|:--:|
| **고유 엔드포인트** | 273 |
| `/{id}` 변형 (개별 조회/수정/삭제) | 212 |
| `/INFO/DB/` 스키마 조회용 | 212 |
| `/REQUESTINFO/` 메타 정보 | 3 |
| **총 경로 수 (paths)** | 700 |

---

## 2. 카테고리별 엔드포인트 수

| 카테고리 | 엔드포인트 수 | 설명 |
|----------|:-----------:|------|
| **CONFIG** | 2 | 프로젝트 설정, 버전 정보 |
| **DOC** | 12 | 문서 관리 (NEW, OPEN, SAVE, ANAL 등) |
| **DB** | 212 | 데이터베이스 CRUD (전체의 ~78%) |
| **OPE** | 38 | 연산/오퍼레이션 |
| **VIEW** | 7 | 뷰 제어 |
| **POST** | 6 | 후처리/설계 결과 |
| `/INFO/DB/` | (212) | DB 스키마 정보 조회용 — 학습 대상 아님 |
| `/REQUESTINFO/` | (3) | POST 테이블 타입 정보 — 학습 대상 아님 |

---

## 3. DB 212개 세부 분류

### DB_Node/Element (6개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /DB/DOEL | DELETE, GET, POST, PUT |
| /DB/ELEM | DELETE, GET, POST, PUT |
| /DB/MADO | DELETE, GET, POST, PUT |
| /DB/NODE | DELETE, GET, POST, PUT |
| /DB/SBDO | DELETE, GET, POST, PUT |
| /DB/SKEW | DELETE, GET, POST, PUT |

### DB_Properties (28개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /DB/DMAS | DELETE, GET, POST, PUT |
| /DB/EDMP | DELETE, GET, POST, PUT |
| /DB/EPMT | DELETE, GET, POST, PUT |
| /DB/ESSF | DELETE, GET, POST, PUT |
| /DB/EWSF | DELETE, GET, POST, PUT |
| /DB/FIBR | DELETE, GET, POST, PUT |
| /DB/FIMP | DELETE, GET, POST, PUT |
| /DB/GRDP | DELETE, GET, POST, PUT |
| /DB/IEHC | DELETE, GET, POST, PUT |
| /DB/IEHG | DELETE, GET, POST, PUT |
| /DB/IEHP | DELETE, GET, POST, PUT |
| /DB/IMFM | DELETE, GET, POST, PUT |
| /DB/MATL | DELETE, GET, POST, PUT |
| /DB/MTCS | DELETE, GET, POST, PUT |
| /DB/POSL | DELETE, GET, POST, PUT |
| /DB/PSSF | DELETE, GET, POST, PUT |
| /DB/RPSC | DELETE, GET, POST, PUT |
| /DB/SECF | DELETE, GET, POST, PUT |
| /DB/SECT | DELETE, GET, POST, PUT |
| /DB/STOR | DELETE, GET, POST, PUT |
| /DB/TDME | DELETE, GET, POST, PUT |
| /DB/TDMF | DELETE, GET, POST, PUT |
| /DB/TDMT | DELETE, GET, POST, PUT |
| /DB/THIK | DELETE, GET, POST, PUT |
| /DB/TMAT | DELETE, GET, POST, PUT |
| /DB/TSGR | DELETE, GET, POST, PUT |
| /DB/VBEM | DELETE, GET, POST, PUT |
| /DB/VSEC | DELETE, GET, POST, PUT |

### DB_Boundary (27개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /DB/CGLP | DELETE, GET, POST, PUT |
| /DB/CLDR | GET, POST, PUT |
| /DB/CONS | DELETE, GET, POST, PUT |
| /DB/DRLS | DELETE, GET, POST, PUT |
| /DB/EARE | DELETE, GET, POST, PUT |
| /DB/ELNK | DELETE, GET, POST, PUT |
| /DB/FRLS | DELETE, GET, POST, PUT |
| /DB/GSPR | DELETE, GET, POST, PUT |
| /DB/GSTP | DELETE, GET, POST, PUT |
| /DB/IEPI | DELETE, GET, POST, PUT |
| /DB/MCON | DELETE, GET, POST, PUT |
| /DB/MLFC | DELETE, GET, POST, PUT |
| /DB/NLLP | DELETE, GET, POST, PUT |
| /DB/NLNK | DELETE, GET, POST, PUT |
| /DB/NSPR | DELETE, GET, POST, PUT |
| /DB/OFFS | DELETE, GET, POST, PUT |
| /DB/PRLS | DELETE, GET, POST, PUT |
| /DB/PZEF | GET, POST, PUT |
| /DB/RIGD | DELETE, GET, POST, PUT |
| /DB/RISS | DELETE, GET, POST, PUT |
| /DB/SDHY | DELETE, GET, POST, PUT |
| /DB/SDIS | DELETE, GET, POST, PUT |
| /DB/SDST | DELETE, GET, POST, PUT |
| /DB/SDVE | DELETE, GET, POST, PUT |
| /DB/SDVI | DELETE, GET, POST, PUT |
| /DB/SSPS | DELETE, GET, POST, PUT |
| /DB/STDG | DELETE, GET, POST, PUT |

### DB_Static Loads (22개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /DB/ARPR | DELETE, GET, POST, PUT |
| /DB/BMLD | DELETE, GET, POST, PUT |
| /DB/BODF | DELETE, GET, POST, PUT |
| /DB/CNLD | DELETE, GET, POST, PUT |
| /DB/EPSE | DELETE, GET, POST, PUT |
| /DB/EPST | DELETE, GET, POST, PUT |
| /DB/FBLA | DELETE, GET, POST, PUT |
| /DB/FBLD | DELETE, GET, POST, PUT |
| /DB/FMLD | DELETE, GET, POST, PUT |
| /DB/LTOM | DELETE, GET, POST, PUT |
| /DB/NBOF | DELETE, GET, POST, PUT |
| /DB/NMAS | DELETE, GET, POST, PUT |
| /DB/PNLA | DELETE, GET, POST, PUT |
| /DB/PNLD | DELETE, GET, POST, PUT |
| /DB/POSP | DELETE, GET, POST, PUT |
| /DB/PRES | DELETE, GET, POST, PUT |
| /DB/PSLT | DELETE, GET, POST, PUT |
| /DB/SDSP | DELETE, GET, POST, PUT |
| /DB/SEIS | DELETE, GET, POST, PUT |
| /DB/STLD | DELETE, GET, POST, PUT |
| /DB/WIND | DELETE, GET, POST, PUT |
| /DB/WNAT | DELETE, GET, POST, PUT |

### DB_Analysis (17개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /DB/ACTL | DELETE, GET, POST, PUT |
| /DB/BCCT | DELETE, GET, POST, PUT |
| /DB/BLDC | DELETE, GET, POST, PUT |
| /DB/BUCK | DELETE, GET, POST, PUT |
| /DB/DCTL | DELETE, GET, POST, PUT |
| /DB/EIGV | DELETE, GET, POST, PUT |
| /DB/HHCT | DELETE, GET, POST, PUT |
| /DB/MVCT | DELETE, GET, POST, PUT |
| /DB/MVCTTR | DELETE, GET, POST, PUT |
| /DB/MVCTbs | DELETE, GET, POST, PUT |
| /DB/MVCTch | DELETE, GET, POST, PUT |
| /DB/MVCTid | DELETE, GET, POST, PUT |
| /DB/NLCT | DELETE, GET, POST, PUT |
| /DB/PDEL | DELETE, GET, POST, PUT |
| /DB/SBCT | DELETE, GET, POST, PUT |
| /DB/SMCT | DELETE, GET, POST, PUT |
| /DB/STCT | DELETE, GET, POST, PUT |

### DB_Project (7개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /DB/BNGR | DELETE, GET, POST, PUT |
| /DB/GRUP | DELETE, GET, POST, PUT |
| /DB/LDGR | DELETE, GET, POST, PUT |
| /DB/PJCF | DELETE, GET, POST, PUT |
| /DB/STYP | DELETE, GET, POST, PUT |
| /DB/TDGR | DELETE, GET, POST, PUT |
| /DB/UNIT | GET, PUT |

### DB_Design (18개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /DB/CMFT | DELETE, GET, POST, PUT |
| /DB/DCON | DELETE, GET, POST, PUT |
| /DB/EBMW | DELETE, GET, POST, PUT |
| /DB/KFAC | DELETE, GET, POST, PUT |
| /DB/LENG | DELETE, GET, POST, PUT |
| /DB/LTSR | DELETE, GET, POST, PUT |
| /DB/MATD | GET, PUT |
| /DB/MBTP | DELETE, GET, POST, PUT |
| /DB/MDGN | DELETE, GET, POST, PUT |
| /DB/MEMB | DELETE, GET, POST, PUT |
| /DB/MRFT | DELETE, GET, POST, PUT |
| /DB/REBB | DELETE, GET, POST, PUT |
| /DB/REBC | DELETE, GET, POST, PUT |
| /DB/REBR | DELETE, GET, POST, PUT |
| /DB/REBW | DELETE, GET, POST, PUT |
| /DB/TRFT | DELETE, GET, POST, PUT |
| /DB/ULCT | DELETE, GET, POST, PUT |
| /DB/WMAK | DELETE, GET, POST, PUT |

### DB_Analysis Results (12개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /DB/CLWP | DELETE, GET, POST, PUT |
| /DB/CUTL | DELETE, GET, POST, PUT |
| /DB/LCOM-ALUM | DELETE, GET, POST, PUT |
| /DB/LCOM-CFSTEEL | DELETE, GET, POST, PUT |
| /DB/LCOM-CONC | DELETE, GET, POST, PUT |
| /DB/LCOM-FDN | DELETE, GET, POST, PUT |
| /DB/LCOM-GEN | DELETE, GET, POST, PUT |
| /DB/LCOM-LINEAR | DELETE, GET, POST, PUT |
| /DB/LCOM-SEISMIC | DELETE, GET, POST, PUT |
| /DB/LCOM-SRC | DELETE, GET, POST, PUT |
| /DB/LCOM-STEEL | DELETE, GET, POST, PUT |
| /DB/LCOM-STLCOMP | DELETE, GET, POST, PUT |

### DB_Dynamic Loads (9개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /DB/SPFC | DELETE, GET, POST, PUT |
| /DB/SPLC | DELETE, GET, POST, PUT |
| /DB/THFC | DELETE, GET, POST, PUT |
| /DB/THGA | DELETE, GET, POST, PUT |
| /DB/THGC | DELETE, GET, POST, PUT |
| /DB/THIS | DELETE, GET, POST, PUT |
| /DB/THMS | DELETE, GET, POST, PUT |
| /DB/THNL | DELETE, GET, POST, PUT |
| /DB/THSL | DELETE, GET, POST, PUT |

### DB_Heat of Hydration Loads (8개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /DB/CCFC | DELETE, GET, POST, PUT |
| /DB/ETFC | DELETE, GET, POST, PUT |
| /DB/HAHS | DELETE, GET, POST, PUT |
| /DB/HECB | DELETE, GET, POST, PUT |
| /DB/HPCE | DELETE, GET, POST, PUT |
| /DB/HSFC | DELETE, GET, POST, PUT |
| /DB/HSPT | DELETE, GET, POST, PUT |
| /DB/HSTG | DELETE, GET, POST, PUT |

### DB_Prestress Loads (7개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /DB/EXLD | DELETE, GET, POST, PUT |
| /DB/PRST | DELETE, GET, POST, PUT |
| /DB/PTNS | DELETE, GET, POST, PUT |
| /DB/TDCS | DELETE, GET, POST, PUT |
| /DB/TDNA | DELETE, GET, POST, PUT |
| /DB/TDNT | DELETE, GET, POST, PUT |
| /DB/TDPL | DELETE, GET, POST, PUT |

### DB_Miscellaneous Loads (7개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /DB/EFCT | DELETE, GET, POST, PUT |
| /DB/IELC | DELETE, GET, POST, PUT |
| /DB/IFGS | DELETE, GET, POST, PUT |
| /DB/INMF | DELETE, GET, POST, PUT |
| /DB/IPCR | DELETE, GET, POST, PUT |
| /DB/IPDT | DELETE, GET, POST, PUT |
| /DB/LDSQ | DELETE, GET, POST, PUT |

### DB_Construction Stage Loads (6+1개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /DB/CRPC | DELETE, GET, POST, PUT |
| /DB/CSCS | DELETE, GET, POST, PUT |
| /DB/ESQW | DELETE, GET, POST, PUT |
| /DB/STAG | DELETE, GET, POST, PUT |
| /DB/STBK | DELETE, GET, POST, PUT |
| /DB/TMLD | DELETE, GET, POST, PUT |
| /DB/SGLD | GET *(Test)* |

### DB_View (6개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /DB/CO_F | GET, PUT |
| /DB/CO_M | GET, PUT |
| /DB/CO_S | GET, PUT |
| /DB/CO_T | GET, PUT |
| /DB/NPLN | DELETE, GET, POST, PUT |
| /DB/SINF | GET, PUT |

### DB_Moving Loads (24개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /DB/IMPF | DELETE, GET, POST, PUT |
| /DB/LLAN | DELETE, GET, POST, PUT |
| /DB/LLANch | DELETE, GET, POST, PUT |
| /DB/LLANid | DELETE, GET, POST, PUT |
| /DB/LLANjp | DELETE, GET, POST, PUT |
| /DB/LLANop | DELETE, GET, POST, PUT |
| /DB/LLANtr | DELETE, GET, POST, PUT |
| /DB/MLSP | DELETE, GET, POST, PUT |
| /DB/MLSR | DELETE, GET, POST, PUT |
| /DB/MVCD | DELETE, GET, POST, PUT |
| /DB/MVHC | DELETE, GET, POST, PUT |
| /DB/MVHL | DELETE, GET, POST, PUT |
| /DB/MVHLTR | DELETE, GET, POST, PUT |
| /DB/MVLD | DELETE, GET, POST, PUT |
| /DB/MVLDBS | DELETE, GET, POST, PUT |
| /DB/MVLDCH | DELETE, GET, POST, PUT |
| /DB/MVLDEU | DELETE, GET, POST, PUT |
| /DB/MVLDID | DELETE, GET, POST, PUT |
| /DB/MVLDJP | DELETE, GET, POST, PUT |
| /DB/MVLDPL | DELETE, GET, POST, PUT |
| /DB/MVLDTR | DELETE, GET, POST, PUT |
| /DB/SLAN | DELETE, GET, POST, PUT |
| /DB/SLANch | DELETE, GET, POST, PUT |
| /DB/SLANop | DELETE, GET, POST, PUT |

### DB_Time History Analysis Results (4개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /DB/THRE | DELETE, GET, POST, PUT |
| /DB/THRG | DELETE, GET, POST, PUT |
| /DB/THRI | DELETE, GET, POST, PUT |
| /DB/THRS | DELETE, GET, POST, PUT |

### DB_Pushover (3개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /DB/PHGE | DELETE, GET, POST, PUT |
| /DB/POGD | DELETE, GET, POST, PUT |
| /DB/POLC | DELETE, GET, POST, PUT |

### DB_Settlement Loads (2개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /DB/SMLC | DELETE, GET, POST, PUT |
| /DB/SMPT | DELETE, GET, POST, PUT |

### DB_Heat of Hydration Results (1개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /DB/HHND | DELETE, GET, POST, PUT |

### DB_Grid Model Analysis Loads (1개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /DB/GALD | DELETE, GET, POST, PUT |

### DB_Misc. Result (1개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /DB/ULFC | DELETE, GET, POST, PUT |

---

## 4. DB 외 카테고리 상세

### CONFIG (2개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /CONFIG/PROJECT | GET |
| /CONFIG/VER | GET |

### DOC (12개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /DOC/ANAL | GET, POST |
| /DOC/CLOSE | POST |
| /DOC/EXIT | POST |
| /DOC/EXPORT | POST |
| /DOC/EXPORT/{NAME} | POST |
| /DOC/EXPORTMXT | POST |
| /DOC/IMPORT | POST |
| /DOC/IMPORTMXT | POST |
| /DOC/NEW | POST |
| /DOC/OPEN | POST |
| /DOC/SAVE | POST |
| /DOC/SAVEAS | POST |

### OPE (38개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /OPE/ANALSTATUS | POST |
| /OPE/APIEND | GET |
| /OPE/APISTART | GET |
| /OPE/AUTOMESH | POST |
| /OPE/BOM | POST |
| /OPE/DIVIDEELEM | POST |
| /OPE/EDMP | POST |
| /OPE/ELEMPAR | POST |
| /OPE/ELEMTDNT | POST |
| /OPE/LINEBMLD | POST |
| /OPE/MATL_DB/{TYPE or STANDARD} | GET |
| /OPE/MATL_STANDARD/{TYPE} | GET |
| /OPE/MEMB | POST |
| /OPE/MODALDAMPINGRATIO | POST |
| /OPE/MXTCMDSHELL | POST |
| /OPE/MXTCMDSHELL/{CMD} | GET |
| /OPE/MXTINFO | GET |
| /OPE/POSP | POST |
| /OPE/PROJECTSTATUS | GET |
| /OPE/SECTCORD | POST |
| /OPE/SECTPROP | GET, POST |
| /OPE/SECT_DBNAME/{shape} | GET |
| /OPE/SECT_NAME | POST |
| /OPE/SECT_SHAPE | GET |
| /OPE/SECT_SHAPE/{opt} | GET |
| /OPE/SMARTGRAPHVALUE | POST |
| /OPE/SSPS | POST |
| /OPE/STOR | POST |
| /OPE/STORYPROP | POST |
| /OPE/STORY_IRR_PARAM | GET, POST |
| /OPE/STORY_PARAM | GET, POST |
| /OPE/SWPCGIRDER_DLG | POST |
| /OPE/TEMP_TABLETYPE | GET |
| /OPE/TRANSACTION-CANCEL | POST |
| /OPE/TRANSACTION-CLOSE | POST |
| /OPE/TRANSACTION-OPEN | POST |
| /OPE/USLC | POST |
| /OPE/UTBLTYPES | GET |

### VIEW (7개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /VIEW/ACTIVE | POST |
| /VIEW/ANGLE | POST |
| /VIEW/CAPTURE | POST |
| /VIEW/DISPLAY | POST |
| /VIEW/PRECAPTURE | POST |
| /VIEW/RESULTGRAPHIC | POST |
| /VIEW/SELECT | GET |

### POST (6개)

| 엔드포인트 | HTTP 메서드 |
|-----------|------------|
| /POST/ANL | POST |
| /POST/CHART | POST |
| /POST/PM | POST |
| /POST/STEELCODECHECK | POST |
| /POST/TABLE | POST |
| /POST/TEXT | POST |

---

## 5. Tier 분류 (1차)

### 분류 기준

- **Tier 1**: 모델링 기본 워크플로우에 필수. 이것 없이는 모델을 만들거나 해석할 수 없음
- **Tier 2**: 실무에서 자주 사용하지만, 기본 워크플로우 외의 기능
- **Tier 3**: 특수 해석/국가별 특화 기능. 사용 빈도 낮음, 일반화에 의존

### DB Tier 분류

| 태그 | 수 | Tier | 근거 |
|------|:--:|:----:|------|
| DB_Node/Element | 6 | **1** | 모든 모델링의 기본. NODE, ELEM 없이 아무것도 못함 |
| DB_Properties | 28 | **1** | 재료(MATL), 단면(SECT) 등 필수 물성치 |
| DB_Boundary | 27 | **1** | 경계조건(CONS), 스프링, 강체 등 해석에 필수 |
| DB_Static Loads | 22 | **1** | 가장 기본적인 하중 (STLD, BMLD, CNLD 등) |
| DB_Project | 7 | **1** | 프로젝트 설정, 단위, 하중 그룹 등 기초 |
| DB_Analysis | 17 | **1** | 해석 제어 (ACTL, EIGV 등) |
| DB_Design | 18 | **2** | 설계 관련 (DCON, MEMB, REBB 등) |
| DB_Analysis Results | 12 | **2** | 하중 조합 (LCOM-*) |
| DB_Dynamic Loads | 9 | **2** | 동적 하중 (지진, 시간이력 등) |
| DB_Temperature Loads | 5 | **2** | 온도 하중 |
| DB_Prestress Loads | 7 | **2** | 프리스트레스 하중 |
| DB_Miscellaneous Loads | 7 | **2** | 기타 하중 |
| DB_Construction Stage Loads | 7 | **2** | 시공단계 하중 |
| DB_View | 6 | **2** | 색상, 평면 등 뷰 데이터 |
| DB_Moving Loads | 24 | **3** | 이동하중 (교량 특화, 국가별 변형 많음) |
| DB_Heat of Hydration Loads | 8 | **3** | 수화열 (매스콘크리트 특화) |
| DB_Time History Analysis Results | 4 | **3** | 시간이력 해석 결과 |
| DB_Pushover | 3 | **3** | 푸시오버 해석 |
| DB_Settlement Loads | 2 | **3** | 침하 하중 |
| DB_Heat of Hydration Results | 1 | **3** | 수화열 결과 |
| DB_Grid Model Analysis Loads | 1 | **3** | 격자 모델 해석 |
| DB_Misc. Result | 1 | **3** | 기타 결과 |

### DB 외 카테고리 Tier 분류

| 카테고리 | 엔드포인트 | Tier | 근거 |
|----------|-----------|:----:|------|
| DOC — OPEN, SAVE, ANAL, NEW, CLOSE | 5 | **1** | 파일 열기/저장/해석 실행은 기본 워크플로우 |
| DOC — IMPORT, EXPORT, EXIT 등 | 7 | **2** | 부가적 문서 관리 |
| OPE — AUTOMESH, SECTPROP, SSPS, STOR 등 | ~10 | **1** | 메시, 단면 속성, 층 등 핵심 연산 |
| OPE — APISTART/END, TRANSACTION, MXTCMD 등 | ~28 | **2-3** | 시스템/유틸리티 |
| VIEW — SELECT, CAPTURE, DISPLAY | 3 | **1** | 기본 뷰 조작 |
| VIEW — 나머지 | 4 | **2** | 부가적 뷰 기능 |
| POST — TABLE, PM | 2 | **1** | 결과 테이블/PM 조회 |
| POST — 나머지 | 4 | **2** | 특정 설계 결과 |
| CONFIG | 2 | **2** | 조회 전용, 자주 사용하지 않음 |

### Tier 요약

| Tier | 엔드포인트 수 | 예시 당 학습 데이터 | 소계 |
|:----:|:-----------:|:----------------:|:----:|
| **Tier 1** (핵심) | ~127 | 15-20 | ~1,900-2,540 |
| **Tier 2** (보조) | ~102 | 5-10 | ~510-1,020 |
| **Tier 3** (특수) | ~44 | 0-2 | ~0-88 |
| **합계** | **273** | — | **~2,400-3,650** |

> **참고**: Tier 1이 127개로 많기 때문에, 예시 당 데이터 수를 15-20개로 조정하여
> Plan.md의 목표 2,500개와 현실적으로 맞출 수 있음.
> 부정 예시(도구 호출 불필요 케이스)와 도메인 Q&A는 별도로 추가.
