# 🤖 DDPG (Deep Deterministic Policy Gradient) 완전 정복

> **참고 논문**
> - Lillicrap et al. (2016), *"Continuous Control with Deep Reinforcement Learning"*, ICLR 2016 (원조 DDPG 논문)
> - Tavakkoli et al., *"Model Free Deep Deterministic Policy Gradient Controller for Setpoint Tracking of Non-minimum Phase Systems"*

---

## 📚 목차

1. [강화학습 기초 개념](#1-강화학습-기초-개념)
2. [DDPG 알고리즘 원리](#2-ddpg-알고리즘-원리)
3. [핵심 구성요소 상세 설명](#3-핵심-구성요소-상세-설명)
4. [DDPG 알고리즘 슈도코드](#4-ddpg-알고리즘-슈도코드)
5. [제어공학 응용: NMP 시스템](#5-제어공학-응용-nmp-시스템)
6. [성능 평가 프레임워크 (12가지 기준)](#6-성능-평가-프레임워크-12가지-기준)
7. [실험 결과 분석](#7-실험-결과-분석)
8. [Colab 실습 가이드](#8-colab-실습-가이드)
9. [한계점 및 향후 연구방향](#9-한계점-및-향후-연구방향)

---

## 1. 강화학습 기초 개념

### 1.1 강화학습 패러다임

강화학습(Reinforcement Learning, RL)은 **에이전트(Agent)** 가 **환경(Environment)** 과 상호작용하며 누적 보상을 최대화하는 정책을 학습하는 방법론입니다.

```
        상태(State) s_t
환경 ─────────────────────► 에이전트
 ▲                               │
 │   보상(Reward) r_t             │ 행동(Action) a_t
 └───────────────────────────────┘
```

| 요소 | 기호 | 설명 |
|------|------|------|
| 상태 | $s \in S$ | 환경의 현재 상태 (측정값) |
| 행동 | $a \in A$ | 에이전트의 제어 신호 |
| 보상 | $r$ | 행동에 대한 즉각적 피드백 |
| 정책 | $\pi: S \rightarrow A$ | 상태 → 행동 매핑 함수 |
| 할인율 | $\gamma \in [0,1]$ | 미래 보상의 현재 가치 |

### 1.2 마르코프 결정 과정 (MDP)

환경은 다음 요소로 구성된 **MDP**로 모델링됩니다:
- 상태 공간 $S$
- 행동 공간 $A = \mathbb{R}^N$ (연속 공간)
- 초기 상태 분포 $p(s_1)$
- 전이 동학 $p(s_{t+1}|s_t, a_t)$
- 보상 함수 $r(s_t, a_t)$

**누적 할인 보상 (Return):**
$$R_t = \sum_{i=t}^{T} \gamma^{(i-t)} r(s_i, a_i)$$

**목표:** 기대 누적 보상 최대화
$$J = \mathbb{E}_{r_i, s_i \sim E, a_i \sim \pi}[R_1]$$

### 1.3 행동-가치 함수 (Q-function)

상태 $s_t$에서 행동 $a_t$를 취하고 이후 정책 $\pi$를 따를 때의 기대 누적 보상:

$$Q^\pi(s_t, a_t) = \mathbb{E}_{r_{\geq t}, s_{i>t} \sim E, a_{i>t} \sim \pi}[R_t | s_t, a_t]$$

**벨만 방정식 (Bellman Equation):**
$$Q^\pi(s_t, a_t) = \mathbb{E}_{r_t, s_{t+1} \sim E} \left[ r(s_t, a_t) + \gamma \mathbb{E}_{a_{t+1} \sim \pi} [Q^\pi(s_{t+1}, a_{t+1})] \right]$$

결정론적 정책 $\mu: S \rightarrow A$의 경우:
$$Q^\mu(s_t, a_t) = \mathbb{E}_{r_t, s_{t+1} \sim E} \left[ r(s_t, a_t) + \gamma Q^\mu(s_{t+1}, \mu(s_{t+1})) \right]$$

---

## 2. DDPG 알고리즘 원리

### 2.1 DQN의 한계와 DDPG의 등장

**DQN의 문제점:**
- 이산(discrete) 및 저차원 행동 공간만 처리 가능
- 연속 행동 공간에서 `argmax Q(s,a)` 계산이 매 스텝마다 필요 → 비현실적
- **차원의 저주:** 7 자유도 시스템을 가장 거친 이산화로도 $3^7 = 2187$개 행동

**DDPG의 해결책:**
- **결정론적 정책 경사(DPG)** 알고리즘 기반
- **Actor-Critic** 구조로 연속 행동 공간 직접 처리
- DQN의 핵심 기법(경험 재생, 타겟 네트워크) 계승

### 2.2 DDPG = DPG + DQN 아이디어

```
        ┌─────────────────────────────────────────┐
        │              DDPG                       │
        │                                         │
        │  DPG 알고리즘        DQN 안정화 기법     │
        │  (결정론적 정책경사) + (경험 재생 버퍼)  │
        │                      (타겟 네트워크)     │
        │                      (배치 정규화)       │
        └─────────────────────────────────────────┘
```

### 2.3 Actor-Critic 구조

```
                    관측(Observation) s
                           │
           ┌───────────────┼───────────────┐
           ▼                               ▼
    ┌─────────────┐                ┌─────────────┐
    │   Actor     │                │   Critic    │
    │  $\mu(s|\theta^\mu)$  │                │  $Q(s,a|\theta^Q)$ │
    │  (정책망)   │                │  (가치망)   │
    └─────────────┘                └─────────────┘
           │                               ▲
           │ 행동 $a = \mu(s)$                  │
           └───────────────────────────────┘
                      $Q(s, \mu(s))$ 계산
```

**Actor (정책망):** 상태를 입력받아 최적 행동을 결정론적으로 출력
$$a = \mu(s|\theta^\mu)$$

**Critic (가치망):** 상태-행동 쌍의 Q값(기대 누적 보상) 추정
$$Q = Q(s, a|\theta^Q)$$

**Actor 업데이트 (정책 경사):**
$$\nabla_{\theta^\mu} J \approx \mathbb{E}_{s_t \sim \rho^\beta} \left[ \nabla_a Q(s, a | \theta^Q) |_{s=s_t, a=\mu(s_t)} \nabla_{\theta^\mu} \mu(s | \theta^\mu) |_{s=s_t} \right]$$

**Critic 업데이트 (TD 오차 최소화):**
$$L(\theta^Q) = \mathbb{E} \left[ (y_i - Q(s_i, a_i | \theta^Q))^2 \right]$$
$$y_i = r_i + \gamma Q'(s_{i+1}, \mu'(s_{i+1} | \theta^{\mu'}) | \theta^{Q'})$$

---

## 3. 핵심 구성요소 상세 설명

### 3.1 경험 재생 버퍼 (Experience Replay Buffer)

**필요성:**
- 순차적 환경 탐색으로 생성된 샘플들은 **시간적 상관관계**를 가짐
- 대부분의 최적화 알고리즘은 i.i.d(독립동일분포) 샘플을 가정
- 하드웨어 효율을 위해 미니배치 학습 필요

**동작 방식:**
```python
# 전이(transition) 저장
replay_buffer.store(($s_t, a_t, r_t, s_{t+1}$))

# 무작위 미니배치 샘플링
batch = replay_buffer.sample(batch_size=64)

# 비상관(uncorrelated) 샘플로 학습
```

**장점:**
- 샘플 효율 향상 (여러 번 재사용)
- 시간적 상관관계 제거로 학습 안정화
- Off-policy 알고리즘이므로 큰 버퍼 사용 가능

### 3.2 타겟 네트워크 (Target Networks)

**문제:** Q값 계산 시 업데이트 대상인 네트워크를 그대로 사용 → Q값이 발산할 위험

**해결책:** Actor와 Critic의 **복사본(타겟 네트워크)** 생성

```python
# 소프트 업데이트 (Soft Update)
$\theta'  \leftarrow  \tau \cdot \theta + (1-\tau) \cdot \theta'$    # $\tau \ll 1$ (예: 0.001)
```

- **하드 업데이트(DQN):** 주기적으로 가중치를 직접 복사
- **소프트 업데이트(DDPG):** 천천히 추적 → 더 안정적인 타겟값

**효과:** 학습 중 타겟값이 천천히 변화 → 지도학습에 가까운 안정적 학습

### 3.3 배치 정규화 (Batch Normalization)

**문제:** 저차원 특성 벡터의 각 요소가 다른 물리적 단위를 가짐
- 예: 위치(m) vs 속도(m/s) vs 각도(rad)

**해결책:** 각 차원을 미니배치 내에서 단위 평균/분산으로 정규화

```python
# 배치 정규화 적용 위치
# - 상태 입력
# - $\mu$ 네트워크의 모든 레이어
# - 행동 입력 이전의 $Q$ 네트워크 레이어들
```

### 3.4 탐험 정책 (Exploration Policy)

**Off-policy 알고리즘의 장점:** 학습 알고리즘과 탐험 정책을 분리 가능

**노이즈 추가 방식:**
$$\mu'(s_t) = \mu(s_t|\theta^\mu_t) + \mathcal{N}$$

**Ornstein-Uhlenbeck (OU) 프로세스:**
$$d\mathbf{x}_t = \theta(\mu - \mathbf{x}_t)dt + \sigma d\mathbf{W}_t$$

- 물리적 제어 환경에서 **관성(inertia)** 을 고려한 시간적 상관 탐험 노이즈
- 파라미터: $\theta = 0.15, \sigma = 0.2$

---

## 4. DDPG 알고리즘 슈도코드

```
알고리즘: DDPG

[초기화]
1. Critic 네트워크 $Q(s,a|\theta^Q)$와 Actor $\mu(s|\theta^μ)$를 무작위 초기화
2. 타겟 네트워크 $Q'$, $\mu'$ 초기화: $\theta^{Q'} \leftarrow \theta^Q, \theta^{\mu'} \leftarrow \theta^\mu$
3. 경험 재생 버퍼 $R$ 초기화

[에피소드 루프] for episode = 1 to $M$:
  4. 탐험 노이즈 프로세스 $\mathcal{N}$ 초기화
  5. 초기 관측 상태 $s_1$ 수신

  [시간 스텝 루프] for $t = 1$ to $T$:
    6. 현재 정책과 탐험 노이즈로 행동 선택:
       $a_t = \mu(s_t|\theta^\mu) + \mathcal{N}_t$

    7. 행동 $a_t$ 실행 → 보상 $r_t$ 획득, 새 상태 $s_{t+1}$ 관측

    8. 전이 $(s_t, a_t, r_t, s_{t+1})$를 $R$에 저장

    9. $R$에서 무작위 미니배치 $N$개 샘플링: $(s_i, a_i, r_i, s_{i+1})$

   10. TD 타겟 계산:
       $y_i = r_i + \gamma \cdot Q'(s_{i+1}, \mu'(s_{i+1}|\theta^{\mu'})|\theta^{Q'})$

   11. Critic 업데이트 (손실 최소화):
       $L = \frac{1}{N} \sum_i (y_i - Q(s_i, a_i|\theta^Q))^2$

   12. Actor 업데이트 (샘플 정책 경사):
       $\nabla_{\theta^\mu} J \approx \frac{1}{N} \sum_i \nabla_a Q(s,a|\theta^Q)|_{s=s_i, a=\mu(s_i)} \cdot \nabla_{\theta^\mu} \mu(s|\theta^\mu)|_{s_i}$

   13. 타겟 네트워크 소프트 업데이트:
       $\theta^{Q'} \leftarrow \tau \cdot \theta^Q + (1-\tau) \cdot \theta^{Q'}$
       $\theta^{\mu'} \leftarrow \tau \cdot \theta^\mu + (1-\tau) \cdot \theta^{\mu'}$
```

---

## 5. 제어공학 응용: NMP 시스템

### 5.1 비최소위상 (Non-Minimum Phase) 시스템

**전달함수:**
$$G_P = \frac{0.5s - 1}{s^2 + 3s + 2}$$

**특징:**
- 안정한 시스템이지만 우반면 영점(right-half plane zero) 보유
- 스텝 응답에서 **언더슈트(undershoot)** 발생 불가피
- 달성 가능한 제어 성능에 근본적 한계 존재

**상태공간 표현 (제어가능 표준형):**
$$A_p = \begin{bmatrix} 0 & 1 \\\\ -2 & -3 \end{bmatrix}, \quad B_p = \begin{bmatrix} 0 \\\\ 1 \end{bmatrix}, \quad C_p = \begin{bmatrix} -1 & 0.5 \end{bmatrix}$$

### 5.2 LQI (Linear Quadratic Integral) 제어기

**비교 기준 고전 제어기**로 선택된 이유:
1. PID 유사 성능을 가진 널리 사용되는 고전 제어기
2. 추적 오차와 제어 신호를 포함하는 비용 함수 → DDPG의 보상 함수와 직접 비교 가능

**적분 오차 상태 도입:**
$$\begin{bmatrix} \dot{x}_p \\\\ \dot{e}_I \end{bmatrix} = \underbrace{\begin{bmatrix} A_p & 0 \\\\ -C_p & 0 \end{bmatrix}}_{A} \begin{bmatrix} x_p \\\\ e_I \end{bmatrix} + \underbrace{\begin{bmatrix} B_p \\\\ 0 \end{bmatrix}}_{B} u + \begin{bmatrix} 0 \\\\ 1 \end{bmatrix} r$$

**이차 비용 함수:**
$$J = \frac{1}{2}\int_0^\infty (x^T Q_{LQR} x + u_c^T R_{LQR} u_c) dt$$

**최적 제어 신호:**
$$u_c(t) = -K_{LQR} x_p(t) + K_I e_I(t)$$

### 5.3 DDPG 제어기 변형

**DDPG₁:** Critic 네트워크가 상태 피드백 이득 $K_{DDPG}$ 추정
$$a = u_c = K_1 x_1 + K_2 x_2 + K_3 x_3 = K_{DDPG} \mathbf{x}$$

**DDPG₂:** 완전한 블랙박스 데이터 기반 제어기 (End-to-End)
- 어떤 모델 구조도 가정하지 않음
- Critic: 2개의 완전연결층 + ReLU (은닉층 크기 100)
- Actor: 4개의 완전연결층 + tanh + 스케일링 레이어

**공통 관측값:**
$$\mathbf{s} = \mathbf{x} = [x_1, x_2, x_3]^T = [x_p, e_I]^T$$

**보상 함수 (LQR 비용 함수의 음수):**
$$r(t) = -(\mathbf{x}(t)^T Q_{LQR} \mathbf{x}(t) + u_c^T R_{LQR} u_c)$$

---

## 6. 성능 평가 프레임워크 (12가지 기준)

### 6.1 과도 및 정상상태 성능 (C1-C5)

| 기준 | 기호 | 설명 | 목표 |
|------|------|------|------|
| C1. 상승 시간 | $t_r$ | 응답이 최종값의 0→100% 도달 시간 | 작을수록 좋음 |
| C2. 최대 오버슈트 | $M_p$ | 원하는 응답 대비 최대 초과값 (%) | 작을수록 좋음 |
| C3. 최대 언더슈트 | $M_u$ | 신호의 최대 하강값 (%) | NMP에서 불가피, 최소화 목표 |
| C4. 정착 시간 | $t_s$ | 최종값 ±$e_{ss}$ 범위 내 유지 시작 시간 | 작을수록 좋음 |
| C5. 정상상태 오차 | $e_{ss} = r - y(t_f)$ | 원하는 출력과 실제 출력의 차이 | 0에 가까울수록 좋음 |

### 6.2 누적 오차 지수 (C6-C7)

**ISE (Integral of Square Error):**
$$ISE = \int_0^{t_f} e^2(t) dt$$
- 큰 오차를 강조, 단기 큰 오차에 민감

**ITAE (Integral of Time × Absolute Error):**
$$ITAE = \int_0^{t_f} t|e(t)| dt$$
- 시간 가중치 → 정상상태 오차를 더 강하게 패널티

### 6.3 제어 신호 품질 지수 (C8-C10)

**IACE (Integral of Absolute Control Effort):**
$$IACE = \int_0^{t_f} |u_c(t)| dt$$
- 소비 에너지를 나타냄

**IACER (Integral of Absolute Control Effort Rate):**
$$IACER = \int_0^{t_f} |du_c(t)| dt$$
- 제어 신호의 변화율 → 엑추에이터 포화(rate saturation) 방지에 중요

**최대 제어 신호 ($u_{c_{max}}$):**
$$u_{c_{max}} = \max_t |u_c(t)|$$
- 진폭 포화 방지를 위해 시스템 입력 한계 이하 유지 필요

### 6.4 주파수 영역 안정성 지수 (C11-C12)

**이득 여유 (Gain Margin, GM):**
$$GM = 20\log\left(\frac{1}{|G(j\omega_{pc})|}\right) \text{ [dB]}$$
- $\omega_{pc}$: 위상 교차 주파수 (위상 = -180°인 주파수)
- GM이 클수록 안정성 여유 큼

**지연 여유 (Delay Margin, DM):**
$$DM = \frac{PM}{|G(j\omega_{gc})|}$$
- 시스템이 불안정해지기 전 허용 가능한 최대 시간 지연
- 실용적인 안정성 지수

> ⚠️ **주의:** GM과 DM은 선형 시스템에 대한 주파수 영역 지표입니다. ML 기반 또는 비선형 루프에는 직접 적용되지 않을 수 있으나, 이득이나 지연 불확실성을 수동으로 증가시켜 이 여유를 추정할 수 있습니다.

---

## 7. 실험 결과 분석

### 7.1 명목 추적 성능 비교

| 기준 | LQI | DDPG₁ | DDPG₂ | 최우수 |
|------|-----|--------|--------|--------|
| $t_r$ (상승 시간) | 3.0 | 2.5 | **1.7** | DDPG₂ |
| $M_p$ (오버슈트 %) | 21.5% | 45.1% | **15.3%** | DDPG₂ |
| $M_u$ (언더슈트 %) | **7.7%** | **7.7%** | 22.5% | LQI=DDPG₁ |
| $t_s$ (정착 시간) | 6.7 | 9.4 | **3.7** | DDPG₂ |
| $e_{ss}$ (정상상태 오차) | 0 | 0 | 0 | 동일 |
| ISE | 19.4 | 20.9 | **12.8** | DDPG₂ |
| ITAE | 52.2 | 121.9 | **12.6** | DDPG₂ |
| IACE | **406.8** | 419.3 | 420.3 | LQI |
| IACER | **40.4** | 82.7 | 26.9 | DDPG₂ |
| $u_{c_{max}}$ | **2.8** | 3.4 | 4.0 | LQI |
| GM (이득 여유) | **27.8** | 2.3 | 18.1 | LQI |
| DM (지연 여유) | **0.85** | 0.08 | 0.55 | LQI |

### 7.2 외란/노이즈 조건에서의 성능

외란(진폭 0.2, 15초 주입) + 백색 노이즈(표준편차 0.1, 20초 주입):

| 기준 | LQI | DDPG₁ | DDPG₂ |
|------|-----|--------|--------|
| ISE | 21.2 | 22.3 | **14.1** |
| ITAE | 268.3 | 297.6 | **170.7** |
| IACE | **446.6** | 477.2 | 481.4 |
| IACER | **72.7** | 237.6 | 62.8 |

> 🔑 **흥미로운 결과:** DDPG 제어기는 외란/노이즈 조건에 대해 특별히 훈련받지 않았음에도 LQI보다 약간 우수한 성능을 보임

### 7.3 핵심 결론

| 측면 | 결론 |
|------|------|
| 추적 성능 | DDPG₂가 대부분의 기준에서 우수 |
| 제어 신호 에너지 | LQI가 더 경제적 |
| **강건성 (GM, DM)** | **LQI가 현저히 우수** |
| 외란/노이즈 | DDPG₂가 약간 우수 |
| 초기 조건 | 유사한 성능 |

---

## 8. Colab 실습 가이드

### 8.1 실습 환경 설정

```python
# Google Colab에서 실행
!pip install stable-baselines3 gymnasium torch numpy matplotlib scipy control
```

### 8.2 실습 구성

| 실습 | 내용 |
|------|------|
| Lab 1 | CartPole 환경에서 DDPG 기초 구현 |
| Lab 2 | NMP 시스템 환경 구축 및 LQI 제어기 설계 |
| Lab 3 | DDPG₁, DDPG₂ 제어기 훈련 및 비교 |
| Lab 4 | 12가지 성능 지수 계산 및 시각화 |
| Lab 5 | 외란/노이즈 강건성 테스트 |

### 8.3 실습 코드 파일

→ **[`DDPG.ipynb`](./DDPG.ipynb)** 참조

---

## 9. 한계점 및 향후 연구방향

### 9.1 DDPG의 한계

1. **대규모 훈련 에피소드 필요:** 모델 프리 접근법 특성상 많은 샘플 필요
2. **하이퍼파라미터 민감도:** 약간의 파라미터 변경으로도 수렴/발산이 달라짐
3. **낮은 강건성:** LQI 대비 이득 여유(GM) 및 지연 여유(DM) 현저히 낮음
4. **계산 부담:** 훈련에 상당한 계산 자원 필요

### 9.2 향후 연구방향

- **모델 기반 DRL 통합:** 모델 기반 제어기와 DRL 기법의 결합
- **하이퍼파라미터 자동 최적화:** Bayesian Optimization 등 적용
- **TD3 (Twin Delayed DDPG):** DDPG의 과대평가 문제를 해결한 개선판
- **SAC (Soft Actor-Critic):** 엔트로피 정규화로 탐험-활용 균형 개선
- **실시간 제어 적용:** 시뮬레이션에서 실제 시스템으로의 전이 학습

### 9.3 DRL vs 고전 제어기 선택 가이드

```
모델이 존재하는가?
    ├── Yes → 시스템이 복잡한가?
    │         ├── No  → LQI/PID 등 고전 제어기 권장 (안정적, 해석 가능)
    │         └── Yes → 모델 기반 DRL 또는 하이브리드 접근법 고려
    └── No  → DRL (모델 프리) 접근법
                  ├── 복잡한 환경 → DDPG/TD3/SAC
                  └── 단순한 환경 → 고전 제어기로 충분할 수 있음
```

---

## 📖 참고 문헌

1. Lillicrap, T. P., et al. (2016). *Continuous control with deep reinforcement learning*. ICLR 2016. [arXiv:1509.02971]
2. Tavakkoli, F., et al. *Model Free Deep Deterministic Policy Gradient Controller for Setpoint Tracking of Non-minimum Phase Systems*. IEEE (submitted).
3. Silver, D., et al. (2014). *Deterministic Policy Gradient Algorithms*. ICML 2014.
4. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
5. Fujimoto, S., et al. (2018). *Addressing Function Approximation Error in Actor-Critic Methods (TD3)*. [arXiv:1802.09477]

---


