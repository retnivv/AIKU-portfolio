# (CS231n) Lecture 14 | Deep Reinforcement Learning

Created: 2025년 5월 25일 오전 12:03

## Reinforcement Learning (강화학습)

환경 내에서 **행동**을 취할 수 있는 Agent가 있고, Agent는 행동에 대한 **보상**을 받음.

**Goal :** **보상을 최대화시킬 수 있는 행동을 학습하는 것.**

![image.png](image.png)

Environment는 Agent에게 State(s_t)를 제공하고, Agent는 이를 바탕으로 Action(a_t)을 취한다.

여기에 대해 Environment는 Reward(r_t)와 다음 State(s_t+1)을 제공한다.

위의 과정을 반복하다가 Environment가 최종 상태가 되면 episode를 종료한다.

### Ex)

![image.png](image%201.png)

## **Markov Decision Process (MDP)**

![image.png](image%202.png)

강화학습 문제의 수학적 공식화.

MDP는 다음과 같은 Markov property를 만족한다.

(**Markov property** : 현재 상태가 전체 세계의 상태를 완전히 결정짓는다.)

![image.png](image%203.png)

time step t=0부터 시작. 초기 상태 s_0를 sampling함.

그리고 종료될 때까지 다음을 반복함 :

1. Agent가 action을 취함
2. 보상 sampling
3. next state sampling
4. Agent가 보상과 next state를 전달받음.

**policy(Pi)** : S → A 는 각 state에서 어떤 action을 취할지 결정하는 함수.

policy는 결정론적일 수도 있고 확률론적일 수도 있음.

![image.png](image%204.png)

**목표 : cumulative discounted reward 를 최대화하는 최적의 policy를 찾아내는 것.**

→ 그런데 MDP 는 초기 state와 transition probability에 대해 랜덤성을 가지고 있음.

→ reward의 합계의 **기댓값을 최대화**시키자.

### Bellman eqation

![image.png](image%205.png)

**Value function** : state가 얼마나 좋은지를 나타내는 함수.

**Q-value function** : state - action pair가 얼마나 좋은지를 나타내는 함수.

![image.png](image%206.png)

최적의 Q-value function은 위와 같이 정의됨.

그리고 이것은 위의 두 번째 식인 Bellman equation을 만족함.

해석 :

optimal Q-value function은 최적의 policy를 선택했을 때의 Q-value function이라고 해석할 수 있음.

현재 상태 s에서 a라는 action을 취할 때 예상되는 보상의 합계는 우선 현재 취한 action에 대한 보상에 이후 얻을 보상의 합계를 더한 것이라고 볼 수 있음. 그런데 여기서 항상 최선의 선택(보상이 최대화되는 선택)을 할 것이므로 이는 다음 상태인 s’와 다음 action인 a’를 사용하여 위와 같이 나타낼 수 있음. (보상이 최대화되는 a’를 선택할 것이므로)

(약간 재귀적인 느낌?)

![image.png](image%207.png)

Optimal Q-value function은 다음과 같은 반복을 통해 얻어낼 수 있음.

(i → infinity 일 때 Q가 Q_star에 수렴한다는 것이 알려져 있음.)

Problem : Not scalable. 모든 (s,a) pair에 대해 Q를 계산해야 하는데, 이는 계산량이 너무 많음.

→ **신경망을 사용해 Q(s,a)를 근사시키자.**

## Q-learning

function approximator 를 사용하여 $Q(s,a)$를 근사시킴.

![image.png](image%208.png)

이때, function approximator 가 신경망이라면, 이를 deep Q-learning 이라고 부름.

(theta : 가중치)

![image.png](image%209.png)

Forward Pass & Backward Pass.

Loss function : **목표값 y_i와 우리가 예측한 Q(s,a)의 제곱 오차.**

목표값 y_i는 Bellman equation을 만족하는 정답 함수.

Loss funciton에서 theta에 대한 gradient를 계산하면 다음과 같이 나오고 이를 이용해 Gradient update를 진행함.

각 step i마다 Q-value function을 목표값 yi에 근사시키는 과정을 반복함.

![image.png](image%2010.png)

Atari game에서 Q-network의 구조.

### Experience Replay

현재 Q-network 학습의 문제점 :

연속된 sample의 batch에서 학습을 하게 되면 **sample들 간의 상관관계**가 생기기 때문에 비효율적임.

→ **Experience Replay** : game이 실행되는 동안 얻은 상태들을 바로 사용하지 않고, table of transition의 형태로 저장해뒀다가, 이후에 무작위로 여기서 minibatch를 만들어 학습함. (sample이 연속하지 않도록.)

![image.png](image%2011.png)

Deep Q-Learning with Experience Replay의 전체 알고리즘.

(Phi : s에 대한 전처리된 feature representation. 신경망의 input.)

(capacity N : D에 저장할 수 있는 transition의 최대 개수)

의문점 : 각 반복마다 D에 하나의 transition을 저장하고, 여기서 minibatch를 sapling하는데, 그러면 학습 초기에는 minibatch를 뽑아낼 만큼의 transition이 없는 거 아닌가?

→ 그래서 학습 초기에는 학습을 하지 않고 transition을 쌓기만 하는 **“replay buffer warm-up”** 단계가 있음. 이 단계는 D에 transition이 일정 개수만큼 쌓일 때까지  계속된다고 함.

또한 이후 특정  transition이 샘플링 됐어도 이 transition은 재사용 가능. 계속 D에 남아있음. 

그러다가 transition이 점점 쌓여서 D의 크기가 N보다 커지게 되면 오래된 것부터 삭제(FIFO)

## Policy Gradients

Q-Learning의 문제점 : **Q-function이 매우 복잡할 수 있음.**
(ex. 높은 차원의 state에서 로봇이 어떤 물체를 집는 동작을 학습시키고자 할 때, state, action pair가 매우 많고 학습하기 어려움.)

→ **policy gradients** : **policy를 직접 학습함**. (ex. just close your hand)

![image.png](image%2012.png)

PI : 매개변수화된 policy들의 집합. (매개변수 : theta)

각 policy를 평가하기 위한 Value function J(theta)를 위와 같이 정의함. Q-learning에서 사용했던 것과 유사함. 조건 부분이 s,a,pi 에서 pi로 바뀐 것. (policy만 고려)

value funciton의 값을 최대화 시키는 theta를 찾고 싶음.

→ Gradient ascent 수행

![image.png](image%2013.png)

여기서 r은  trajectory에 대한 보상의 총합임.

trajectory = (s0,a0,r0,s1,…)

J(theta)의 gradient를 구하고 싶은데 이는 intractable임. 따라서 다음과 같은 조작을 통해 적분 안에 p(tau)가 나오게 만들어 이를 **기댓값 형태**로 나타냄. 이후 **Monte Carlo sampling**을 통해 추정 가능.

(tau를 sampling하여 대괄호 안의 값을 계산하고 이 값들의 평균을 계산함.)

→ transition probability를 몰라도 되는가? → yes. 

![image.png](image%2014.png)

log p(tau)에 대한 gradient를 계산해보면 다음과 같이 pi_theta에 대한 식만 남는다는 것을 알 수 있음.

따라서 **transition probability를 모르더라도 J(theta)를 sampling을 통해 추정 가능.**

![image.png](image%2015.png)

직관적 해석 : 

특정 trajectory에 대한 보상이 높았다면, 이 과정에서 선택했던 모든 action들의 확률을 높이고, 

보상이 낮았다면 선택했던 action들의 확률을 낮춤.

최종 보상이 높았을 때, 그동안 선택했던 모든 action들이 좋은 action은 아닐 수 있지만, 여러 번 sampling을 하다보면 결국 평균으로 수렴할 것임.

하지만, 위에 언급한 이유 때문에, variance가 높다는 문제가 있긴 함. → **Variation reduction**

### Variation reduction

![image.png](image%2016.png)

Value function의 gradient 추정 식을 다음과 같이 변경함.

1. 원래는 특정 trajectory 전체에서의 reward의 총합을 고려했었음. 그러나 이제 각 action이 좋은 action이었는지를 개별적으로 판단해주기 위해 **그 action을 취한 시점 이후에 대한 reward만 고려해줌.** (원래 r은 t에 대해 독립이었지만 이제는 t 별로 다르게 설정) 
2. **지연된 효과**(특정 시점에 한 action이 먼 미래에 가서야 영향을 미치는 것)를 **무시**하고 싶음. 즉, 당장 좋은 reward를 가져다주는 action들에게 더 좋은 점수를 부여하고 싶음. → dicount factor gamma를 사용해서 action을 취한 시점에서 시간이 지날수록 reward가 감소하도록 설정.

![image.png](image%2017.png)

1. 특정 action들을 선택한 결과 reward가 양수라고 해서 그 action들이 무조건 좋았다고 볼 수는 없음. 특정한 임계값 이상으로 reward를 얻었을 때만 선택했던 action들의 확률을 높이고 싶음. → **baseline function** b 설정. b는 state에 따라 다른 값을 가짐.

- baseline을 어떻게 선택할까?

→ 가장 간단하게는 지금까지 각 trajectories에서 얻었던 reward의 평균으로 설정할 수 있겠지만, 더 좋은 baseline이 있음.

![image.png](image%2018.png)

직관 : 우리는 **특정한 state에서 우리가 얻을 수 있는 reward의 기댓값보다 높은 reward를 얻을 경우, 이때 선택했던 action을 좋은 action이라고 평가하고 싶음**. 즉, 특정 state에서 얻을 수 있는 reward의 기댓값보다 특정 state에서 어떤 action을 취할 경우 얻을 수 있는 reward의 기댓값이 더 클 수록, 이 action은 좋은 action임. → **Q-value function과 value function 값의 차가 크면 좋은 action.**

※ 위의 1,2번 variation reduction 전략 (미래의 reward만 고려, 감가율 적용) 들이 이미 Q function과 value function에 포함되어 있기 때문에 여기서는 별도로 적어주지 않음. 

Problem : Q와 V를 모름.

→ Q-learning 사용.

### Actor-Critic Algorithm

![image.png](image%2019.png)

Actor : policy

Critic : Q-funciton

Policy에 의해 어떤 action을 취할 것인지가 결정되고, Q-function은 이 action들을 평가함.

policy에 의해 생성된 (s,a) pair만 다루기 때문에 위에서 봤던 Q-function Learning에 비해 계산해야 할 경우가 적음.

 **Advantage function** : Q - V (특정 state에서 어떤 action이 얼마나 좋은지를 나타내는 함수)

![image.png](image%2020.png)

정리하자면 Actor-critic algorithm은 다음과 같은 알고리즘을 가짐.

Phi : critic parameters. value function V를 근사하기 위한 신경망의 매개변수.

**V를 Q에 가까워지도록 학습시키는 이유** : 

초기의 V가 있고, 이것에 대해 A = Q - V를 계산해서 좋은 action의 확률은 높이고, 나쁜 action의 확률은 줄임 → 그렇게 되면 특정 state에서 얻을 수 있는 reward의 기댓값도 변할 것이므로 V도 업데이트 해줘야함. 그리고 그 방향은 Q에 가까워지는 방향으로 설정됨.

Phi의 업데이트가 위와 같이 되는 이유 :

우리는 신경망 근사를 통해 V가 Q에 가깝게 학습시키고 싶음. 이를 L2 loss를 사용해서 학습시킬 건데 A가 Q-V로 정의되므로 Q와 V에 대한 L2 loss는 위와 같이 표현할 수 있음.

## Recurrent Attention Model (RAM)

이미지 분류 작업을 수행할 때, 전체 이미지를 처리하는 것이 아니라, focusing할 특정 영역들을 결정해서 그 부근에서 이미지를 처리하여 결정을 내림.

→ computation resource를 절약할 수 있고, 큰 이미지를 효율적으로 처리할 수 있음. 또한 복잡하고 관련 없는 부분을 무시함으로써 실제 분류 성능을 높여줄 수 있음.

glimpse를 결정하는 과정은 미분불가능하기 때문에, 여기에 강화학습이 활용됨.

![image.png](image%2021.png)

s, a, r을 다음과 같이 설정.

(지금까지 봤던 glimpses(집중해서 볼 부분)을 state, 다음으로 선택할 glimpse를 action으로 설정.)

![image.png](image%2022.png)

RAM의 이미지 처리 과정을 그림으로 나타내면 다음과 같음.

### Example of policy gradients :

![image.png](image%2023.png)

알파고의 학습 과정. (지도학습 + 강화학습)

프로 바둑 기보를 바탕으로 지도학습을 통해 policy network를 초기화함.

이후 policy gradient를 통해 학습.
