---
title:  "2.빅투 환경 만들기 - 족보 클래스"
excerpt: "2.Creating Big Two environment - Hand Ranking Class"

categories:
  - BigTwo
tags:
  - [BigTwo, RL]

toc: true
toc_sticky: true
 
date: 2024-02-22
last_modified_at: 2024-02-22
---

# 1.개요
여기서는 빅투 환경에서의 사용될 족보 클래스에 대해서 설명한다. 빅투의 경우 이전 플레이어가낸 족보에 따라서 플레이어의 행동이 제한되기 때문에, 두 족보의 강함을 비교하는 것은 매우 중요하다. 따라서, 우리는 특정한 족보를 만족하는 손패 행렬이 주어졌을 때, 해당 손패 행렬을 통해서 족보 클래스를 만들어, 족보들의 강함을 비교하는데 사용할 것이다.

# 2.손패의 Encoding
기본적으로 플레잉 카드는 13개의 끗수와 4개의 슈트로 구성되어있다. 또한, 빅투에서 각 슈트와 끗수의 카드의 개수는 1개로 유일하기 때문에, 우리는 카드의 있고 없음을 0(없음)과 1(있음)으로 표현이 가능하다. 따라서, (4, 13)의 크기의 행렬을 통해서 손패를 나타낼 수 있다. 따라서, 앞으로 카드들을 다음과 같은 방식으로 Encoding해서 카드들을 나타낸 것을 카드 행렬이라고하자.
![encoding](https://github.com/jh2525/jh2525.github.io/assets/160830734/26b070bf-cab5-4727-928d-ce70b28f7f27)

# 3.족보의 일반화
빅투에서 명시적인 족보의 강함만을 비교할 수 있다고해서, 플레이어가 할 수 있는 행동을 모두 알 수 있는 것은 아니다. 예를 들어서, 다음과 같은 상황들이 존재한다:

- **플레이어가 첫 번째 카드를 내는 경우** : 예를 들어서, 시작할 때 D3을 갖고 있는 플레이어나, 해당 플레이어를 제외한 나머지 플레이어가 연속적으로 패스를 했을 때 플레이어는 어떤 족보도 낼 수 있다. 이 때는 모든 족보의 경우가 가능하기에 족보의 비교를 통해서 낼 수 있는 카드를 알 수 없다.
- **패스** : '패스'라는 행위 자체는 다른 족보와 비교할 수 없다.  

우리는 빅투 환경에서 단순히 족보들의 비교만으로 플레이어가 가능한 행동들을 알고싶은 것이기에 다음과 같은 일반화된 족보 두개를 추가한다.

- **None** : 3번의 연속적인 패스가 이루어졌거나, 플레이어가 첫 번째 플레이어 일 때, 현재 나와있는 가장 강한 족보를 None 족보로 정의한다.
- **Pass** : 현재 나와 있는 가장 강한 족보가 'None'이 아닐 때, 플레이어가 아무 카드도 버리지 않는 것을 Pass 족보로 정의한다.

이 두 족보를 통해서 우리는 플레이어가 가능한 행동을 모두 족보들의 비교를 통해서만 알 수 있게 된다. 구체적인 과정은 다음과 같다:

모든 None과 Pass를 포함한 가능한 족보들의 집합을 $\mathcal{R}$ 이라고하자. 

그러면  $\mathcal{R}$에 다음과 같은 원순서(preorder)를 줄 수 있다:

임의의 $r, r' \in \mathcal{R}$에 대하여, </br>
(1) if $r \neq \text{Pass}, \text{None} < r$, </br>
(2) if $r \neq \text{None}, \ r < \text{Pass}$, </br>
(3) if $r, r' \notin \{\text{None, Pass}\}, $ 족보 $r'$이 $r$ 보다 강하고, 버리는 카드의 개수가 같으면 $r < r'$ 그리고 </br>
(4) (1), (2), (3)의 경우가 아니면, $r < r'$은 항상 거짓이다.

그러면 현재 나와있는 가장 강한 족보를 $r$ 어떤 플레이어의 손패에 대해서 모든 가능한 족보를 $\mathcal{R}_p$ 라고한다면, 단순히 플레이어가 가능한 버릴 수 있는 족보들의 집합 $\mathcal{A}_p$는 다음과 같다. $\mathcal{A}_p = \{r' \in \mathcal{R}_p \mid r < r' \}$

# 4.족보의 강함의 수치화
만약 어떤 두 족보가 Pass도 None도 아니고 같은 족보라면 거기서 가장 강한 카드로 족보의 강함이 결정된다. 그렇다면, 어떤 족보의 동일 족보내에서 강함은 수치화 할 수 있다. D3을 0으로 시작해서 S2를 51까지로 오름차순으로 족보내에서 강함(power)로 정의하면, 같은 족보끼리의 비교에서 유용하게 이용할 수 있다.  
![power_matrix](https://github.com/jh2525/jh2525.github.io/assets/160830734/3465afb8-cf6e-424f-bd23-4c5f4537d952)

따라서 단순히, 특정 족보의 카드 행렬이 주어진다면, 위의 사진과 같이 강함 행렬(power matrix)를 정의해주고, 인덱싱을 통해서 구한 강함값 중 최댓값이 그 족보의 강함이 되는 것이다. 이를 코드로 작성하면 다음과 같다:
```python
power_matrix = torch.arange(0, max_num * kind).view(max_num, kind).T.roll(1, dims=1) 

def get_maximum_number_power(hand_tensor: torch.Tensor) -> int:
    """
    인자로 주어진 손패 행렬에 대해서, 가장 강한 끗수와 슈트를 대소가 바뀌지 않는 숫자로 반환합니다.
    """
    if 3 != hand_tensor.dim():
        raise ValueError
    _, suits, numbers = torch.where(hand_tensor)
    return power_matrix[suits, numbers].max().item()
```

# 5.족보 클래스

**3.족보의 일반화**에 나왓듯이 이제 족보의 강함을 비교할 수 있는 클래스는 단순히 어떤 손패 행렬을 받았을 때, 해당 손패 행렬과, 해당 손패 행렬에 해당하는 족보의 이름을 가진 클래스와 가장 높은 끗수를 멤버로 구성하고, '<' 연산자를 오버로딩만 해주면 쉽게 구현할 수 있다. 족보 클래스 구현 코드는 길지 않으므로 주석과 함께 코드를 작성해보면 다음과 같다.


## 족보의 열거형 데이터
```python
class Ranking(Enum):
    NONE = 0
    SINGLE = 1
    PAIR = 2
    TRIPLE = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_CARD = 7
    STRAIGHT_FLUSH = 8
    PASS = 9
```

## 족보 클래스

```python
def get_maximum_number_power(hand_tensor: torch.Tensor) -> float:
    """
    인자로 주어진 손패 행렬에 대해서, 가장 강한 끗수와 슈트를 대소가 바뀌지 않는 숫자로 반환합니다.
    """
    if 3 != hand_tensor.dim():
        raise ValueError
    _, suits, numbers = torch.where(hand_tensor)
    return power_matrix[suits, numbers].max().item()


class HandRanking:

    def __init__(self, card_tensor: torch.Tensor, ranking: Ranking):
        self._card_tensor = card_tensor #손패 행렬
        self._ranking = ranking #족보 이름
        if ranking in [Ranking.SINGLE, Ranking.PAIR, Ranking.TRIPLE, Ranking.STRAIGHT, Ranking.STRAIGHT_FLUSH, Ranking.FLUSH]:
            self.power = get_maximum_number_power(card_tensor) #싱글, 페어, 트리플, 스트레이트, 스트레이트 플러시인 경우는 단순히 가장 강한 끗수와 슈트를 비교하기만 하면된다.
        elif ranking in [Ranking.FULL_HOUSE, Ranking.FOUR_CARD]: #풀하우스나, 포카드인 경우는 가장 많은 사용한 끗수의 가장 강한 슈트와 끗수를 비교해야한다.
            target = torch.zeros_like(card_tensor)
            target_indicies = card_tensor.sum(dim=1).argmax()
            target[:, :, target_indicies] = card_tensor[:, :, target_indicies] #가장 많이 사용한 끗수의 카드들만 1로 만들어서 족보를 구성하는 가장 강한 끗수와 슈트를 구한다.
            self.power = get_maximum_number_power(target)
        else:
            self.power = 0 #None 또는 Pass인 경우는 항상 power가 0으로 정의한다.

    def __lt__(self, other):
        ranking_value = self._ranking.value
        if self._ranking == Ranking.NONE:
            return (other.ranking != Ranking.PASS)
        elif other.ranking == Ranking.PASS:
            return True
        elif ranking_value <= 3:
            return (ranking_value == other.ranking.value) and (self.power < other.power)
        else:
            if ranking_value < other.ranking.value:
                return True
            else:
                return (ranking_value == other.ranking.value) and (self.power < other.power)

    def get_card_tensor(self):
        return self._card_tensor.clone()
    
    def get_ranking(self):
        return self._ranking
    
    card_tensor = property(get_card_tensor)
    ranking = property(get_ranking)

HandRanking.PASS = HandRanking(torch.zeros((1, 4, 13)), Ranking.PASS) #Pass 족보는 항상 똑같기 때문에 정적 변수로 생성해준다
HandRanking.NONE = HandRanking(torch.zeros((1, 4, 13)), Ranking.NONE) #None 족보도 마찬가지
```

