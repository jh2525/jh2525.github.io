---
title:  "2.빅투 환경 만들기 - 손패 클래스"
excerpt: "2.Creating Big Two environment - Hand Class"

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
여기서는 빅투 환경에서의 사용될 손패 클래스에 대해서 소개한다. 기본적으로 플레이어는 족보에 있는 카드의 조합으로만 손패를 버릴 수 있기 때문에, 가지고 있는 손패가 어떤 족보를 만족시키는지를 구현하는 것은 매우 필수적인 일이다. 여기에선, Pytorch Tensor를 기반으로 손패에서 카드를 버리고, 카드를 얻을 때마다 자동으로 가능한 족보를 나타내주는 손패 클래스의 구현에 대해서 설명한다.



# 2.족보의 추상화
빅투의 스트레이트 (플러시)를 제외한 모든 족보들은 손패 행렬의 행방향의 합 또는 열방향의 합으로 해당 족보를 충족하는지를 알 수 있다. 예를 들어서 1행(슈트 D)의 모든 원소들의 합이 5이상이면 다이아몬드로 이루어진 플러시를 만들 수 있단 뜻이며, 3열(끗수 4)의 원소들의 합이 3이상이면 끗수 4로 이루어진 싱글, 페어, 트리플을 낼 수 있다는 것을 의미한다.
![BigTwoSuit](https://github.com/jh2525/jh2525.github.io/assets/160830734/285eef28-3f24-48d7-b8dc-6a8428b2acd2)

(D4589J C1 S1의 손패 행렬)

따라서, 다음과 같이 열방향으로 n개 이상인 모든 경우의 수를 구할 수 있는 함수를 정의하자. 이 때, 추후 행방향으로 구할 때도, 일반성을 잃지 않고 같은 방식의 함수를 이용할 수 있도록 손패 행렬의 전치 여부를 결정하는 transpose 인자를 추가해준다.


 ```python
def get_n_common_cards(card_tensor: torch.Tensor, n: int, transpose: bool = False) -> List[torch.Tensor]:
    if transpose:
        card_tensor = card_tensor.permute(0, 2, 1)

    n_sums = card_tensor.sum(dim=1)
    _, numbers = torch.where(n_sums >= n)

    result = []

    for i in numbers:
        indicies,  = torch.where(card_tensor[0, :, i])
        cases = torch.combinations(indicies, n)
        for j in range(len(cases)):
            n_same_cards = torch.zeros_like(card_tensor)
            n_same_cards[0, cases[j], i] = 1.0
            
            if transpose:
                n_same_cards = n_same_cards.permute(0, 2, 1)
            
            result.append(n_same_cards)
        
    return result
 ```


그러면 스트레이트를 제외한 모든 족보들은 다음과 같은 방법으로 구할 수 있다:
- 싱글, 페어, 트리플 : 각각 $n = 1, 2, 3$인 경우에 해당한다.
- 풀하우스, 포카드 : 기존의 카드 행렬을 $H$라고 하고, 각각 $n = 3, 4$ 일 때를 모두 구하고 해당 카드 행렬들을 ${h_0, h_1, ..., h_n}$이라고하자. 다시 ${H - h_0, ..., H - h_n}$ 에대해서 $n = 2, 1$ 일때의 $i$번째 카드 행렬에 대해 구해진 카드 행렬들을 $j_i^0, j_i^1, ..., j_i^{m_i}$이라고하자. 그러면 $h_0 + j_0^0, h_0 + j_0^1, ..., h_0 + j_0^{m_0} + ... + h_n + j_n^{m_n}$이 풀하우스, 포카드에 관한 가능한 모든 카드 행렬이다. 이를 코드로 나타내면 다음과 같다: (풀하우스는 $n = 3, m = 2$, 포카드는 $n = 4, m = 1$인 경우에 해당한다.)
```python
def get_n_m_common_cards(card_tensor: torch.Tensor, n: int, m: int) -> List[torch.Tensor]:
    result = []
    n_common_cards = get_n_common_cards(card_tensor, n)
    for n_card in n_common_cards:
        m_common_cards = get_n_common_cards(card_tensor - n_card, m)
        for m_card in m_common_cards:
            result.append(n_card + m_card)
    return result
```
- 플러시 : 손패 행렬의 전치 행렬에 대해서 $n = 5$ 일때의 카드 행렬들을 구한다음에, 다시 각 행렬들을 전치시키면 된다. 앞서 만든 함수에서, transpose인자를 True로 해주면된다.
## 스트레이트
스트레이트의 경우는 다른 족보들과는 다르게 끗수가 연속적이어야 된다는 조건이 붙는다. 또한 백스트레이트라는 예외가 존재한다. 먼저 스트레이트의 경우의 수를 구하기 위해서는 단순히 루프문을 이용해서 결과를 구해도 되지만, 깔끔한 방법은 아니라고 생각한다. 따라서, 먼저 각 끗수에 대해서 카드가 존재(1)하는지를 안하는지(0)를 나타내는 (13,) 크기의 벡터를 구하고(5, ) 크기의 모든 성분의 값이 1인 벡터를 필터로 사용하여 합성곱 연산을 통해서, 값이 5이상인 위치에서 스트레이트가 가능함을 알 수 있다. 또한 백스트레이트의 경우 간단하게 손패 행렬의 1번째 열을 마지막 열에 다시 추가한 행렬을 통해서 합성곱 연산을 진행해주면, 백스트레이트의 경우도 간단하게 구할 수 있다. 다음은 코드와 해당 과정을 설명해주는 사진이다:
![straight2](https://github.com/jh2525/jh2525.github.io/assets/160830734/2815fa6a-4faf-4830-ba51-82cda82f68e0)



```python
straight_filter = torch.full((1, 1, 5,), 1.0)

def get_straight(card_tensor: torch.Tensor) -> List[torch.Tensor]:
    concat_tensor = torch.concat([card_tensor[:, :, -1:], card_tensor], dim=-1)
    column_sum = (concat_tensor.sum(dim=1) > 0).float()
    straight_num = torch.conv1d(column_sum, straight_filter)
    straight_indicies = torch.where(straight_num.squeeze() > 4)[0]

    result = []

    for i in straight_indicies:
        if i == 0: #in case of back straight
            straight_sequences = torch.Tensor([-1, 0, 1, 2, 3]).long()
        else:
            straight_sequences = torch.arange(i-1, i+4)

        straight_target = torch.cartesian_prod(
            *[torch.where(concat_tensor[0, :, i+j])[0] for j in range(5)])
        
        for j in range(len(straight_target)):
            straight_tensor = torch.zeros_like(card_tensor)
            straight_tensor[:, straight_target[j], straight_sequences] = 1.0
            result.append(straight_tensor)

    return result
```


 ## 스트레이트 플러시
스트레이트 플러시의 경우 스트레이트 손패행렬과 플러시 손패행렬에서 교집합을 구해주면된다. 또한 스트레이트와 플러시인 경우에서, 스트레이트인 경우를 제외 해주어야한다. 

# 3.손패 클래스
손패 클래스는 위에서 구현한 함수를 바탕으로 플레이어의 손패를 관리해준다. 빅투에서 플레이어가 버릴 수 있는 카드의 목록들은 각 족보에 맞는 패와, 이전에 다른 플레이어가 버린 카드를 고려하여 행동이 제한되기 떄문에 손패 클래스는 특정 플레이어가 취할 수 있는 action의 목록을 알 수 있게 해주는 핵심적인 클래스이다.

손패 클래스를 요약하자면 특정 플레이어가 카드를 얻고, 버릴 때마다 해당 플레이어의 족보의 경우의 수를 갱신해주는 클래스이다. 

## 핸드 클래스의 주요 멤버
- ***card_tensor: torch.Tensor*** : 플레이어가 가지고 있는 손패를 encoding한 행렬이다.
- ***hand_rankings: Dict[Ranking, HandRanking]*** : 플레이어의 카드가 변경될 때마다, 가능한 족보를 저장해주는 dictionary이다.
## 핸드 클래스의 주요 메소드
- ***update()*** : 플레이어의 손패의 변화가 있을 때마다, ***hand_rankings***를 갱신해주는 메소드다.
- ***discard_cards(card_tensor: torch.Tensor)*** : 인자로 받은 카드 행렬에 해당하는 카드들을 플레이어의 손패에서 버리는 메소드이다.
- ***get_all_hand_rankings_stronger_than(target_ranking: HandRanking)*** : 현재 손패에서 타겟 족보보다 강한 모든 족보들을 반환한다.

## 전체 코드
<details>
<summary>코드 보기</summary>

```python
class Hand():
    def __init__(self, card_tensor: Optional[torch.Tensor] = None):
        if card_tensor is None:
            card_tensor = torch.zeros(1, 4, 13)
        self._card_tensor = card_tensor
        self._hand_rankings: Dict[Ranking, HandRanking] = {}
        
        self.update()
    
    def get_n_card(self):
        return int(self._card_tensor.sum().item())

    
    def get_card_tensor(self):
        return self._card_tensor.clone()
    
    def update(self):
        card_tensor = self._card_tensor

        single = get_n_common_cards(card_tensor, 1)
        pair = get_n_common_cards(card_tensor, 2)
        triple = get_n_common_cards(card_tensor, 3)

        flush = get_n_common_cards(card_tensor, 5, True)
        four_card = get_n_m_common_cards(card_tensor, 4, 1)
        full_house = get_n_m_common_cards(card_tensor, 3, 2)
        straight = get_straight(card_tensor)
        
        if len(straight) == 0:
            straight_flush = []
        else:
            straight_flush_indices = torch.where(torch.max(torch.concat(straight).sum(dim=-1), dim=-1).values == 5)[0].tolist()
            straight_flush = [straight[i] for i in straight_flush_indices]
        
        if len(straight_flush) > 0 and len(flush) > 0:
            straight_flush_indices.reverse()
            for i in straight_flush_indices:
                del straight[i]

            concat_flush = torch.concat(flush)
            indicies = []
            for sf in straight_flush:
                indicies.append(torch.where((sf.expand_as(concat_flush) * concat_flush).view(concat_flush.size(0), -1).sum(dim=-1) == 5)[0].item())
                
            for i in sorted(indicies, reverse = True):
                del flush[i]
        
        for (ranking_tensors, ranking) in zip([None, single, pair, triple, straight, flush, full_house, four_card, straight_flush], Ranking):
            if ranking is Ranking.NONE:
                continue
            self._hand_rankings[ranking] = [HandRanking(card_tensor, ranking) for card_tensor in ranking_tensors]
            
        return self
    
    def discard_cards(self, card_tensor: torch.Tensor):
        self._card_tensor = self._card_tensor - card_tensor
        self.update()
        return self
    
    def get_all_hand_rankings(self) -> List[HandRanking]:
        return  [HandRanking.PASS] + list(chain(*self._hand_rankings.values()))
    
    def get_all_hand_rankings_stronger_than(self, target_ranking: HandRanking) -> List[HandRanking]:
        results = []
        for hand_ranking in self.all_hand_rankings:
            if target_ranking < hand_ranking:
                results.append(hand_ranking)
        return results
    
    card_tensor = property(get_card_tensor)
    all_hand_rankings = property(get_all_hand_rankings)
    n_card = property(get_n_card)
```

</details>
</br>

# 4.요약
지금까지 손패 클래스의 구현에 대해서 알아보았다. 빅투에서 플레이어는 항상 어느 환경에서든 동일한 행동을 취할 수 있는 것이 아니라, 자신의 손패 상황에 따라서 취할 수 있는 행동이 제한되기 때문에, 해당 손패에 해당되는 족보를 알아야되며, 플레이어가 카드를 받거나 버릴 때, 가능한 족보들의 목록은 갱신되어야된다. 손패 클래스는 족보들을 명확하게 관리할 수 있게 해준다.