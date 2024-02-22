---
title:  "2.빅투 환경 만들기 - 보드와 환경 클래스"
excerpt: "2.Creating Big Two environment - Board and environment class"

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
여기서는 앞서 만든 클래스들을 통해서 빅투의 환경을 완성하는 것에 대해서 설명을 하도록한다. 보드 클래스는 환경 클래스와 독립적으로 작동하며, 단순히 시작할 때 플레이어에게 카드를 나눠주고, 플레이어의 손패를 관리해주는 역할을한다. 빅투 환경 클래스에서는 보드 클래스를 이용하여, 실제 플레이어들이 할 수 있는 행동을 관리한다.

# 2.보드 클래스
보드 클래스는 단순히 플레이어들의 카드를 관리하고, 초기화시 랜덤으로 카드를 분배하는 역할을 한다. 앞서 손패 클래스와 크게 다른 부분은 없으므로 간단하게 코드만을 제공한다.

```python
class Board():
    def __init__(self):
        self.initialize()
    
    def initialize(self, seed: Optional[int] = None, player_fix_cards: Optional[np.ndarray] = None):
        if player_fix_cards is None:
            player_fix_cards = np.zeros((4, 4*13))
        if not seed is None:
            np.random.seed(seed)
        
        n_fix_card = player_fix_cards.sum(axis=1)
        fix_cards = player_fix_cards.sum(axis=0)

        shuffled_cards = np.arange(13*4)[~fix_cards.astype(bool)]
        np.random.shuffle(shuffled_cards)

        hands = np.zeros((4, 52)) + player_fix_cards

        for i in range(4):
            start = 0 if i == 0 else end
            end = 13 - int(n_fix_card[i]) + (end if i != 0 else 0)
            hands[i, shuffled_cards[start:end]] = 1

        self.hands : List[Hand] = [Hand(torch.from_numpy(hands[i]).float().view(1, 4, 13)) for i in range(players)]

        return self
    
    def get_card_tensor(self, player: int) -> torch.Tensor:
        return self.hands[player].card_tensor
    
    def get_n_card(self, player: int) -> int:
        return self.hands[player].n_card
    
    def get_all_rankings(self, player: int) -> List[HandRanking]:
        return self.hands[player].get_all_hand_rankings()
    
    def discard_cards(self, player: int, card_tensor: torch.Tensor):
        self.hands[player].discard_cards(card_tensor)
        return self

    def get_all_hand_rankings_stronger_than(self, player: int, hand_suit: HandRanking) -> List[HandRanking]:
        return self.hands[player].get_all_hand_rankings_stronger_than(hand_suit)

```

# 3.환경 클래스
결국 이때까지 만든 것들은 환경 클래스를 만들기 위함이라고해도 과언이 아니다. 기본적으로 환경 클래스는 다음과 같은 기능을 제공한다.

- 게임 진행 : ***step*** 함수를 통해서, 빅투의 규칙에 따라서 게임을 진행한다.
- 관측값 제공 : 현재 게임 상황의 관측값을 제공해준다. 추후 관측값의 변화에 유연하게 다루기 위하여 **Observer** 클래스를 통해서 관측이 이루어진다. 
- 게임기록 저장 : **Observer** 클래스가 이전 플레이어의 행동에 따라서 관측값을 반환해주기 위해서 각 플레이어가 했던 행동들을 저장한다.

각 기능들을 코드와 함께 자세히 살펴보자

## 1.게임 진행
여기선 빅투 클래스가 게임을 진행하는 방법에 대해서 알아본다. 만약 현재 나와있는 가장 강한 족보를 $r$이라고하자. 그러면, 플레이어가 가능한 행동의 수는 플레이어의 손패에서 가능한 모든 족보들을 $R_p$라고 했을 때, $\mathcal{A} = \{r' \in R_p \mid r < r'\}$이 된다. 또한 $\mathcal{A}$는 유한하므로, $\mathcal{A} = \{a_1, a_2, \dots, a_n\}$로 일관성있게 표현가능하고, 우리는 $i$-번째 행동을 족보 $a_i$를 버리는 것으로 정의할 것이다. 이를 파이썬 코드로 나타내면 다음과 같다.

```python
#현재 나와있는 가장 강한 족보보다 강한 현재 행동하는 플레이어들의 족보를 가져온다.
available_rankings = self.board.get_all_hand_rankings_stronger_than(self.turn, self.strongest_hand_ranking) 
#플레이어가 행동을 취할 시 버릴 족보
action_ranking = available_rankings[action] 
```

플레이어가 버릴 족보를 정했다고 가정해보자. 해당 족보가 Pass이면, 빅 투의 규칙에 따라서, 나와있는 가장 강한 족보는 바뀌지 않을 것이다. 하지만, 만약 플레이어가 Pass를 했을 때, 플레이어가 3번째 연속적으로 Pass를 한 것이면 나와있는 가장 강한 족보는 None이 되어야한다. (다음 플레이어가 첫 시작이므로) 플레이어가 Pass를 하지 않았다면, 당연히 나와있는 족보보다 더 강한 족보를 냈다는 것이므로 빅투의 규칙에 따라서 나와있는 가장 강한 족보는 갱신해야한다. 이를 파이썬 코드로 나타내면 다음과 같다.

```python
discard = action_ranking.cards #해당 족보를 버릴 시 버려질 카드 행렬
act_pass = (action_ranking.suit == Ranking.PASS) #플레이어가 택한 행동이 패스인가?

if act_pass:
    self.pass_count += 1 #만약 패스한 경우, 패스 카운트를 1올려준다 이 패스 카운트가 3이되면, 한 플레이어를 제외하고 나머지 모든 플레이어가 연속적으로 패스했다는 것이다.
else:
    self.pass_count = 0 #패스가 아니면 연속적인 패스가 끊긴 것이므로 패스 카운트를 초기화해준다.
    self.strongest_hand_ranking = action_ranking #패스가 아니면 플레이어가 해당 족보를 버렸다는 것이므로, 현재 나와있는 가장 강한 족보보다 강한 족보를 버린 것이다. 따라서 나와있는 가장 강한 족보를 플레이어가 택한 족보로 변경한다.

#만약 3번 연속으로 패스가 된 경우
if self.pass_count >= 3:
    self.strongest_hand_ranking = HandRanking.NONE #나와있는 가장 강한 족보는 None이 된다.
    self.pass_count = 0 #Pass 카운트를 초기화 해준다.

#Board 클래스를 통해서 플레이어가 카드를 버린 것을 처리한다.
if not act_pass:
    self.board.discard_cards(self.turn, discard)
    if self.board.hands[self.turn].card_tensor.sum().item() < 1:
        terminated = True
    
#플레이어의 턴을 넘긴다
self.turn = (self.turn + 1) % players
```

## 2.관측값 제공
앞서 언급했듯이 관측값에 따른 성능 비교를 위해서, 빅투 환경에서의 관측값은 Observer 클래스를 통해서 행해진다. Observer 클래스의 추상 클래스는 다음과 같다
```python
class Observer():
    def get_observation(self, logs, card_tensor: torch.Tensor, strongest_ranking: HandRanking) -> torch.Tensor:
        return 
```

단순히 환경에서의 로그와, 플레이어의 손패와, 나와있는 가장 강한 족보를 인수로 받고, 관측값을 반환한다. 빅투 클래스는 Observer클래스의 객체를 통해서, step시 관측값을 반환해준다.

## 3.게임 기록 저장
step시 플레이어, 손패 행렬, 가장 강한 족보의 dictionary를 리스트 형태로 단순히 저장해둔다. Observer가 observation을 만들거나, 플레이 기록을 확인하기 위해서 사용한다.

## 4.전체 코드
<details>
<summary>코드 보기</summary>

```python

class Big2Env():
    def __init__(self, observer: Observer, render = False):
        self.board: Board = Board()
        self.logs = []
        self.observer = observer
        self.strongest_hand_ranking: HandRanking
        self.pass_count = 0

    def reset(self, seed: Optional[int] = None, player_fix_cards: Optional[np.ndarray] = None):
        if player_fix_cards is None:
            player_fix_cards = np.zeros((4, 4*13))
        self.board.initialize(seed, player_fix_cards)

        #find the player who discards cards at first
        for i in range(players):
            if self.board.hands[i].card_tensor[0][sn_index('diamond' , '3')].item() > 0:
                self.turn = i
                break
        
        self.pass_count = 0
        self.strongest_hand_ranking = HandRanking.NONE
        self.logs = []
        for i in range(4):
            self.append_log(
                player = (self.turn + i)  % players,
                card_tensor = self.board.hands[(self.turn + i) % players].card_tensor,
                discard = torch.zeros(1, 4, 13),
                act_pass = False,
                strongest_ranking = HandRanking.NONE
            )
        observation = self.observer.get_observation(self.logs, self.board.hands[self.turn])
        

        return observation, {}
    
    def step(self, action: int):
        
        available_rankings = self.board.get_all_hand_rankings_stronger_than(self.turn, self.strongest_hand_ranking)
        action_ranking = available_rankings[action]

        discard = action_ranking.card_tensor
        act_pass = (action_ranking.ranking == Ranking.PASS)

        if act_pass:
            self.pass_count += 1
        else:
            self.pass_count = 0
            self.strongest_hand_ranking = action_ranking
        
        self.append_log(
            player = self.turn, 
            card_tensor = self.board.get_card_tensor(self.turn),
            discard = discard, 
            act_pass = act_pass, 
            strongest_ranking = self.strongest_hand_ranking
        )

        terminated = False
        info = {}
        reward = discard.sum().item() / 5.0
        truncated = False
        next_observation = None

        #if passed 3 times in a row
        if self.pass_count >= 3:
            self.strongest_hand_ranking = HandRanking.NONE
            self.pass_count = 0

        #discard
        if not act_pass:
            self.board.discard_cards(self.turn, discard)
            if self.board.get_n_card(self.turn) < 1:
                terminated = True
        
        #turn next player
        self.turn = (self.turn + 1) % players
        next_observation = self.observer.get_observation(
            logs = self.logs, 
            player_hand = self.board.hands[self.turn], 
        )
        
        return next_observation, reward, terminated, truncated, info
    

    def append_log(self, player, card_tensor, discard, act_pass, strongest_ranking):
        self.logs.append({'player':player, 'card_tensor':card_tensor, 'discard':discard, 'pass':act_pass, 'strongest_ranking':strongest_ranking})

        return self
    
    def hand_to_str(self, card_tensor: torch.Tensor):
        shape = ['D', 'C', 'H', 'S']
        result = ''
        for i in range(4):
            numbers = [str(i.item()) for i in torch.where(card_tensor[0][i] > 0)[0] + 2]
            if len(numbers) > 0:
                result += shape[i]
            for n in numbers:
                result += n + ' '
        return result
        
    
```

</details>
</br>