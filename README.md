# dqn_bitcoin_auto_trading
Try rainforcement learning AI auto trading using tensorflow(RNN, FC + DQN).

## 학습코드
train.py

## Environment (coin sequence data)
env/train.py
==> real.py 추가 예정

## Neural net design
![image](https://user-images.githubusercontent.com/15869525/46462596-d12eee80-c7fc-11e8-8325-924e745799ad.png)



## Learning result (Starting Money : 1,000,000)
### Episode 0
Bankruptcy in half a year (최종 잔액 = balance)
![image](https://user-images.githubusercontent.com/15869525/46568642-14b36500-c983-11e8-94fb-d9e0e3223f1f.png)

### Episode 44
Starting no bankruptcy (balance = 2178)
![image](https://user-images.githubusercontent.com/15869525/46568672-b0dd6c00-c983-11e8-8b1f-26e3fc406478.png)

### Episode 87
The balance has begun to increase (balance = 15466)
![image](https://user-images.githubusercontent.com/15869525/46568676-e5512800-c983-11e8-8a1f-b1ea3560b3b4.png)
