import pymysql
import random
from data_convert import encode, decode, decode_with_idx
import copy

SQL_GET_DATA = """
SELECT dt, avg(btc), avg(eth), avg(xrp), avg(btc_quantity), avg(eth_quantity), avg(xrp_quantity) FROM (
    SELECT
        DATE_FORMAT(dt_kst, '%Y-%m-%d %H:%i') dt,
        CASE WHEN code='CRIX.UPBIT.KRW-BTC' THEN trade_price END AS btc,
        CASE WHEN code='CRIX.UPBIT.KRW-ETH' THEN trade_price END AS eth,
        CASE WHEN code='CRIX.UPBIT.KRW-XRP' THEN trade_price END AS xrp,

        CASE WHEN code='CRIX.UPBIT.KRW-BTC' THEN acc_trade_volume END AS btc_quantity,
        CASE WHEN code='CRIX.UPBIT.KRW-ETH' THEN acc_trade_volume END AS eth_quantity,
        CASE WHEN code='CRIX.UPBIT.KRW-XRP' THEN acc_trade_volume END AS xrp_quantity
    FROM bitcoin.upbit 
    WHERE (code='CRIX.UPBIT.KRW-BTC' 
    OR code='CRIX.UPBIT.KRW-ETH' 
    OR code='CRIX.UPBIT.KRW-XRP')
    AND dt_kst >= STR_TO_DATE('2017-09-25 22:05:00', '%Y-%m-%d %H:%i:%s')
    ORDER BY dt_kst ASC
) A
GROUP BY dt
ORDER BY dt ASC
"""

IDX_XRP = 2

TARGET = IDX_XRP

class Environment:
    MODE_BUY = 0
    MODE_SELL = 1
    MODE_NONE = 2
    MODES = [MODE_BUY, MODE_SELL, MODE_NONE]

    def __init__(self, start_money, seq_size):
        connection = pymysql.connect(host='비밀', user='비밀', password='비밀',
                                     db='bitcoin', charset='utf8')

        # 데이터를 불러오고 없는 데이터는 바로 이전의 데이터로 채워넣는다.
        data = []
        with connection.cursor() as cursor:
            cursor.execute(SQL_GET_DATA)
            rows = cursor.fetchall()
            for row in rows:
                r = []
                for v in range(1, len(row)):
                    if row[v] == None:
                        r.append(decode_with_idx(float(data[-1][v - 1]), v - 1))
                    else:
                        r.append(float(row[v]))
                data.append(encode(r))
        connection.close()

        self.data = data
        self.current_step = 0
        self.start_money = start_money
        self.money = start_money
        self.seq_size = seq_size
        self.coin_cnt = 0.

    def get_random_actions(self):
        return random.randint(0, 2)

    def reset(self):
        self.money = self.start_money
        self.current_step = 0
        self.coin_cnt = 0
        return self.get_current_state(), self.money, self.coin_cnt

    def get_current_state(self):
        return self.data[self.current_step : self.current_step+self.seq_size]

    # 규칙
    # 1. 한번 구매시 보유금의 50%를 사용한다. 그 다음에 구매시에도 잔액의 50%만 사용 (잔액 < 코인가격 일경우 패널티)
    # 2. 판매시에는 전량 판매한다.
    # 보상 : 코인이 포함된 자산
    def step(self, action):
        if action not in self.MODES:
            raise Exception("Invalid action")

        die = False
        clear = False
        penalty = False
        current_state = self.get_current_state()
        now_price = decode(copy.copy(current_state[-1]))[TARGET]
        before_money = self.money + self.coin_cnt * now_price
        before_coin_cnt = self.coin_cnt

        if action == self.MODE_BUY:
            available_money = round(self.money / 2)

            if available_money > now_price:
                buy_cnt = round(available_money / now_price)
                self.money -= buy_cnt * now_price
                buy_cnt = buy_cnt * 0.9985
                self.coin_cnt += buy_cnt
            elif self.coin_cnt == 0:
                die = True
        elif action == self.MODE_SELL:
            if self.coin_cnt == 0:
                penalty = True
            else:
                self.money += self.coin_cnt * now_price * 0.9985
                self.coin_cnt = 0

        self.current_step += 1

        next_state = self.get_current_state()
        future_price = decode(copy.copy(next_state[-1]))[TARGET]
        now_money = self.money + self.coin_cnt * future_price
        reward = now_money

        if penalty:
            reward = -10000

        # 데이터의 끝에 도달하면 클리어
        if self.current_step + self.seq_size >= len(self.data):
            clear = True

        return self.current_step, before_money, before_coin_cnt, now_money, next_state, reward, die, clear

