import pymysql
import random
from data_convert import encode, decode, decode_with_idx
import copy
import math

SQL_GET_DATA = """
SELECT dt, avg(btc), avg(eth), avg(xrp), avg(btc_quantity), avg(eth_quantity), avg(xrp_quantity) FROM (
    SELECT
        substring(DATE_FORMAT(dt_kst, '%Y-%m-%d %H:%i'), 1, 15) dt,
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
    AND dt_kst >= STR_TO_DATE('2017-11-15 00:00:00', '%Y-%m-%d %H:%i:%s')
    ORDER BY dt_kst ASC
) A
GROUP BY dt
ORDER BY dt ASC
"""

FEES = 0.9985

IDX_XRP = 2

TARGET = IDX_XRP

class Environment:
    MODE_BUY = 0
    MODE_SELL = 1
    MODE_NONE = 2
    MODES = [MODE_BUY, MODE_SELL, MODE_NONE]

    def __init__(self, start_money, seq_size):
        connection = pymysql.connect(host='', user='', password='',
                                     db='', charset='utf8')

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
        self.buy_price = 0

    def get_random_actions(self):
        return random.randint(0, 2)

    def reset(self):
        self.money = self.start_money
        self.current_step = 0
        self.coin_cnt = 0
        self.buy_price = 0
        return self.get_current_state(), self.money, self.coin_cnt, self.buy_price

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
        current_state = self.get_current_state()
        now_price = decode(copy.copy(current_state[-1]))[TARGET]
        reward = 0

        if action == self.MODE_BUY:
            if self.money > now_price:
                buy_cnt = math.floor(self.money / now_price * FEES)
                self.money -= buy_cnt * now_price
                self.coin_cnt += buy_cnt

                # 구매 리스트에 추가
                self.buy_price = now_price
            elif self.coin_cnt == 0:
                die = True
        elif action == self.MODE_SELL:
            if self.coin_cnt != 0:
                # 이득 계산
                total_buy_price = self.coin_cnt * self.buy_price

                total_sell_price = self.coin_cnt * now_price * FEES

                # 총 판매 금액에서 총 구매 금액을 빼면 얼마가 이득인지 나온다.
                if total_sell_price > total_buy_price:
                    reward = total_sell_price / total_buy_price * 100.
                elif total_sell_price < total_buy_price:
                    reward = -(total_buy_price / total_sell_price * 100.)

                self.money += total_sell_price
                self.coin_cnt = 0
                self.buy_price = 0

        self.current_step += 1

        next_state = self.get_current_state()
        future_price = decode(copy.copy(next_state[-1]))[TARGET]
        now_money = self.money + self.coin_cnt * future_price

        # 데이터의 끝에 도달하면 클리어
        if self.current_step + self.seq_size >= len(self.data):
            clear = True

        next_money = self.money
        next_coin_cnt = self.coin_cnt

        if self.coin_cnt == 0 and self.money < future_price:
            die = True

        return self.current_step, now_money, next_state, next_money, next_coin_cnt, next_avg_buy_price, reward, die, clear

