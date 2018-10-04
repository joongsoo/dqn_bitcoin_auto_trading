

# 데이터 값 범위 정규화를 위한 보정 값 (0~1 사이의 수로 만들어주기 위함)
data_reg = [10000000.0, 1000000.0, 1000.0, # 가격 [BTC, ETH, XRP]
            1000.0, 10000.0, 10000000.0] # 거래량 [BTC, ETH, XRP]

def encode(li):
    for i in range(len(li)):
        li[i] = li[i] / data_reg[i]
    return li

def decode(li):
    for i in range(len(li)):
        try:
            li[i] = round(li[i] * data_reg[i], 10)
        except:
            pass
    return li

def encode_with_idx(val, idx):
    try:
        return val / data_reg[idx]
    except:
        return val

def decode_with_idx(val, idx):
    try:
        return round(val * data_reg[idx], 10)
    except:
        return val

# 10억단위로 나눈다.
def encode_money(money):
    return float(money) / 1000000000.

def encode_coin_cnt(coin_cnt):
    return float(coin_cnt) / 100000.
