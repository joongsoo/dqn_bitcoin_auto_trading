

# 데이터 값 범위 정규화를 위한 보정 값 (0~1 사이의 수로 만들어주기 위함)
data_reg = [100000.0, 10000.0, 10.0, # 가격 [BTC, ETH, XRP]
            10.0, 100.0, 100000.0] # 거래량 [BTC, ETH, XRP]

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

def decode_with_idx(val, idx):
    try:
        return round(val * data_reg[idx], 10)
    except:
        return val

# 100만 단위로 나눈다.
def encode_money(money):
    return float(money) / 100000.

def encode_coin_cnt(coin_cnt):
    return float(coin_cnt) / 100.
