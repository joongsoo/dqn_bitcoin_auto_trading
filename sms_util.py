import requests

userid = ""
passwd = ""
sender_phone_number = ""
receiver_phone_number = ""

# 문자나라
# https://www.munjanara.co.kr
def send_sms(msg):
    sms_url = "http://211.233.20.184/MSG/send/web_admin_send.htm?userid=" \
              + userid \
              + "&passwd=" \
              + passwd \
              + "&sender=" \
              + sender_phone_number \
              + "&receiver=" \
              + receiver_phone_number \
              + "&message=" \
              + msg
    return requests.get(sms_url).text
