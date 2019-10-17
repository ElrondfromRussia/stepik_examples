# 11c0d3dc6093f7442898ee49d2430d20
# http://api.openweathermap.org/data/2.5/weather?q=London&APPID=11c0d3dc6093f7442898ee49d2430d20
import requests

# 11c0d3dc6093f7442898ee49d2430d20
# http://api.openweathermap.org/data/2.5/weather?q=London&APPID=11c0d3dc6093f7442898ee49d2430d20


# api_url = "http://api.openweathermap.org/data/2.5/weather"
# params = {
#     'q':'Moscow',
#     'appid':'11c0d3dc6093f7442898ee49d2430d20',
#     'units':'metric'
# }
# res = requests.get(api_url, params=params)
# print(res.status_code)
# print(res.headers['Content-Type'])
# data = res.json()
# print(data['main']['temp'])

# ###########################################


# patt = "http://numbersapi.com/{}/math?json=true"
# nums = []
# while True:
#     try: nums.append(int(input()))
#     except: break
# for n in nums:
#     print("Interesting") if requests.get(patt.format(n)).json()['found'] else print("Boring")

# ##########################################
import requests
import json

client_id = 'd9b4136d42f467dadacd'
client_secret = 'c204116c352c793bf353f672754f0e07'
r = requests.post("https://api.artsy.net/api/tokens/xapp_token",
                  data={
                      "client_id": client_id,
                      "client_secret": client_secret
                  })
j = json.loads(r.text)
token = j["token"]

headers = {"X-Xapp-Token" : token}
patt = "https://api.artsy.net/api/artists/{}"

names = []
while True:
    inp = input()
    if inp == '': break
    names.append(inp)
res = {}
for n in names:
    r = requests.get(patt.format(n), headers=headers)
    j = json.loads(r.text)
    try:
        res[j['birthday']].append(j['sortable_name'])
    except KeyError:
        res[j['birthday']] = [j['sortable_name']]
print(res)
sor_set = sorted(res, key=lambda x: int(x))
print(sor_set)
for s in sor_set:
    res[s].sort()
    for n in res[s]: print(n)
