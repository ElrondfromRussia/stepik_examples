import json
import requests

# st1 = {
#     "fn": "Gr",
#     "scores": [
#       40,
#       20,
#       10
#     ],
#     "cert":True
#   }
# st2 = {
#     "fn": "Ad",
#     "scores": [
#       45,
#       34,
#       2
#     ],
#     "cert":True
#   }
#
# data = [st1, st2]
# st = json.dumps(data, indent=4, sort_keys=True)
# with open("studs.json", "w") as f:
#     json.dump(data, f, indent=4, sort_keys=True)
# dat_ag = json.loads(st)
# print(sum(dat_ag[0]["scores"]))
# with open("studs.json", "r") as f:
#     dt = json.load(f)
#     print(sum(dt[0]["scores"]))

import json

data = json.loads(input())
cldic = {}

def is_parent(par, child):
    if par == child:
        return True
    pars = cldic.get(child)
    if not pars:
        return False
    if par in pars:
        return True
    else:
        anses = []
        for p in pars:
            anses.append(is_parent(par, p))
        if True in anses:
            return True
    return False


def add_rep(inf):
    if len(inf) > 1:
        atrs = inf[1]
        cldic[inf[0]] = atrs
    else:
        cldic[inf[0]] = None


for d in data:
    add_rep([d['name'], d['parents']])

res = []
for d in cldic.keys():
    a = 0
    for d2 in cldic.keys():
        if is_parent(d, d2):
            a+=1
    res.append(str(d) + " : " + str(a))
res.sort()
for r in res:
    print(r)

#адекватное
# import json
# 
# def test(x, c):
#     for i in z:
#         if x in i['parents']:
#             c.add(i['name'])
#             c = test(i['name'], c)
#     return (c)
#
# z = json.loads(input())
# z.sort(key=(lambda x: x['name']))
# for i in z:
#     print(i['name'], ':', len(test(i['name'], c = set()))+ 1)
