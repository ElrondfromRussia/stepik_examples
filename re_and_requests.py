
# print(re.match)
# print(re.findall)
# print(re.sub)


# pattern = r"((\w+)-\2)"
# string = "test-test chow-chow"
# dupls = re.findall(pattern, string)
# match = re.match(pattern, string)
# print(dupls)

# import re
# import sys
#

# pattern = r"(\w)\1+"
# for line in sys.stdin:
#     line = line.rstrip()
#     print(re.sub(pattern, r"\1", line))

# pattern = r"[(10)+|(01)+|(00)+]"
# for line in sys.stdin:
#     line = line.rstrip()
#     if line.count(" ") > 0:
#         continue
#     gr = re.findall(pattern, line)
#     if gr:
#         a = 0
#         b = 0
#         for i, g in enumerate(gr):
#             if i % 2 == 0:
#                 a += g == '1'
#             else:
#                 b += g == '1'
#         if (a - b) % 3 == 0:
#             print(line)

import requests
import re
from itertools import groupby


# pattern = r"<a href=\"(.+)\">"
# url1 = input().strip()
# url2 = input().strip()
# all_urs = re.findall(pattern, requests.get(url1).text)
# ne = []
# for u in all_urs:
#     try:
#         ne.extend(re.findall(pattern, requests.get(u).text))
#     except:
#         pass
# if url2 in ne:
#     print("Yes")
# else:
#     print("No")

pattern = r"<a.*href=[\'\"](?:[\w\d]*://)*(\w+[\w\d\.-]*)[\'\":/].*>"
all_urs = re.findall(pattern, requests.get(input().strip()).text)
all_urs.sort()
new_all = [el for el, _ in groupby(all_urs)]
for u in new_all:
    print(u)
