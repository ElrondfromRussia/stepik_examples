import numpy as np

n = 4
max_c = n*n
narr = np.eye(n)
strt = 0
st = 0
blocked = 0
where_strt = 1
where_st = 1
padding_strt = 0
padding_st = 0
cnt = 1
while cnt <= max_c:
    narr[strt][st] = cnt
    cnt += 1

    if blocked == 0:
        st += where_st
    else:
        strt += where_strt

    if blocked == 0:
        if st >= n - padding_st:
            st = n - padding_st - 1
            blocked = 1
            strt += where_strt
            where_st = -1
        elif st < 0 + padding_st - 1:
            st = 0 + padding_st - 1
            blocked = 1
            strt += where_strt
            where_st = 1
            padding_strt += 1
    else:
        if strt >= n - padding_strt:
            strt = n - padding_strt - 1
            blocked = 0
            st += where_st
            where_strt = -1
            padding_st += 1
        elif strt <= 0 + padding_strt - 1:
            strt = 0 + padding_strt
            blocked = 0
            st += where_st
            where_strt = 1


print(narr)
