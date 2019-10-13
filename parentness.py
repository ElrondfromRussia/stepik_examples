cldic = {}

def add_rep(inf):
    if len(inf) > 1:
        atrs = inf[1].split()
        cldic[inf[0]] = atrs
    else:
        cldic[inf[0]] = None

def is_parent(par, child):
    if par == child:
        return "Yes"
    pars = cldic.get(child)
    if not pars:
        return "No"
    if par in pars:
        return "Yes"
    else:
        anses = []
        for p in pars:
            anses.append(is_parent(par, p))
        if "Yes" in anses:
            return "Yes"
    return "No"
            

n = int(input())
while n:
    n-=1
    cl = input().split(" : ")
    add_rep(cl)
    
        
n = int(input())
while n:
    n-=1
    clat = input().split()
    print(is_parent(clat[0], clat[1]))
    


