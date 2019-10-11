nsps = {'global':[]}
parents = {'global':None}

def create(nsp, par):
    if nsp not in nsps.keys():
        nsps[nsp] = []
        parents[nsp] = par
    
def add(nsp, var):
    if nsp in nsps.keys():
        if var not in nsps.get(nsp):
            nsps.get(nsp).append(var)
   
def get(nsp, var):
    if nsp not in nsps.keys():
        return None
    elif var in nsps.get(nsp):
        return nsp
    else:
        return get(parents.get(nsp), var)

funcs = {'create':create, 'add':add, 'get':get}

def printthem():
    print("____________")
    print(nsps)
    print(parents)
    print("____________")

n = int(input())
while n:
    n-=1
    cmd, namesp, arg = input().split()
    if cmd == "print":
        printthem()
    else:
        if cmd == "get":
            print(funcs.get(cmd)(namesp, arg))
        else:
            funcs.get(cmd)(namesp, arg)

