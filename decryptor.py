import simplecrypt

with open("encrypted.bin", "rb") as inp:
    encrypted = inp.read()

with open("passwords.txt", "r") as psw:
    pasws = psw.read()
    pasws = pasws.split("\n")
    for ps in pasws:
        try:
            ps = ps.strip()
            decr = simplecrypt.decrypt(ps, encrypted)
        except:
             pass
        else:
            print(decr)
            print("___________")
