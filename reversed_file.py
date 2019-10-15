with open("encrypted.txt", "r") as f, open("passwords.txt", "w") as h:
    h.writelines(f.readlines()[::-1])
