import os
import os.path
import shutil

shutil.unpack_archive('main.zip', '.')
with open("res.txt", "w") as res:
    fs = []
    for cur, dirs, files in os.walk('main'):
        print(cur, dirs, files)
        if len(list(filter(lambda x: x.__contains__(".py"), files))) > 0:
            fs.append(cur)
    print(fs)
    res.writelines("\n".join(fs))


