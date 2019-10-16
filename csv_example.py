import csv

with open("Crimes.csv") as f:
    reader = csv.reader(f)
    headers = reader.__next__()
    i1 = headers.index("Primary Type")
    i2 = headers.index("Date")
    i3 = headers.index("ID")
    prims = {}
    for row in reader:
        if row[i2].__contains__("2015"):
            if row[i1] not in prims:
                prims[row[i1]] = list(row[i3])
            else:
                prims.get(row[i1]).append(row[i3])
    m = max(list(map(lambda x: len(x), prims.values())))
    for pr in prims:
        if len(prims[pr]) == m:
            print(pr)

"""
для написания в файле значений float чисел с запятой вместо точки,
необходимо заключить эти числа в кавычки каждое
для использования переносов (для лучшей читабельности файла, например,
заключить всё предложение с переносом в кавычки
для изменения символа-разделителя - use в параметрах csv.reader(f) 
delimiter="\t" или что-то еще
для заключения в кавычки элементов - use в параметрах csv.reader(f) 
quoting=csv.QUOTE_ALL, к примеру
"""
