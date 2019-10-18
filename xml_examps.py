from xml.etree import ElementTree

# colors = {'red': 0, 'green': 0, 'blue': 0}
#
# def funct(rooter, adder):
#     colors[rooter.get('color')] += adder
#     for r in rooter:
#         funct(r, adder+1)
#
# root = ElementTree.fromstring(input())
# funct(root, 1)
# print(' '.join(map(str, colors.values())))


# <cube color="blue"><cube color="red"><cube color="green"><cube color="green"><cube color="green"><cube color="blue"></cube><cube color="green"></cube><cube color="red"></cube></cube></cube></cube></cube><cube color="red"><cube color="blue"></cube></cube></cube>

# for child in root:
#     print(child.tag, child.attrib)
#
# for el in root.iter("scores"):
#     print(el)

#greg = root[0]
# module1 = next(greg.iter("mod1"))
# cert = next(greg.iter("cert"))
# print(cert, cert.text)
# print(module1, module1.text)
#module1.text = str(float(module1.text) + 30)
# cert.set("type", "with distinition")

# descr = ElementTree.SubElement(greg, "descr")
# descr.text = "Bad boy with well knowledges"
# greg.append(descr)

# descr = greg.find("descr")
# greg.remove(descr)

#tree.write("xxmmll.xml")


