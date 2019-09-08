f = open("dress_3_45_test.txt",'r')
g = open("dress_test.txt", 'w')
f1 = open("jean_3_45_test.txt",'r')
g1 = open("jean_test.txt", 'w')



for lines in f.readlines():
	ls =lines.strip().split(" ")
	if len(ls) == 2:
		if ls[1] == 'O':
			g.write(lines)
		else:
			g.write(ls[0]+" "+ls[1]+"_dress"+"\n")

	else:
		g.write("\n")



for lines in f1.readlines():
	ls =lines.strip().split(" ")
	if len(ls) == 2:
		if ls[1] == 'O':
			g1.write(lines)
		else:
			g1.write(ls[0]+" "+ls[1]+"_jean"+"\n")

	else:
		g1.write("\n")





f.close()
g.close()
f1.close()
g1.close()