f = open("dress_3_45_train.txt",'r')
g = open("jean_3_45_train.txt",'r')
fg = open("dress_jean_train.txt",'w')

#flag = 0
for lines in f.readlines():
	ls = lines.strip().split(" ")
	if len(ls) == 2:
		if ls[1] == 'O':
			fg.write(lines)
		else:
			fg.write(ls[0]+" "+ls[1]+"_dress"+"\n")

	else:

		fg.write("\n")

		while(1):
			l = g.readline()
			if (l == ''):
				#flag =1
				break
			gs = l.strip().split(" ")
			if (len(gs) != 2):
				fg.write("\n")
				break

			else:
				if gs[1] == 'O':
					fg.write(l)
				else:
					fg.write(gs[0]+" "+gs[1]+"_jean"+"\n")

	#if flag ==1:
		#break



f.close()
g.close()
fg.close()