# Author: Mayank
def createWindow(vertical,filename):
    """
    creates a file: filename that contains data in form
    wordi-1 tagi-1 wordi tagi wordi+1 tagi+1 centralWord_id sentID
    this is the intermediate file to be used by findSim() function
    """

    sentId=0
    dress = ""
    if (vertical == "dress"):
        dress = f.readlines()
    else:
        dress= g.readlines()
    wordList = ["$UNK$"]
    tagList = ['O']
    for idx,line in enumerate(dress):
        line = line.strip().split()
        if(len(line)<2):
            if(len(wordList)>2):
                wordList.append("$UNK$")
                tagList.append("O")
                for i in range(1,len(wordList)-1):
                    filename.write(wordList[i-1]+" " +tagList[i-1]+" "+wordList[i]+" " +tagList[i]+" "+wordList[i+1]+" "+tagList[i+1]+" "+str(i-1)+" "+str(sentId)+"\n")
            sentId+=1
            wordList = ["$UNK$"]
            tagList = ['O']
        else:
            wordList.append(line[0])
            if(line[1]=='O'):
                tagList.append(line[1])
            else:
                tagList.append(line[1]+"_"+vertical)

def loadGloveModel(gloveFile):
    import numpy as np
    print ("Loading Glove Model")
    with open(gloveFile, encoding="utf8" ) as f:
       content = f.readlines()
    model = {}
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model

def findSim():
    """
    finds cosine similarity of word vectors of window size 3 
    and saves output in windowLambda_dressJean.txt
    this data needs to be sorted on cosine sim
    that task is done by sort_data()
    """
    f1 = open('data/sentences_dress.txt','r')
    g1 = open('data/sentences_jean.txt','r')
    import numpy as np
    zeros = np.zeros(300)
    jeansVectors = []
    jeans = g1.readlines()
    #print("#Examples in jean",len(jeans))
    jean_window_vectors = {} # key: (sentno,centralWordPos); Value = wordvector of window 3
    dress_window_vectors = {}
    for line in jeans:
        line = line.strip().split(" ") 
        sentNo = int(line[7].strip())
        centralWordPos = int(line[6].strip())
        if(line[0].lower() in model):

            vector = model[line[0].lower()]
        else:
            vector = zeros
            
        if(line[2].lower() in model):

            vector = np.concatenate((vector,model[line[2].lower()]))
        else:

            vector = np.concatenate((vector,zeros))
        if(line[4].lower() in model):

            vector = np.concatenate((vector,model[line[4].lower()]))
        else:
            vector= np.concatenate((vector,zeros))
        jean_window_vectors[(sentNo,centralWordPos)]=vector
        jeansVectors.append(vector)
    import pickle
    with open("data/windowVectors_jean.pkl","wb")as o:
        pickle.dump(jean_window_vectors,o,pickle.HIGHEST_PROTOCOL)
    f1.close()
    g1.close()
    out = open("data/windowLambda_dressJean.txt",'w')
    zeros = np.zeros(300)
    f1 = open('data/sentences_dress.txt','r')
    dress = f1.readlines()
    g1 = open('data/sentences_jean.txt','r')

    jean = g1.readlines()
    for idx,line in enumerate(dress):
        print(idx)
        data =line
        line = line.strip().split(" ") 
        sentNo = int(line[7].strip())
        centralWordPos = int(line[6].strip())
        if(line[0].lower() in model):
            dress_vector = model[line[0].lower()]
        else:
            dress_vector = zeros       
        if(line[2].lower() in model):
            dress_vector = np.concatenate((dress_vector,model[line[2].lower()]))
        else:
            dress_vector = np.concatenate((dress_vector,zeros))
        if(line[4].lower() in model):
            dress_vector = np.concatenate((dress_vector,model[line[4].lower()]))
        else:
            dress_vector= np.concatenate((dress_vector,zeros))
        dress_window_vectors[(sentNo,centralWordPos)]=dress_vector
        for idx,jeans_vector in enumerate(jeansVectors):
            dress_norm = np.linalg.norm(dress_vector)
            jean_norm  = np.linalg.norm(jeans_vector)
            denominator = dress_norm * jean_norm
            if(denominator==0):
                dot=0.0
            else:
                dot = np.dot(jeans_vector,dress_vector)/(denominator)
            out.write(data.strip()+" "+jean[idx].strip()+"\t"+str(dot)+"\n" )
    import pickle
    with open("data/windowVectors_dress.pkl","wb")as o:
        pickle.dump(dress_window_vectors,o,pickle.HIGHEST_PROTOCOL)

def sort_data():
    """
    sorts data of findSim() on cosine_sim and save that 
    in window_lambda_dressJean_sorted.pkl file
    """
    print("SORTING_STARTED")
    f = open("data/windowLambda_dressJean.txt")
    data = f.readlines()
    data.sort(key = lambda x: float(x.strip().split("\t")[1]),reverse=True)
    import pickle
    with open("data/window_lambda_dressJean_sorted.pkl",'wb') as o:
        pickle.dump(data,o,pickle.HIGHEST_PROTOCOL)
    print("SORTING_COMPLETE")

def createLists():
    """
    creates 2 separate pkl files that contain list of sentences of individual tasks
    """
    dress = open("data/dress_3_45_train.txt").read().split("\n\n")
    jean  = open("data/jean_3_45_train.txt").read().split("\n\n")
    dress_copy=[]
    for idx,sent in enumerate(dress):
        str=""
        for line in sent.split("\n"):
            line = line.split(" ")
            if(len(line)!=2):
                continue
            if(line[1]!='O'):
                line[1]=line[1]+"_dress"
            line = line[0]+" "+line[1]
            str+=line+"\n"
        dress_copy.append(str)
    
    jean_copy = []
    for idx,sent in enumerate(jean):
        str = ""
        for line in sent.split("\n"):
            line = line.strip().split(" ")
            if(len(line)==2):
                if(line[1]!='O'):
                    line[1]=line[1]+"_jean"
                line = line[0]+" "+line[1]
                str+=line+"\n"
        jean_copy.append(str)
    jean = jean_copy
    dress =dress_copy
    print("#Examples in dress: ",len(dress))
    print("#Examples in jean: ",len(jean))
    import pickle
    with open("data/dress_sent_raw_in_list.pkl",'wb') as o:
        pickle.dump(dress,o,pickle.HIGHEST_PROTOCOL)
    with open("data/jean_sent_raw_in_list.pkl",'wb') as o:
        pickle.dump(jean,o,pickle.HIGHEST_PROTOCOL)

def convertTagsFileToPkl(tagFileName):
    file = open(tagFileName,'r').readlines()
    l=[]
    for tag in file:
        l.append(tag.strip())
    import pickle 
    with open("tags.pkl","wb")as o:
        pickle.dump(l,o,pickle.HIGHEST_PROTOCOL)

        
    
    


# MAIN                
from model.config import Config
config = Config()
createLists()
f =open('data/dress_3_45_train.txt','r')
g =open('data/jean_3_45_train.txt','r')
f1 = open('data/sentences_dress.txt','w')
g1 = open('data/sentences_jean.txt','w')
createWindow("dress",f1)
createWindow("jean",g1)
f.close()
g.close()
f1.close()
g1.close()
model = loadGloveModel(config.filename_glove)
findSim()
sort_data()
convertTagsFileToPkl(config.filename_tags)



