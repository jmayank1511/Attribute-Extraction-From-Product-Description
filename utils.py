
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import pickle
max_possible_batch_size = 40
def breakDict(data, name):
    """
    This function will break dict into sublists of size : max_possible_batch_size each
    """

    batches = []
    for i in data:
        values = data[i]
        for i in range(0,len(values)-1,max_possible_batch_size):
            if(i+max_possible_batch_size<=len(values)):
                batches.append(values[i:i+max_possible_batch_size])
    batches_shuffled = random.sample(batches, len(batches)) 
    print(name,len(batches_shuffled))
    with open(name,"wb")as o:
        pickle.dump(batches_shuffled, o, pickle.HIGHEST_PROTOCOL)

def filterDicts(data,name):
    """
    This function removes all the keys with values of len < max_possible_batch__size
    """

    l = []
    for i in data:
        if(len(data[i])<max_possible_batch_size):
            l.append(i)
    for i in l:
        del data[i]
    breakDict(data,name)




def createWindow(vertical,filename,dress):
    """
    creates a file: filename that contains data in form
    wordi-1 tagi-1 wordi tagi wordi+1 tagi+1 centralWord_id sentID
    this is the intermediate file to be used by findSim() function
    """
    for sentno,line in enumerate(dress):
        line = line.strip().split("\n")
        for i in range(0,len(line)-2):
            toprint = line[i]+" " +line[i+1]+" "+line[i+2]+" "+str((i+1))+" "+str(sentno)
            filename.write(toprint+"\n")

def loadGloveModel(gloveFile):
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

def createLambdaLists(cosine_8,dress_list,jean_list):
    dress_lambda = {}
    jean_lambda = {}
    import copy
    print(len(cosine_8[0]))
    for idx,i in enumerate(cosine_8[0]):
        j = cosine_8[1][idx]
        x = dress_list[i]
        x.extend(jean_list[j])
        if(i in dress_lambda):
            dress_lambda[i].append(x)
        else:
            dress_lambda[i] = [x]
        if(j in jean_lambda):
            jean_lambda[j].append(x)
        else:
            jean_lambda[j] = [x]
    print("filtering...")
    filterDicts(dress_lambda,"data/dress_lambda.pkl")
    filterDicts(jean_lambda,"data/jean_lambda.pkl")
    




def findSim():
    """
    finds cosine similarity of word vectors of window size 3 
    and saves output in windowLambda_dressJean.txt
    this data needs to be sorted on cosine sim
    that task is done by sort_data()
    """
    f1 = open('data/sentences_dress.txt','r')
    g1 = open('data/sentences_jean.txt','r')
    zeros = np.zeros(300)
    jeans = g1.readlines()
    jean_window_vectors = [] # key: (sentno,centralWordPos,tag); Value = wordvector of window 3
    jean_list = []
    dress_window_vectors = []
    dress_list = []
    for line in jeans:  # line is :  word tag centerword centertag word tag centerwordpos sentno
        line = line.strip().split(" ")
        centralTag =  line[3].strip()
        if(centralTag =='O' or len(line)!=8):
            continue
        sentNo = int(line[7].strip())
        centralWordPos = int(line[6].strip())
        if(js[sentNo]<centralWordPos):
            print("ouch")
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
        jean_list.append([sentNo,centralWordPos,centralTag,vector])        
        jean_window_vectors.append(vector)
    f1.close()
    g1.close()
    zeros = np.zeros(300)
    f1 = open('data/sentences_dress.txt','r')
    dress = f1.readlines()
    g1 = open('data/sentences_jean.txt','r')
    jean = g1.readlines()
    for idx,line in enumerate(dress):
        line = line.strip().split(" ") 
        centralTag =  line[3].strip()
        if(centralTag =='O' or len(line)!=8):
            continue
        sentNo = int(line[7].strip())
        centralWordPos = int(line[6].strip())
        if(ds[sentNo]<centralWordPos):
            print("ouch")
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
        dress_list.append([sentNo,centralWordPos,centralTag,dress_vector])        
        dress_window_vectors.append(dress_vector)
    dress_window_vectors_np = np.array(dress_window_vectors)
    jean_window_vectors_np = np.array(jean_window_vectors)
    cosine_sim = cosine_similarity(dress_window_vectors_np,jean_window_vectors_np)
    cosine_8 = np.nonzero(cosine_sim>0.69)  #creates a tuple ([x cordinate],[y cordinate])
    createLambdaLists( cosine_8,dress_list,jean_list)

        





def createLists():
    """
    creates 2 separate pkl files that contain list of sentences of individual tasks
    """
    dress = open("data/dress_3_45_train.txt").read().split("\n \n")
    jean  = open("data/jean_3_45_train.txt").read().split("\n \n")
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
    return dress,jean

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
dress_raw_list,jean_raw_list = createLists()
ds = []
js = []
for i in dress_raw_list:
    ds.append(len(i.split("\n"))-1)
for i in jean_raw_list:
    js.append(len(i.split("\n"))-1)
print(len(ds),len(js))
f1 = open('data/sentences_dress.txt','w')
g1 = open('data/sentences_jean.txt','w')
createWindow("dress",f1,dress_raw_list)
createWindow("jean",g1,jean_raw_list) 
f1.close()
g1.close()
model = loadGloveModel(config.filename_glove)
findSim()
convertTagsFileToPkl(config.filename_tags)
