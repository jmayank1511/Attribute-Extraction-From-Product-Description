import numpy as np
import os


# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"


# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)

        super(MyIOError, self).__init__(message)


class FKDataset(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is applied

    Example:
        ```python
        data = FKDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """
    def __init__(self, filename, processing_word=None, processing_tag=None,
                 max_iter=None):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None


    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            f = f.readlines()
            words, tags = [], []
            for idx,line in enumerate(f):
                line = line.strip()
                #print(line)
                if (len(line) == 0 or line.startswith("-DOCSTART-") or idx==len(f)-1):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split(' ')
                    word, tag = ls[0],ls[-1]
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += [word]
                    tags += [tag]
            #print(str(words))
            #print(str(tags))


    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

class DatasetForCouplingLoss(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is applied

    """
    def __init__(self, filename, processing_word=None, processing_tag=None,
                 max_iter=None, kValofKmer=None):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None
        self.kValofKmer = kValofKmer

    def findKmers(self, tag, k):
        if(tag=='O'):
            return ['O']
        kmers =[]
        for i in range(len(tag)-k+1):
            kmers.append(tag[i:i+k])
        return kmers


    def __iter__(self):
        '''
        description of data sent by this method:
        returns a tuple : word,tag
        word:
            a list of 4 lines:
                line1: words of dress_example
                line2: dress_sent_no
                line3: dress_central_tag_id
                line3: words of jean_example
                line4: jean_sent_no
                line6: jean_central_tag_id
                line7: dress_central_tag_pos
                line8: jean_central_tag_pos
                line9: kmers of tag_dress | for eg: if tag is 'abc' and k for k-mer is 2 then return ['ab','bc']
                line10: kmers of tag_jean
        tag:
            a list of 2 lines:
                line1: tag list of dress_example
                line2: tag list of jean_example
        '''
        lambda_file = "data/"+self.filename
        import pickle
        with open(lambda_file,"rb") as o:
            lamdas = pickle.load(o)
        with  open("data/jean_sent_raw_in_list.pkl","rb")as o:
            jean_sent_list = pickle.load(o)
        with open("data/dress_sent_raw_in_list.pkl","rb") as o:
            dress_sent_list =pickle.load(o)
        with open("data/windowVectors_dress.pkl","rb") as o:
            windowVectors_dress =pickle.load(o)
        with open("data/windowVectors_jean.pkl","rb") as o:
            windowVectors_jean =pickle.load(o)
       
        from model.config import Config
        import pickle
        with open("tags.pkl","rb")as o:
            tagsList = pickle.load(o)
        idx=0
        config = Config() 
        for ix,line in enumerate(lamdas):     
            if(idx==config.num_of_examples_for_coupling):
                break 
            line = line.strip().split(" ")
            dress_sent_no = line[7]
            dress_tag_id = tagsList.index(line[3])
            dress_tag_pos = int(line[6])
            jean_sent_no = line[15].split("\t")[0].strip()
            jean_tag_id = tagsList.index(line[11])
            jean_tag_pos = int(line[14])
            if(int(dress_sent_no),int(dress_tag_pos))not in windowVectors_dress:
                continue
            if(int(jean_sent_no),int(jean_tag_pos))not in windowVectors_jean:
                continue
            idx+=1
            dress_tag = line[3]
            jean_tag = line[11]
            kmer_dress = self.findKmers(dress_tag,self.kValofKmer)
            kmer_jean = self.findKmers(jean_tag,self.kValofKmer)
            dress_ex = dress_sent_list[int(dress_sent_no)]
            jean_ex = jean_sent_list[int(jean_sent_no)]
            word_dress,word_jean,tag_dress,tag_jean = [],[],[],[] # contain data for one example only
            dress_ex = dress_ex.split("\n")
            for word_tag_pair in dress_ex:
                word_tag_pair = word_tag_pair.split(" ")
                if(len(word_tag_pair)==2):
                    word,tag = word_tag_pair[0].strip(),word_tag_pair[-1].strip()
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    word_dress += [word]
                    tag_dress += [tag]
            jean_ex = jean_ex.split("\n")
            for word_tag_pair in jean_ex: 
                word_tag_pair = word_tag_pair.split(" ") 
                if(len(word_tag_pair)==2): 
                    word,tag = word_tag_pair[0],word_tag_pair[-1]
                    #print(word,tag)
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    word_jean += [word]
                    tag_jean += [tag]

            #generating result in required format
            words = []
            words.append(word_dress)
            words.append(dress_sent_no)
            words.append(dress_tag_id)
            words.append(word_jean)
            words.append(jean_sent_no)
            words.append(jean_tag_id)
            words.append(dress_tag_pos)
            words.append(jean_tag_pos)
            words.append(kmer_dress)
            words.append(kmer_jean)
            tags=[]
            tags.append(tag_dress)
            tags.append(tag_jean)
            yield words,tags
            


        

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length





def get_vocabs(datasets):
    """Build vocabulary from an iterable of datasets objects

    Args:
        datasets: a list of dataset objects

    Returns:
        a set of all the words in the dataset

    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags


def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_char = set()
    for words, _ in dataset:
        for word in words:
            vocab_char.update(word)

    return vocab_char


def get_glove_vocab(filename):
    """Load vocab from file

    Args:
        filename: path to the glove vectors

    Returns:
        vocab: set() of strings
    """
    print("Building vocab...")
    vocab = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
        f.close()
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx
        f.close()

    except IOError:
        raise MyIOError(filename)
    return d


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings

    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)


def get_processing_word(vocab_words=None, vocab_chars=None,
                    lowercase=False, chars=False, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct"+word)

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)

    return sequence_padded, sequence_length



def minibatches(data, minibatch_size,coupling_data,minibatch_size_coupling):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples for data and coupling data
    X = [x_batch,
         y_batch,words,
         x_dress, y_dress,
         words_dress,
         sent_no_dress,
         id_central_tag_dress,
         x_jean, y_jean,
         words_jean,
         sent_no_jean,
         id_central_tag_jean,
         tag_pos_dress,
         tag_pos_jean,
         windowvector_dress
         windowvector_jean
         kmer_dress
         kmer_jean
          ]

    """

    #print(coupling_data)
    import pickle
    with open("data/windowVectors_dress.pkl","rb") as o:
        dress_dict = pickle.load(o)
    with open("data/windowVectors_jean.pkl","rb") as p:
        jean_dict = pickle.load(p)

    x_batch, y_batch, word= [], [], []
    x_batch_dress, y_batch_dress, word_dress,x_batch_jean, y_batch_jean, word_jean= [], [], [], [], [], []
    sent_no_dress,sent_no_jean, id_central_tag_dress, id_central_tag_jean = [],[],[],[]
    tag_pos_dress,tag_pos_jean = [],[]
    window_dress, window_jean= [],[]
    kmer_dress,kmer_jean = [],[]
    X = []
    # for crf loss data
    for (x, y) in data:
        #print(str(x[0]))
        if len(x_batch) == minibatch_size:
            #print(x_batch)
            X.append([])
            X[-1].append(x_batch)
            X[-1].append(y_batch)
            X[-1].append(word)

            #yield x_batch, y_batch, word
            x_batch, y_batch, word = [], [], []

        if type(x[0]) == tuple:
            word += [x]
            x = zip(*x)
            #print("Inside if")
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        X.append([])
        X[-1].append(x_batch)
        X[-1].append(y_batch)
        X[-1].append(word)


    #for coupling loss data
    if(coupling_data is not None): # coupling data is set to None for run_evaluate function.
        i=0       
        for (x, y) in coupling_data:
            x_dress = x[0]
            x_jean = x[3]
            y_dress=y[0]
            y_jean = y[1]
            if i==len(X)-1:
                break


            if len(x_batch_dress) == minibatch_size_coupling:
                #print(x_batch)               
                X[i].append(x_batch_dress)
                X[i].append(y_batch_dress)
                X[i].append(word_dress)
                X[i].append(sent_no_dress)
                X[i].append(id_central_tag_dress)

                X[i].append(x_batch_jean)
                X[i].append(y_batch_jean)
                X[i].append(word_jean)
                X[i].append(sent_no_jean)
                X[i].append(id_central_tag_jean)
                X[i].append(tag_pos_dress)
                X[i].append(tag_pos_jean)
                X[i].append(window_dress)
                X[i].append(window_jean)   
                X[i].append(kmer_dress)
                X[i].append(kmer_jean)            

                i+=1
                #yield x_batch, y_batch, word
                x_batch_dress, y_batch_dress, word_dress,x_batch_jean, y_batch_jean, word_jean= [], [], [], [], [], []
                sent_no_dress,sent_no_jean, id_central_tag_dress, id_central_tag_jean = [],[],[],[]
                tag_pos_dress,tag_pos_jean =[],[]
                window_dress, window_jean = [],[]
                kmer_dress,kmer_jean = [],[]


            if type(x_dress[0]) == tuple:
                word_dress += [x_dress]
                x_dress = zip(*x_dress)
                #print("Inside if")
            x_batch_dress += [x_dress]
            y_batch_dress += [y_dress]

            if type(x_jean[0]) == tuple:
                word_jean += [x_jean]
                x_jean = zip(*x_jean)
                #print("Inside if")
            x_batch_jean += [x_jean]
            y_batch_jean += [y_jean]  
            sent_no_dress.append(int(x[1]))
            sent_no_jean.append(int(x[4]))
            id_central_tag_dress.append(int(x[2]))
            id_central_tag_jean.append(int(x[5]))
            tag_pos_dress.append(x[6])
            tag_pos_jean.append(x[7])
            try:
                window_dress.append(dress_dict[(int(x[1]),x[6])])
            except:
                window_dress.append([0]*900)
            try:
                window_jean.append(jean_dict[(int(x[4]),x[7])]) 
            except:
                window_jean.append([0]*900)
            kmer_dress.append(x[8])
            kmer_jean.append(x[9])
            # print("1",x[8])

        if len(x_batch_dress) != 0:
            X[-1].append(x_batch_dress)
            X[-1].append(y_batch_dress)
            X[-1].append(word_dress)
            X[-1].append(sent_no_dress)
            X[-1].append(id_central_tag_dress)

            X[-1].append(x_batch_jean)
            X[-1].append(y_batch_jean)
            X[-1].append(word_jean)
            X[-1].append(sent_no_jean)
            X[-1].append(id_central_tag_jean)
            X[-1].append(tag_pos_dress)
            X[-1].append(tag_pos_jean)
            X[-1].append(window_dress)
            X[-1].append(window_jean)
            X[-1].append(kmer_dress)
            X[-1].append(kmer_jean) 

    return X


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks
