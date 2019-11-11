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
    def __init__(self, filename_lambda_dress, filename_lambda_jean, processing_word=None, processing_tag=None,
                 max_iter=None, kValofKmer=None):
        """
        Args:
            filenames: path to the lambda files created by generateDatasetFilesForCoupling.py
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filename_lambda_dress = filename_lambda_dress
        self.filename_lambda_jean = filename_lambda_jean
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.KValofKmer = kValofKmer
        self.length = None

    def findKmers(self, tag):
        k = self.KValofKmer
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
            word_left_dress       #0
            dress_left_sent_no    #1
            dress_left_tag_pos    #2
            word_left_jean        #3
            jean_left_sent_no     #4
            jean_left_tag_pos     #5
            words_right_dress     #6
            dress_right_sent_no   #7
            dress_right_tag_pos   #8
            words_right_jean     #9
            jean_right_sent_no   #10
            jean_right_tag_pos   #11
        tag:
            tag_left_dress       #0
            tag_left_jean        #1
            tag_right_dress      #2
            tag_right_jean       #3
        '''
        lambda_file_dress = self.filename_lambda_dress
        lambda_file_jean = self.filename_lambda_jean
        import pickle
        with open(lambda_file_dress,"rb") as o:
            lambda_dress = pickle.load(o)
        with open(lambda_file_jean,"rb") as o:
            lambda_jean = pickle.load(o)
        with  open("data/jean_sent_raw_in_list.pkl","rb")as o:
            jean_sent_list = pickle.load(o)
        with open("data/dress_sent_raw_in_list.pkl","rb") as o:
            dress_sent_list =pickle.load(o)
        # with open("data/windowVectors_dress.pkl","rb") as o:
        #     windowVectors_dress =pickle.load(o)
        # with open("data/windowVectors_jean.pkl","rb") as o:
        #     windowVectors_jean =pickle.load(o)
       
        from model.config import Config
        idx=0
        config = Config() 
        filetag =  open(config.filename_tags)
        tags_uncleaned = filetag.readlines()
        tagList = []
        filetag.close()
        for i in tags_uncleaned:
            tagList.append(i.strip())
        coupling_data = zip(lambda_dress,lambda_jean)
        for ix,line in enumerate(coupling_data):     
            if(idx==config.num_of_examples_for_coupling):
                break 
            left = line[0]
            right = line[1]                              #    0             1            2          3       4             5             6          7
            #left and right are list of list : sublist == dsentNo, dcentralWordPos, dcentralTag, dvector, jsentNo, jcentralWordPos, jcentralTag, jvector
            for i in range(config.batch_size):
                dress_left_tag = left[i][2]
                dress_left_tag_id = tagList.index(dress_left_tag)
                jean_left_tag = left[i][6]
                jean_left_tag_id = tagList.index(jean_left_tag)
                dress_right_tag = right[i][2]
                dress_right_tag_id = tagList.index(dress_right_tag)
                jean_right_tag = right[i][6]
                jean_right_tag_id = tagList.index(jean_right_tag)
                dress_left_sentno = left[i][0]
                jean_left_sentno = left[i][4]
                dress_right_sentno = right[i][0]
                jean_right_sentno = right[i][4]
                dress_left_tag_pos = left[i][1]
                jean_left_tag_pos = left[i][5]
                dress_right_tag_pos = right[i][1]
                jean_right_tag_pos = right[i][5]
                dress_left_w_vector = left[i][3]
                jean_left_w_vector = left[i][7]
                dress_right_w_vector = right[i][3]
                jean_right_w_vector = right[i][7]
                kmer_left_dress = self.findKmers(dress_left_tag)
                kmer_left_jean = self.findKmers(jean_left_tag)
                kmer_right_dress = self.findKmers(dress_right_tag)
                kmer_right_jean = self.findKmers(jean_right_tag)
                dress_left_ex, jean_left_ex, dress_right_ex, jean_right_ex = 1,1,1,1
                try:
                    dress_left_ex  = dress_sent_list[dress_left_sentno]
                    jean_left_ex  = dress_sent_list[jean_left_sentno]
                    dress_right_ex  = dress_sent_list[dress_right_sentno]
                    jean_right_ex  = dress_sent_list[jean_right_sentno]
                except:
                    continue
                word_left_dress, word_left_jean = [], []
                word_right_dress, word_right_jean = [], []
                tag_left_dress, tag_left_jean = [],[]
                tag_right_dress, tag_right_jean = [], [] # all of these contain data for one example only
                
                dress_left_ex = dress_left_ex.split("\n")
                if(len(dress_left_ex)<dress_left_tag_pos):
                    continue
                for word_tag_pair in dress_left_ex:
                    word_tag_pair = word_tag_pair.split(" ")
                    if(len(word_tag_pair)==2):
                        word,tag = word_tag_pair[0].strip(),word_tag_pair[-1].strip()
                        if self.processing_word is not None:
                            word = self.processing_word(word)
                        if self.processing_tag is not None:
                            tag = self.processing_tag(tag)
                        word_left_dress += [word]
                        tag_left_dress += [tag]
                
                jean_left_ex = jean_left_ex.split("\n")
                if(len(jean_left_ex)<jean_left_tag_pos):
                    continue
                for word_tag_pair in jean_left_ex:
                    word_tag_pair = word_tag_pair.split(" ")
                    if(len(word_tag_pair)==2):
                        word,tag = word_tag_pair[0].strip(),word_tag_pair[-1].strip()
                        if self.processing_word is not None:
                            word = self.processing_word(word)
                        if self.processing_tag is not None:
                            tag = self.processing_tag(tag)
                        word_left_jean += [word]
                        tag_left_jean += [tag]

                dress_right_ex = dress_right_ex.split("\n")
                if(len(dress_right_ex)<dress_right_tag_pos):
                    continue
                for word_tag_pair in dress_right_ex:
                    word_tag_pair = word_tag_pair.split(" ")
                    if(len(word_tag_pair)==2):
                        word,tag = word_tag_pair[0].strip(),word_tag_pair[-1].strip()
                        if self.processing_word is not None:
                            word = self.processing_word(word)
                        if self.processing_tag is not None:
                            tag = self.processing_tag(tag)
                        word_right_dress += [word]
                        tag_right_dress += [tag]

                jean_right_ex = jean_right_ex.split("\n")
                if(len(jean_right_ex)<jean_right_tag_pos):
                    continue
                for word_tag_pair in jean_right_ex:
                    word_tag_pair = word_tag_pair.split(" ")
                    if(len(word_tag_pair)==2):
                        word,tag = word_tag_pair[0].strip(),word_tag_pair[-1].strip()
                        if self.processing_word is not None:
                            word = self.processing_word(word)
                        if self.processing_tag is not None:
                            tag = self.processing_tag(tag)
                        word_right_jean += [word]
                        tag_right_jean += [tag]

                words = []
                words.append(word_left_dress)       #0
                words.append(dress_left_sentno)     #1
                words.append(dress_left_tag_pos)    #2
                words.append(kmer_left_dress)       #3
                words.append(word_left_jean)        #4
                words.append(jean_left_sentno)      #5
                words.append(jean_left_tag_pos)     #6
                words.append(kmer_left_jean)        #7
                words.append(word_right_dress)      #8
                words.append(dress_right_sentno)    #9
                words.append(dress_right_tag_pos)   #10
                words.append(kmer_right_dress)      #11
                words.append(word_right_jean)       #12
                words.append(jean_right_sentno)     #13
                words.append(jean_right_tag_pos)    #14     
                words.append(kmer_right_jean)       #15
                words.append(dress_left_tag_id)     #16
                words.append(jean_left_tag_id)      #17
                words.append(dress_right_tag_id)    #18
                words.append(jean_right_tag_id)     #19
                words.append(dress_left_w_vector)   #20
                words.append(jean_left_w_vector)    #21
                words.append(dress_left_w_vector)   #22
                words.append(jean_left_w_vector)    #23
                tags=[]
                tags.append(tag_left_dress)
                tags.append(tag_left_jean)
                tags.append(tag_right_dress)
                tags.append(tag_right_jean)
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
    X = [       
            x_batch             0
            y_batch             1

            x_left_batch_dress  2
            y_left_batch_dress  3 
            dress_left_tag_id   4
            window_left_dress   5 
            kmer_left_dress     6 

            x_left_batch_jean   7
            y_left_batch_jean   8
            jean_left_tag_id    9
            window_left_jean    10
            kmer_left_jean      11

            x_right_batch_dress 12
            y_right_batch_dress 13
            dress_right_tag_id  14
            window_right_dress  15
            kmer_right_dress    16

            x_right_batch_jean  17
            y_right_batch_jean  18
            jean_right_tag_id   19
            window_right_jean   20
            kmer_right_jean     21
            wordPos_left_dress  22
            wordPos_left_jean   23
            wordPos_right_dress 24
            wordPos_right_jean  25
            
          ]

    """



    x_batch, y_batch, word= [], [], []
    x_left_batch_dress, y_left_batch_dress, word_left_dress = [], [], []
    x_left_batch_jean, y_left_batch_jean, word_left_jean= [], [], []
    x_right_batch_dress, y_right_batch_dress, word_right_dress = [], [], []
    x_right_batch_jean, y_right_batch_jean, word_right_jean= [], [], []
    dress_left_tag_id, jean_left_tag_id, dress_right_tag_id, jean_right_tag_id = [], [], [], []
    kmer_right_jean, kmer_right_dress, kmer_left_dress, kmer_left_jean = [], [], [], []
    window_right_jean, window_right_dress, window_left_dress, window_left_jean = [], [], [], []
    wordPos_left_dress, wordPos_left_jean, wordPos_right_dress, wordPos_right_jean = [], [], [], []


    if coupling_data is not None:
        data = zip(data,coupling_data)

    X = []
    if(coupling_data is None):
    # for crf loss data
        for (x,y) in data:
            if len(x_batch) == minibatch_size:
                #print(x_batch)
                X.append(x_batch)
                X.append(y_batch)
                X.append(word)

                yield X
                X = []
                x_batch, y_batch, word = [], [], []

            if type(x[0]) == tuple:
                word += [x]
                x = zip(*x)
                #print("Inside if")
            x_batch += [x]
            y_batch += [y]

        if len(x_batch) != 0:
            X.append(x_batch)
            X.append(y_batch)
            X.append(word)
            yield X
    else:
        for data_tuple in data:
            (x,y) = data_tuple[0]
            (x_coup,y_coup) = data_tuple[1]
            x_left_dress = x_coup[0]
            x_left_jean =  x_coup[4]
            y_left_dress = y_coup[0]
            y_left_jean =  y_coup[1]
            x_right_dress =x_coup[8]
            x_right_jean = x_coup[12]
            y_right_dress =y_coup[2]
            y_right_jean = y_coup[3]
            if len(x_batch) == minibatch_size:
                X.append(x_batch)
                X.append(y_batch)

                X.append(x_left_batch_dress)
                X.append(y_left_batch_dress)
                X.append(dress_left_tag_id)
                X.append(window_left_dress)
                X.append(kmer_left_dress)

                X.append(x_left_batch_jean)
                X.append(y_left_batch_jean)
                X.append(jean_left_tag_id)
                X.append(window_left_jean)
                X.append(kmer_left_jean)

                X.append(x_right_batch_dress)
                X.append(y_right_batch_dress)
                X.append(dress_right_tag_id)
                X.append(window_right_dress)
                X.append(kmer_right_dress)

                X.append(x_right_batch_jean)
                X.append(y_right_batch_jean)
                X.append(jean_right_tag_id)
                X.append(window_right_jean)
                X.append(kmer_right_jean)
                X.append(wordPos_left_dress)
                X.append(wordPos_left_jean)
                X.append(wordPos_right_dress)
                X.append(wordPos_right_jean)
                # print(wordPos_right_jean)

                yield X
                X = []
                x_batch, y_batch, word= [], [], []
                x_left_batch_dress, y_left_batch_dress, word_left_dress = [], [], []
                x_left_batch_jean, y_left_batch_jean, word_left_jean= [], [], []
                x_right_batch_dress, y_right_batch_dress, word_right_dress = [], [], []
                x_right_batch_jean, y_right_batch_jean, word_right_jean= [], [], []
                dress_left_tag_id, jean_left_tag_id, dress_right_tag_id, jean_right_tag_id = [], [], [], []
                kmer_right_jean, kmer_right_dress, kmer_left_dress, kmer_left_jean = [], [], [], []
                window_right_jean, window_right_dress, window_left_dress, window_left_jean = [], [], [], []
                wordPos_left_dress, wordPos_left_jean, wordPos_right_dress, wordPos_right_jean = [], [], [], []

            if type(x[0]) == tuple:
                word += [x]
                x = zip(*x)
            x_batch += [x]
            y_batch += [y]

            if type(x_left_dress[0]) == tuple:
                word_left_dress += [x_left_dress]
                x_left_dress = zip(*x_left_dress)
            x_left_batch_dress += [x_left_dress]
            y_left_batch_dress += [y_left_dress]

            if type(x_left_jean[0]) == tuple:
                word_left_jean += [x_left_jean]
                x_left_jean = zip(*x_left_jean)
            x_left_batch_jean += [x_left_jean]
            y_left_batch_jean += [y_left_jean]

            if type(x_right_dress[0]) == tuple:
                word_right_dress += [x_right_dress]
                x_right_dress = zip(*x_right_dress)
            x_right_batch_dress += [x_right_dress]
            y_right_batch_dress += [y_right_dress]

            if type(x_right_jean[0]) == tuple:
                word_right_jean += [x_right_jean]
                x_right_jean = zip(*x_right_jean)
            x_right_batch_jean += [x_right_jean]
            y_right_batch_jean += [y_right_jean]
            
            dress_left_tag_id.append(x_coup[16])
            jean_left_tag_id.append(x_coup[17])
            dress_right_tag_id.append(x_coup[18])  
            jean_right_tag_id.append(x_coup[19])

            window_left_dress.append(x_coup[20])  
            window_left_jean.append(x_coup[21])  
            window_right_dress.append(x_coup[22])                   
            window_right_jean.append(x_coup[23]) 

            kmer_left_dress.append(x_coup[3])  
            kmer_left_jean.append(x_coup[7])
            kmer_right_dress.append(x_coup[11])
            kmer_right_jean.append(x_coup[15])     
            wordPos_left_dress.append(x_coup[2])  
            wordPos_left_jean.append(x_coup[6])  
            wordPos_right_dress.append(x_coup[10])  
            wordPos_right_jean.append(x_coup[14]) 
        #skipping last batch which is of size < config.batch_size
        # if(len(x_batch) != 0):
        #     X=[]
        #     X.append(x_batch)
        #     X.append(y_batch)

        #     X.append(x_left_batch_dress)
        #     X.append(y_left_batch_dress)
        #     X.append(dress_left_tag_id)
        #     X.append(window_left_dress)
        #     X.append(kmer_left_dress)

        #     X.append(x_left_batch_jean)
        #     X.append(y_left_batch_jean)
        #     X.append(jean_left_tag_id)
        #     X.append(window_left_jean)
        #     X.append(kmer_left_jean)

        #     X.append(x_right_batch_dress)
        #     X.append(y_right_batch_dress)
        #     X.append(dress_right_tag_id)
        #     X.append(window_right_dress)
        #     X.append(kmer_right_dress)

        #     X.append(x_right_batch_jean)
        #     X.append(y_right_batch_jean)
        #     X.append(jean_right_tag_id)
        #     X.append(window_right_jean)
        #     X.append(kmer_right_jean)
        #     X.append(wordPos_left_dress)
        #     X.append(wordPos_left_jean)
        #     X.append(wordPos_right_dress)
        #     X.append(wordPos_right_jean)
        #     yield X


            







  


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
