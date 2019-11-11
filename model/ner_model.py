import numpy as np
import os
import tensorflow as tf


from .data_utils import minibatches, pad_sequences, get_chunks
from .general_utils import Progbar
from .base_model import BaseModel


class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}
        self.idx_to_word = {idx: word for word, idx in self.config.vocab_words.items()}
        self.task =0 
        self.max_kmer_list_size = 45# this data is used by convertTagKmerToEmbedding
        self.minibatch_data = None


    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids")
        self.word_ids_dress_left_coup = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids_dress_left_coup")
        self.word_ids_jean_left_coup = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids_jean_left_coup")
        self.word_ids_dress_right_coup = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids_dress_right_coup")
        self.word_ids_jean_right_coup = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids_jean_right_coup")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")
        self.sequence_lengths_dress_left_coup = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths_dress_left_coup")
        self.sequence_lengths_jean_left_coup = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths_jean_left_coup")
        self.sequence_lengths_dress_right_coup = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths_dress_right_coup")
        self.sequence_lengths_jean_right_coup = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths_jean_right_coup")

        self.dress_left_tag_id = tf.placeholder(tf.int32,shape=[None],name="dress_left_tag_id")
        self.jean_left_tag_id = tf.placeholder(tf.int32,shape=[None],name="jean_left_tag_id")
        self.dress_right_tag_id = tf.placeholder(tf.int32,shape=[None],name="dress_right_tag_id")
        self.jean_right_tag_id = tf.placeholder(tf.int32,shape=[None],name="jean_right_tag_id")
        self.window_left_dress  = tf.placeholder(tf.float32,shape=[None,900],name="window_left_dress")
        self.window_left_jean  = tf.placeholder(tf.float32,shape=[None,900],name="window_left_jean")
        self.window_right_dress  = tf.placeholder(tf.float32,shape=[None,900],name="window_right_dress")
        self.window_right_jean  = tf.placeholder(tf.float32,shape=[None,900],name="window_right_jean")
        self.dress_left_kmer_indices = tf.placeholder(tf.int32,shape=[None, self.max_kmer_list_size],name="dress_left_kmer_indices")   #[None,self.config.label_embedding_size]
        self.jean_left_kmer_indices = tf.placeholder(tf.int32,shape=[None, self.max_kmer_list_size],name="jean_left_kmer_indices")
        self.dress_right_kmer_indices = tf.placeholder(tf.int32,shape=[None, self.max_kmer_list_size],name="dress_right_kmer_indices")   #[None,self.config.label_embedding_size]
        self.jean_right_kmer_indices = tf.placeholder(tf.int32,shape=[None, self.max_kmer_list_size],name="jean_right_kmer_indices")
        self.wordPos_left_dress = tf.placeholder(tf.int32,shape=[None],name="wordPos_left_dress")
        self.wordPos_left_jean = tf.placeholder(tf.int32,shape=[None],name="wordPos_left_jean")
        self.wordPos_right_dress = tf.placeholder(tf.int32,shape=[None],name="wordPos_right_dress")
        self.wordPos_right_jean = tf.placeholder(tf.int32,shape=[None],name="wordPos_right_jean")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="char_ids")
        self.char_ids_dress_left_coup = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="char_ids_dress_left_coup")
        self.char_ids_jean_left_coup = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="char_ids_jean_left_coup")
        self.char_ids_dress_right_coup = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="char_ids_dress_right_coup")
        self.char_ids_jean_right_coup = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="char_ids_jean_right_coup")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths")
        self.word_lengths_dress_left_coup = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths_dress_left_coup")
        self.word_lengths_jean_left_coup = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths_jean_left_coup")
        self.word_lengths_dress_right_coup = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths_dress_right_coup")
        self.word_lengths_jean_right_coup = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths_jean_right_coup")

        # shape = (batch size, max length of sentence in batch)
        self.y_batch = tf.placeholder(tf.int32, shape=[None, None],
                        name="y_batch")
        self.y_left_batch_dress = tf.placeholder(tf.int32, shape=[None, None],
                        name="y_left_batch_dress")
        self.y_left_batch_jean = tf.placeholder(tf.int32, shape=[None, None],
                        name="y_left_batch_jean")
        self.y_right_batch_dress = tf.placeholder(tf.int32, shape=[None, None],
                        name="y_right_batch_dress")
        self.y_right_batch_jean = tf.placeholder(tf.int32, shape=[None, None],
                        name="y_right_batch_jean")                        
        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")
    def convertTagKmerToEmbedding(self,kmer_batch) :
        """
        Each example is a list of kmer it converts it into list of their respective indices
        eg [br, ra, an, nd,] --> [3, 2, 4, 5]
        """
        list1 = []
        for example in kmer_batch:   # loop for no of examples in batch
            list2 = []
            for kmer in example:
                list2.append(self.label_dict[kmer])
            for i in range(self.max_kmer_list_size-len(list2)):
                list2.append(-1)
            list1.append(list2)
        return list1



    def get_feed_dict(self, x_batch, y_batch = None,           
        x_left_batch_dress = None, x_left_batch_jean= None, x_right_batch_dress= None, x_right_batch_jean= None, 
        y_left_batch_dress = None, y_left_batch_jean = None, y_right_batch_dress = None,y_right_batch_jean = None,
        dress_left_tag_id = [], jean_left_tag_id = [], dress_right_tag_id = [], jean_right_tag_id = [],
        window_left_dress = [], window_left_jean = [], window_right_dress = [], window_right_jean = [],
        dress_left_kmer_indices = [], jean_left_kmer_indices = [], dress_right_kmer_indices = [], jean_right_kmer_indices = [],        
        wordPos_left_dress =[], wordPos_left_jean =[], wordPos_right_dress =[], wordPos_right_jean =[],
        lr = None,
        dropout= None):

        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        

        # perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = zip(*x_batch)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(x_batch, 0)
        if x_left_batch_dress is not None  :     
            if self.config.use_chars  :
                char_ids_dress_left_coup, word_ids_dress_left_coup = zip(*x_left_batch_dress)
                char_ids_jean_left_coup, word_ids_jean_left_coup = zip(*x_left_batch_jean)
                char_ids_dress_right_coup, word_ids_dress_right_coup = zip(*x_right_batch_dress)
                char_ids_jean_right_coup, word_ids_jean_right_coup = zip(*x_right_batch_jean)

                word_ids_dress_left_coup, sequence_lengths_dress_left_coup = pad_sequences(word_ids_dress_left_coup, 0)
                word_ids_jean_left_coup, sequence_lengths_jean_left_coup = pad_sequences(word_ids_jean_left_coup, 0)
                word_ids_dress_right_coup, sequence_lengths_dress_right_coup = pad_sequences(word_ids_dress_right_coup, 0)
                word_ids_jean_right_coup, sequence_lengths_jean_right_coup = pad_sequences(word_ids_jean_right_coup, 0)

                char_ids_dress_left_coup, word_lengths_dress_left_coup = pad_sequences(char_ids_dress_left_coup, pad_tok=0,nlevels=2)
                char_ids_jean_left_coup, word_lengths_jean_left_coup = pad_sequences(char_ids_jean_left_coup, pad_tok=0,nlevels=2)
                char_ids_dress_right_coup, word_lengths_dress_right_coup = pad_sequences(char_ids_dress_right_coup, pad_tok=0,nlevels=2)
                char_ids_jean_right_coup, word_lengths_jean_right_coup = pad_sequences(char_ids_jean_right_coup, pad_tok=0,nlevels=2)
            else:
                word_ids_dress_left_coup, sequence_lengths_dress_left_coup = pad_sequences(x_left_batch_dress, 0)
                word_ids_jean_left_coup, sequence_lengths_jean_left_coup = pad_sequences(x_left_batch_jean, 0)
                word_ids_dress_right_coup, sequence_lengths_dress_right_coup = pad_sequences(x_right_batch_dress, 0)
                word_ids_jean_right_coup, sequence_lengths_jean_right_coup = pad_sequences(x_right_batch_jean, 0)
        # build feed dictionary
        if(x_left_batch_dress is not None):
            feed = {
                self.word_ids: word_ids,
                self.sequence_lengths: sequence_lengths,
                self.word_ids_dress_left_coup: word_ids_dress_left_coup,
                self.word_ids_jean_left_coup: word_ids_jean_left_coup,
                self.word_ids_dress_right_coup: word_ids_dress_right_coup,
                self.word_ids_jean_right_coup: word_ids_jean_right_coup,
                self.sequence_lengths_dress_left_coup: sequence_lengths_dress_left_coup,
                self.sequence_lengths_jean_left_coup: sequence_lengths_jean_left_coup,
                self.sequence_lengths_dress_right_coup: sequence_lengths_dress_right_coup,
                self.sequence_lengths_jean_right_coup: sequence_lengths_jean_right_coup,
                self.dress_left_tag_id : dress_left_tag_id,
                self.jean_left_tag_id : jean_left_tag_id,
                self.dress_right_tag_id : dress_right_tag_id,
                self.jean_right_tag_id : jean_right_tag_id,
                self.window_left_dress : window_left_dress,
                self.window_left_jean : window_left_jean,
                self.window_right_dress : window_right_dress,
                self.window_right_jean : window_right_jean,
                self.dress_left_kmer_indices : dress_left_kmer_indices,
                self.jean_left_kmer_indices : jean_left_kmer_indices,
                self.dress_right_kmer_indices : dress_right_kmer_indices,
                self.jean_right_kmer_indices : jean_right_kmer_indices,
                self.wordPos_left_dress : wordPos_left_dress,
                self.wordPos_left_jean : wordPos_left_jean,
                self.wordPos_right_dress : wordPos_right_dress,
                self.wordPos_right_jean : wordPos_right_jean
            }
        else:
            feed = {
                self.word_ids: word_ids,
                self.sequence_lengths: sequence_lengths,
                self.word_ids_dress_left_coup: word_ids,
                self.word_ids_jean_left_coup: word_ids,
                self.word_ids_dress_right_coup: word_ids,
                self.word_ids_jean_right_coup: word_ids,
                self.sequence_lengths_dress_left_coup: sequence_lengths,
                self.sequence_lengths_jean_left_coup: sequence_lengths,
                self.sequence_lengths_dress_right_coup: sequence_lengths,
                self.sequence_lengths_jean_right_coup: sequence_lengths,
                self.dress_left_tag_id : [1],
                self.jean_left_tag_id : [1],
                self.dress_right_tag_id : [1],
                self.jean_right_tag_id : [1],
                self.window_left_dress : [[-1]*900],
                self.window_left_jean : [[-1]*900],
                self.window_right_dress : [[-1]*900],
                self.window_right_jean : [[-1]*900],
                self.dress_left_kmer_indices : [[-1]*self.config.max_kmer_list_size],
                self.jean_left_kmer_indices : [[-1]*self.config.max_kmer_list_size],
                self.dress_right_kmer_indices : [[-1]*self.config.max_kmer_list_size],
                self.jean_right_kmer_indices : [[-1]*self.config.max_kmer_list_size],
                self.wordPos_left_dress : [1],
                self.wordPos_left_jean : [1],
                self.wordPos_right_dress : [1],
                self.wordPos_right_jean : [1]
            }
        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths
        if self.config.use_chars and x_left_batch_dress is not None:
            feed[self.char_ids_dress_left_coup] = char_ids_dress_left_coup
            feed[self.word_lengths_dress_left_coup] = word_lengths_dress_left_coup
            feed[self.char_ids_jean_left_coup] = char_ids_jean_left_coup
            feed[self.word_lengths_jean_left_coup] = word_lengths_jean_left_coup
            feed[self.char_ids_dress_right_coup] = char_ids_dress_right_coup
            feed[self.word_lengths_dress_right_coup] = word_lengths_dress_right_coup
            feed[self.char_ids_jean_right_coup] = char_ids_jean_right_coup
            feed[self.word_lengths_jean_right_coup] = word_lengths_jean_right_coup
        else:
            feed[self.char_ids_dress_left_coup] = char_ids
            feed[self.word_lengths_dress_left_coup] = word_lengths
            feed[self.char_ids_jean_left_coup] = char_ids
            feed[self.word_lengths_jean_left_coup] = word_lengths
            feed[self.char_ids_dress_right_coup] = char_ids
            feed[self.word_lengths_dress_right_coup] = word_lengths
            feed[self.char_ids_jean_right_coup] = char_ids
            feed[self.word_lengths_jean_right_coup] = word_lengths

        # print("")

        if y_batch is not None:
            y_batch, _ = pad_sequences(y_batch, 0)
            feed[self.y_batch] = y_batch
        if y_left_batch_dress is not None:
            y_left_batch_dress, _ = pad_sequences(y_left_batch_dress, 0)
            feed[self.y_left_batch_dress] = y_left_batch_dress
            y_left_batch_jean, _ = pad_sequences(y_left_batch_jean, 0)
            feed[self.y_left_batch_jean] = y_left_batch_jean
            y_right_batch_dress, _ = pad_sequences(y_right_batch_dress, 0)
            feed[self.y_right_batch_dress] = y_right_batch_dress
            y_right_batch_jean, _ = pad_sequences(y_right_batch_jean, 0)
            feed[self.y_right_batch_jean] = y_right_batch_jean
        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths


    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                        shape=[s[0], s[1], 2*self.config.hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)

        # for coupling loss!!
        
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings_dress_left_coup = tf.get_variable(
                        name="_word_embeddings_dress_left_coup",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
                _word_embeddings_jean_left_coup = tf.get_variable(
                        name="_word_embeddings_jean_left_coup",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
                _word_embeddings_dress_left_coup = tf.get_variable(
                        name="_word_embeddings_dress_left_coup",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
                _word_embeddings_jean_left_coup = tf.get_variable(
                        name="_word_embeddings_jean_left_coup",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings_dress_left_coup = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings_dress_left_coup",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)
                _word_embeddings_jean_left_coup = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings_jean_left_coup",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)
                _word_embeddings_dress_right_coup = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings_dress_right_coup",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)
                _word_embeddings_jean_right_coup = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings_jean_right_coup",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings_dress_left_coup = tf.nn.embedding_lookup(_word_embeddings_dress_left_coup,
                    self.word_ids_dress_left_coup, name="word_embeddings_dress_left_coup")
            word_embeddings_jean_left_coup = tf.nn.embedding_lookup(_word_embeddings_jean_left_coup,
                    self.word_ids_jean_left_coup, name="word_embeddings_jean_left_coup")
            word_embeddings_dress_right_coup = tf.nn.embedding_lookup(_word_embeddings_dress_right_coup,
                    self.word_ids_dress_right_coup, name="word_embeddings_dress_right_coup")
            word_embeddings_jean_right_coup = tf.nn.embedding_lookup(_word_embeddings_jean_right_coup,
                    self.word_ids_jean_right_coup, name="word_embeddings_jean_right_coup")

        with tf.variable_scope("chars_",reuse = tf.AUTO_REUSE):
            if self.config.use_chars:
                # get char embeddings matrix
                #for dress_left
                _char_embeddings_dress_left_coup = tf.get_variable(
                        name="_char_embeddings_dress_left_coup",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings_dress_left_coup = tf.nn.embedding_lookup(_char_embeddings_dress_left_coup,
                        self.char_ids_dress_left_coup, name="char_embeddings_dress_left_coup")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings_dress_left_coup)
                char_embeddings_dress_left_coup = tf.reshape(char_embeddings_dress_left_coup,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths_dress_left_coup = tf.reshape(self.word_lengths_dress_left_coup, shape=[s[0]*s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings_dress_left_coup,
                        sequence_length=word_lengths_dress_left_coup, dtype=tf.float32)
        

                #for jean_left
                _char_embeddings_jean_left_coup = tf.get_variable(
                        name="_char_embeddings_jean_left_coup",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings_jean_left_coup = tf.nn.embedding_lookup(_char_embeddings_jean_left_coup,
                        self.char_ids_jean_left_coup, name="char_embeddings_jean_left_coup")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings_jean_left_coup)
                char_embeddings_jean_left_coup = tf.reshape(char_embeddings_jean_left_coup,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths_jean_left_coup = tf.reshape(self.word_lengths_jean_left_coup, shape=[s[0]*s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings_jean_left_coup,
                        sequence_length=word_lengths_jean_left_coup, dtype=tf.float32)

                #for dress_right
                _char_embeddings_dress_right_coup = tf.get_variable(
                        name="_char_embeddings_dress_right_coup",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings_dress_right_coup = tf.nn.embedding_lookup(_char_embeddings_dress_right_coup,
                        self.char_ids_dress_right_coup, name="char_embeddings_dress_right_coup")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings_dress_right_coup)
                char_embeddings_dress_right_coup = tf.reshape(char_embeddings_dress_right_coup,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths_dress_right_coup = tf.reshape(self.word_lengths_dress_right_coup, shape=[s[0]*s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings_dress_right_coup,
                        sequence_length=word_lengths_dress_right_coup, dtype=tf.float32)
        

                #for jean_right
                _char_embeddings_jean_right_coup = tf.get_variable(
                        name="_char_embeddings_jean_right_coup",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings_jean_right_coup = tf.nn.embedding_lookup(_char_embeddings_jean_right_coup,
                        self.char_ids_jean_right_coup, name="char_embeddings_jean_right_coup")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings_jean_right_coup)
                char_embeddings_jean_right_coup = tf.reshape(char_embeddings_jean_right_coup,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths_jean_right_coup = tf.reshape(self.word_lengths_jean_right_coup, shape=[s[0]*s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings_jean_right_coup,
                        sequence_length=word_lengths_jean_right_coup, dtype=tf.float32)                        
        self.word_embeddings_dress_left_coup =  tf.nn.dropout(word_embeddings_dress_left_coup, self.dropout)
        self.word_embeddings_jean_left_coup =  tf.nn.dropout(word_embeddings_jean_left_coup, self.dropout)
        self.word_embeddings_dress_right_coup =  tf.nn.dropout(word_embeddings_dress_right_coup, self.dropout)
        self.word_embeddings_jean_right_coup =  tf.nn.dropout(word_embeddings_jean_right_coup, self.dropout)

    def findKmers(self, tag, k):
        if(tag=='O'):
            return ['O']
        kmers =[]
        self.max_kmer_list_size = max(self.max_kmer_list_size,len(tag)-k+1)
        for i in range(len(tag)-k+1):
            kmers.append(tag[i:i+k])
        return kmers

    def add_label_embedding_op(self):
        '''
        This function creates a label embedding using k-mers
        as no of possible chars for label is 27(alphabets & '_' so vocab_size =27)
        we create a vocab_size^k x label_embedding_size tensor to store label embeddings
        '''
        import itertools
        embsize = self.config.label_embedding_size
        kValOfKmer = self.config.kValOfKmer
        tagFile = open("data/tags.txt")
        tags1 = tagFile.readlines()
        tags = []
        for i in tags1:
            tags.append(i.strip())
        del tags1
        tagFile.close()
        vocab = []
        for i in tags:
            vocab.extend(self.findKmers(i,kValOfKmer))
        vocab = set(vocab)
        vocab = list(vocab)
        """
        k_mers tensor stores embeddings of all the possible kmers of labels 
        we need a default as all zeros because size of lists of kmers of each label is different
        so we need to append zeros to make them all equal
        index 0 of k_mers will store all zeros
        As it is a tensor varialbe, we need to make it all zeros again and again
        as it will be updated after each epoch
        """
        k_mers = tf.Variable(tf.random_normal([len(vocab),embsize], stddev=0.1),name="label_embedding", dtype = tf.float32)
        self.k_mers = k_mers
        table = {}
        for idx,key in enumerate(vocab):
            table[key]=idx
        self.label_dict = table
        """
        label_dict will give the index of given two mer 
        so to get embedding of label 'abc'
        emb('abc') = 0.5*(emb('ab')+em('bc'))      0.5 because it is 2-mer
        em('ab') = two_mers[self.label_dict.lookup('ab')]
        """
        # out = table.lookup(tf.constant("ab"))
        # with tf.Session() as sess:
        #     table.init.run()
        #     print(out.eval()) #

        


    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """


        with tf.variable_scope("bi-lstm"):

            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)
            #print "Main output"
            #print output

        with tf.variable_scope("proj", reuse = tf.AUTO_REUSE):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])
            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())
            nsteps = tf.shape(output)[1]
            #print "nsteps"
            #print nsteps
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            #print "New output"
            #print output
            pred = tf.matmul(output, W) + b
            #print "pred"
            #print pred
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])
            #print "logit"
            #print self.logits
            #print (self.logits.eval())
            #with tf.Session():
               # print (pred.eval())


    


    def add_loss_op(self):
        """Defines the loss"""
        if self.config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.y_batch, self.sequence_lengths)
            #init_op = tf.initialize_all_variables()
            #with tf.Session() as sess:
                #sess.run(init_op) 
                #print (sess.run(self.labels))
            #print (self.labels)

            #self.cp_loss = coupling_loss(self.logits, self.labels, self.sequence_lengths)
            
            self.trans_params = trans_params 
            self.loss = tf.reduce_mean(-log_likelihood)
            #self.loss = tf.Print(self.loss,[self.loss])

        tf.summary.scalar("loss", self.loss)




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



    def attention(self,dress, jean):
        
        As_left = tf.placeholder(tf.float32, shape=[None, 300], name="left")
        Bs_left = tf.placeholder(tf.float32, shape=[None, 300], name="left")
        w_omega_left = tf.Variable(tf.random_normal([1, 300], stddev=0.1),name="left")
        with tf.name_scope('left'):
            v_left = tf.matmul(As_left, Bs_left) 
            temp_left = tf.multiply(v_left, w_omega_left)
            output_left = tf.nn.softmax(temp_left, name='alphas_left')


        As_right = tf.placeholder(tf.float32, shape=[None, 300], name="right")
        Bs_right = tf.placeholder(tf.float32, shape=[None, 300], name="right")
        w_omega_right = tf.Variable(tf.random_normal([1, 300], stddev=0.1),name="right")
        with tf.name_scope('right'):
            v_right = tf.matmul(As_right, Bs_right) 
            temp_right = tf.multiply(v_right, w_omega_right)
            output_right = tf.nn.softmax(temp_right, name='alphas_right') 


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #A = np.random.randint(5, size=(3, 3))
            #B = np.random.randint(5, size=(3, 3))
            left = sess.run(output_left,feed_dict={As_left:dress,Bs_left:jean.T})
            right = sess.run(output_right,feed_dict={As_right:jean,Bs_right:dress.T})
            print(left)
            print(right)
        return left,right

    
    def add_coupling_loss_me(self):
        #For Dress_left
        with tf.variable_scope("bi-lstm2", reuse = tf.AUTO_REUSE):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings_dress_left_coup,
                    sequence_length=self.sequence_lengths_dress_left_coup, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)
        with tf.variable_scope("proj", reuse = tf.AUTO_REUSE):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])
            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())
            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits_dress_left_coup = tf.reshape(pred, [-1, nsteps, self.config.ntags])
        #For jean_left
        with tf.variable_scope("bi-lstm2", reuse = tf.AUTO_REUSE):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings_jean_left_coup,
                    sequence_length=self.sequence_lengths_jean_left_coup, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)
        with tf.variable_scope("proj", reuse = tf.AUTO_REUSE):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])
            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())
            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits_jean_left_coup = tf.reshape(pred, [-1, nsteps, self.config.ntags])
                #For Dress_right
        with tf.variable_scope("bi-lstm2", reuse = tf.AUTO_REUSE):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings_dress_right_coup,
                    sequence_length=self.sequence_lengths_dress_right_coup, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)
        with tf.variable_scope("proj", reuse = tf.AUTO_REUSE):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])
            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())
            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits_dress_right_coup = tf.reshape(pred, [-1, nsteps, self.config.ntags])
        #For jean_right
        with tf.variable_scope("bi-lstm2", reuse = tf.AUTO_REUSE):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings_jean_right_coup,
                    sequence_length=self.sequence_lengths_jean_right_coup, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)
        with tf.variable_scope("proj", reuse = tf.AUTO_REUSE):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])
            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())
            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits_jean_right_coup = tf.reshape(pred, [-1, nsteps, self.config.ntags])

        
        iters = tf.constant(self.config.batch_size_coup)
        w_alpha_left = tf.Variable(tf.random_normal([900], stddev=0.1),name="left")
        w_alpha_right = tf.Variable(tf.random_normal([900], stddev=0.1),name="right")

        sentno = tf.constant([i for i in range(self.config.batch_size)])
        indices_left_dress = tf.stack([sentno,self.wordPos_left_dress,self.dress_left_tag_id], axis = 1)
        indices_left_jean = tf.stack([sentno,self.wordPos_left_jean,self.jean_left_tag_id], axis = 1)
        indices_right_dress = tf.stack([sentno,self.wordPos_right_dress,self.dress_right_tag_id], axis = 1)
        indices_right_jean = tf.stack([sentno,self.wordPos_right_jean,self.jean_right_tag_id], axis = 1)
        scores_left_dress = tf.gather_nd(self.logits_dress_left_coup,indices_left_dress)
        scores_left_jean = tf.gather_nd(self.logits_jean_left_coup,indices_left_jean)
        scores_right_dress = tf.gather_nd(self.logits_dress_right_coup,indices_right_dress)
        scores_right_jean = tf.gather_nd(self.logits_jean_right_coup,indices_right_jean)
        scores_left = tf.abs(tf.subtract(scores_left_dress,scores_left_jean))
        scores_right = tf.abs(tf.subtract(scores_right_dress,scores_right_jean))
        dress_left = tf.multiply(self.window_left_dress,w_alpha_left)
        dress_left = tf.multiply(dress_left,self.window_left_jean)  
        dress_left = tf.reduce_sum(dress_left, axis = 1)
        A = tf.nn.softmax(dress_left)
        A /= 900
        jean_right = tf.multiply(self.window_right_jean,w_alpha_right)
        jean_right = tf.multiply(jean_right,self.window_right_dress)
        jean_right = tf.reduce_sum(jean_right, axis = 1)
        A_ = tf.nn.softmax(jean_right)
        A_ /= 900

        iters = tf.constant(self.config.batch_size_coup)
        def cond(cp_loss,i, iters):
            return tf.less(i, iters) 
        def body(cp_loss,i, iters):
            valid_indices = tf.where(tf.not_equal(self.dress_left_kmer_indices[i],-1))  # stripping off those -1 padding bits
            dress_left_kmer_indices_ = tf.gather(self.dress_left_kmer_indices[i],valid_indices)   # required unpadded tensor
            kmer_left_dress = tf.gather(self.k_mers,dress_left_kmer_indices_,axis=0)  
            kmer_left_dress = tf.reduce_sum(kmer_left_dress,axis = 0)   # final embedding of label i
            valid_indices = tf.where(tf.not_equal(self.jean_left_kmer_indices[i],-1))
            jean_left_kmer_indices_ = tf.gather(self.jean_left_kmer_indices[i],valid_indices)
            kmer_left_jean = tf.gather(self.k_mers,jean_left_kmer_indices_,axis=0)
            kmer_left_jean = tf.reduce_sum(kmer_left_jean,axis = 0)
            valid_indices = tf.where(tf.not_equal(self.dress_right_kmer_indices[i],-1))  
            dress_right_kmer_indices_ = tf.gather(self.dress_right_kmer_indices[i],valid_indices)   
            kmer_right_dress = tf.gather(self.k_mers,dress_right_kmer_indices_,axis=0)  
            kmer_right_dress = tf.reduce_sum(kmer_right_dress,axis = 0)   
            valid_indices = tf.where(tf.not_equal(self.jean_right_kmer_indices[i],-1))
            jean_right_kmer_indices_ = tf.gather(self.jean_right_kmer_indices[i],valid_indices)
            kmer_right_jean = tf.gather(self.k_mers,jean_right_kmer_indices_,axis=0)
            kmer_right_jean = tf.reduce_sum(kmer_right_jean,axis = 0)
            kmer_left = tf.reduce_sum(tf.multiply(kmer_left_dress,kmer_left_jean))
            kmer_right = tf.reduce_sum(tf.multiply(kmer_right_dress,kmer_right_jean))
            kmer_left *= (1/self.config.label_embedding_size)       #tf scalar
            kmer_right *= (1/self.config.label_embedding_size)    
            x = tf.multiply(tf.add(A[i],kmer_left),scores_left[i])
            y = tf.multiply(tf.add(A_[i],kmer_right),scores_right[i]) 
            z = tf.add(tf.abs(x),tf.abs(y))
            cp_loss = tf.add(cp_loss,z)  
            return [cp_loss, tf.add(i, 1), iters]

        cp_loss = tf.constant(0.0)
        cp_loss,_,_ = tf.while_loop(cond, body, [cp_loss,0,iters])
        self.loss = self.loss + tf.abs(cp_loss) 
        return self.loss


  
    def add_coupling_loss(self):

        f = open("data/dress_list.txt",'r')
        g = open("data/tags.txt", 'r')
        tag_list =[]

        cp_loss =0
        
        for lines in g.readlines():
            ls = lines.split("\n")
            tag_list.append(lines)

        #print(self.labels)
        example =[]
        count =[]
        my_lambda =[]

        for lines in f.readlines():
            ls = lines.strip().split("\t")
            example.append(ls[0])
            count.append(ls[1])
            #my_lambda.append(ls[2])

        val = tf.map_fn(lambda x: (x, x), self.labels, dtype=(tf.int32, tf.int32))
        
        #with tf.Session() as sess:
            #print(sess.run(val),feed_dict=fd)


        #fp = "data/glove.6B/glove.6B.{}d.txt".format(dim_word)
        file = "data/glove.6B/glove.6B.300d"

        for i in range (0,len(tag_list)):
            if i in val:
                st = example[i].split(" ")
                a = st[1]
                #b = st[3]
                #m = example.index(a)
                #n = example.index(b)
                aa = st[0]
                #bb = st[2]
                ll = []
                rr = []
                ll_tag =[]
                rr_tag =[]
                model= loadGloveModel(file) 
                ll.append(model[aa])
                ll_tag.append(st[3])

                for items in example:
                    temp = items.split(" ")
                    if temp[0] == aa:
                        rr.append(model[temp[2]])
                        
                        rr_tag.append(temp[3])

                left,right = attention(ll,rr)

                for p in range(0,len(rr_tag)):
                    A = left[0][rr[p]]
                    A_ = right[rr[p][0]]
                    q = example.indes(tag_list[p])

                    cp_loss += (A+A_)*(abs(self.logits[i]-self.logits[q]))

        
        f.close()
        g.close()
        self.loss = self.loss + cp_loss
        return self.loss



    



    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        if(self.config.use_kmer):
            self.add_label_embedding_op()
        self.add_logits_op()
        self.add_loss_op()
        self.add_coupling_loss_me()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, words):
        perform_op_coup=True
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        #for i in words:
            #print(str(i))
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)
        #print("From predict batch: " + str(words))
        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = None, None
            logits_dress_left_coup, logits_jean_left_coup, logits_dress_right_coup, logits_jean_right_coup = None,None,None,None
            if perform_op_coup:
                logits, trans_params,logits_dress_left_coup, logits_jean_left_coup, logits_dress_right_coup, logits_jean_right_coup = self.sess.run(
                        [ self.logits, self.trans_params,self.logits_dress_left_coup, self.logits_jean_left_coup, self.logits_dress_right_coup, self.logits_jean_right_coup], feed_dict=fd)
            else:
                logits, trans_params = self.sess.run(
                     [self.logits, self.trans_params], feed_dict=fd)
                
            #print("todo line 624 @ner_model")
            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]
            #print("LABELS: " +str(viterbi_sequences))
            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths

    def run_epoch(self, train, dev, epoch,data_for_copuling):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size_for_crf_loss = self.config.batch_size
        nbatches = (len(train) + batch_size_for_crf_loss - 1) // batch_size_for_crf_loss
        prog = Progbar(target=nbatches)
        #########
        #todo: calculate batch size for copuling loss calculation
        batch_size_for_coupling_loss = batch_size_for_crf_loss

        # iterate over dataset

        import time
        start = time.time()
        
        minibatch_data = minibatches(train, batch_size_for_crf_loss,data_for_copuling,batch_size_for_coupling_loss)
        for i, returned_data in enumerate(minibatch_data):    
            x_batch             = returned_data[0]     
            y_batch             = returned_data[1]

            x_left_batch_dress  = returned_data[2]
            y_left_batch_dress  = returned_data[3]
            dress_left_tag_id   = returned_data[4]
            window_left_dress   = returned_data[5]
            kmer_left_dress     = returned_data[6]

            x_left_batch_jean   = returned_data[7]
            y_left_batch_jean   = returned_data[8]
            jean_left_tag_id    = returned_data[9]
            window_left_jean    = returned_data[10]
            kmer_left_jean      = returned_data[11]

            x_right_batch_dress = returned_data[12]
            y_right_batch_dress = returned_data[13]
            dress_right_tag_id  = returned_data[14]
            window_right_dress  = returned_data[15]
            kmer_right_dress    = returned_data[16]

            x_right_batch_jean  = returned_data[17]
            y_right_batch_jean  = returned_data[18]
            jean_right_tag_id   = returned_data[19]
            window_right_jean   = returned_data[20]
            kmer_right_jean     = returned_data[21]

            wordPos_left_dress  = returned_data[22]
            wordPos_left_jean   = returned_data[23]
            wordPos_right_dress = returned_data[24]
            wordPos_right_jean  = returned_data[25]

            del returned_data   

            dress_left_kmer_indices = None
            jean_left_kmer_indices = None
            dress_right_kmer_indices = None
            jean_right_kmer_indices = None
            if(self.config.use_kmer ):
                dress_left_kmer_indices  = self.convertTagKmerToEmbedding(kmer_left_dress)
                jean_left_kmer_indices = self.convertTagKmerToEmbedding(kmer_left_jean)
                dress_right_kmer_indices = self.convertTagKmerToEmbedding(kmer_right_dress)
                jean_right_kmer_indices = self.convertTagKmerToEmbedding(kmer_right_jean)

            fd, _ = self.get_feed_dict(x_batch, y_batch,           
              x_left_batch_dress, x_left_batch_jean, x_right_batch_dress, x_right_batch_jean, 
              y_left_batch_dress, y_left_batch_jean, y_right_batch_dress,y_right_batch_jean,
              dress_left_tag_id, jean_left_tag_id, dress_right_tag_id, jean_right_tag_id,
              window_left_dress, window_left_jean, window_right_dress, window_right_jean,
              dress_left_kmer_indices, jean_left_kmer_indices, dress_right_kmer_indices, jean_right_kmer_indices, 
              wordPos_left_dress, wordPos_left_jean, wordPos_right_dress, wordPos_right_jean,
              lr = self.config.lr,
              dropout= self.config.dropout)       
            st = time.time()           
            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)
            en = time.time()
            hours, rem = divmod(en-st, 3600)
            minutes, seconds = divmod(rem, 60)
            prog.update(i + 1, [("train loss", train_loss)])
            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("\nTime Elapsed in this epoch : ")
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]


    def run_evaluate(self, test, k ):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        out = ''
        if(k==0):
            file = open("Task1_actual_pred_labels.txt", "w")
        else:
            file = open("Task2_actual_pred_labels.txt", "w")
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels, sumit in minibatches(test, self.config.batch_size,None,None):
            labels_pred, sequence_lengths = self.predict_batch(words)
            
            for desc, lab, lab_pred, length in zip(sumit, labels, labels_pred,
                                             sequence_lengths):
                
                for i, j, k in zip(desc, lab, lab_pred):
                    out = out + str(self.idx_to_word[i[1]]) + " "+str(self.idx_to_tag[j]) + " " + str(self.idx_to_tag[k]) + "\n"
                out = out + "\n"
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]
                for zz in zip(lab, lab_pred):
                    file.write(str(zz[0]) + " " + str(zz[1]) +"\n")

                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        with open("./test_out.txt", 'w') as f:
            f.write(out)
        print ("Precision:" + " " + str(p) + " " + "Recall:" + " " + str(r))
        return {"acc": 100*acc, "f1": 100*f1}


    # def predict(self, words_raw):
    #     """Returns list of tags

    #     Args:
    #         words_raw: list of words (string), just one sentence (no batch)

    #     Returns:
    #         preds: list of tags (string), one for each word in the sentence

    #     """
    #     #print(words_raw)
    #     words = [self.config.processing_word(w) for w in words_raw]
    #     if type(words[0]) == tuple:
    #         words = zip(*words)
    #     pred_ids, _ = self.predict_batch([words],False)
    #     preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

    #     return preds
