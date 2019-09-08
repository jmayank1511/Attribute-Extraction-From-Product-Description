from model.config import Config
from model.data_utils import FKDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word


def main():
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """
    # get config and processing of words
    config = Config(load=False)
    processing_word = get_processing_word(lowercase=True)

    # Generators
    dev   = FKDataset(config.filename_dev, processing_word)
    test1  = FKDataset(config.filename_test1, processing_word)
    test2  = FKDataset(config.filename_test2, processing_word)

    train = FKDataset(config.filename_train, processing_word)

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test1,test2])
    vocab_glove = get_glove_vocab(config.filename_glove)

    #print ("Inside build data and prinitng vocab_tags")
    

    vocab_tags_task1 =[]
    vocab_tags_task2 =[]

    for items in vocab_tags:
        if "_dress" in items:
            vocab_tags_task1.append(items)
        if "_jean" in items:
            vocab_tags_task2.append(items)

    vocab_tags_task1.append('O')
    vocab_tags_task2.append('O')




    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)

    # Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    # Trim GloVe Vectors
    vocab = load_vocab(config.filename_words)
    export_trimmed_glove_vectors(vocab, config.filename_glove,
                                config.filename_trimmed, config.dim_word)

    # Build and save char vocab
    train = FKDataset(config.filename_train)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.filename_chars)


if __name__ == "__main__":
    main()
