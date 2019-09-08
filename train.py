from model.data_utils import FKDataset,DatasetForCouplingLoss

from model.ner_model import NERModel
from model.config import Config


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    #model.restore_session("results/test/model.weights/") # optional, restore weights
    #model.restore_session(config.dir_model)
    #model.reinitialize_weights("proj")

    # create datasets
    kValOfKmer = config.kValOfKmer
    dev   = FKDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = FKDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)
    

    train_for_coupling = DatasetForCouplingLoss(config.filename_lambda, config.processing_word,
                         config.processing_tag, config.max_iter, kValOfKmer)
            
    # for i in train_for_coupling:
    #     print(i)


    # train model
    model.train(train, dev, train_for_coupling)


if __name__ == "__main__":
    main()
