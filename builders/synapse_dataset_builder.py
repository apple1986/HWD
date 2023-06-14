import os
import pickle
from numpy.core.fromnumeric import transpose
from torch.functional import split
from torch.utils import data
from dataset.most import MOSTDataSet, MOSTValDataSet, MOSTTrainInform, MOSTTestDataSet
from dataset.synapse import Synapse_dataset, SynapseTrainInform, RandomGenerator
# from dataset.synapse_fft import Synapse_dataset, SynapseTrainInform, RandomGenerator
from torchvision import transforms


def build_dataset_train(dataset, input_size, batch_size, train_type, random_scale, random_mirror, num_workers):
    base_dir = os.path.join('/data/gpxu/Synapse/')
    data_dir = os.path.join(base_dir, 'train_npz')
    dataset_list = 'train.txt'
    list_dir = os.path.join(base_dir, 'lists_Synapse')
    # val_data_list = os.path.join(data_dir, 'lists_Synapse',  'train.txt')
    inform_data_file = os.path.join('./dataset/inform/','synapse.pkl')

    # inform_data_file collect the information of mean, std and weigth_class
    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))
        if dataset == 'synapse':
            dataCollect = SynapseTrainInform(data_dir, 9, train_set_file=dataset_list,
                                            inform_data_file=inform_data_file)
        else:
            raise NotImplementedError(
                "This repository now supports two datasets: cityscapes and camvid, %s is not included" % dataset)

        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))

    if dataset == "synapse":
        trainLoader = data.DataLoader(
            Synapse_dataset(data_dir, list_dir, split='train',transform=transforms.Compose(
                                   [RandomGenerator(output_size=[input_size[0], input_size[1]])])),
            # Synapse_dataset(data_dir, list_dir, split='train',transform=None),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        valLoader = None

        return datas, trainLoader, valLoader


def build_dataset_test(dataset, num_workers, none_gt=False):
    base_dir = os.path.join('/data/gpxu/Synapse/')
    data_dir = os.path.join(base_dir, 'test_vol_h5')
    dataset_list = 'test_vol.txt'
    list_dir = os.path.join(base_dir, 'lists_Synapse')
    # val_data_list = os.path.join(data_dir, 'lists_Synapse',  'train.txt')
    inform_data_file = os.path.join('./dataset/inform/','synapse.pkl')

    # inform_data_file collect the information of mean, std and weigth_class
    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))
        if dataset == 'synapse':
            dataCollect = SynapseTrainInform(data_dir, 9, train_set_file=dataset_list,
                                            inform_data_file=inform_data_file)
        elif dataset == 'most_png':
            dataCollect = MOSTTrainInform(data_dir, 2, train_set_file=dataset_list,
                                            inform_data_file=inform_data_file)
        else:
            raise NotImplementedError(
                "This repository now supports two datasets: cityscapes and camvid, %s is not included" % dataset)
        
        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))

    if dataset == "synapse":
        # for synapse, if test on validation set, set none_gt to False
        # if test on the test set, set none_gt to True
        if none_gt:
            testLoader = None
        else:
            # test_data_list = os.path.join(list_dir +  'test_vol.txt')
            testLoader = data.DataLoader(
                Synapse_dataset(data_dir, list_dir, split='test_vol',transform=None),
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

        return datas, testLoader
