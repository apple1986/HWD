import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import pickle
import cv2


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # 3 channel?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath,mode='r')
            image, label = data['image'][:], data['label'][:]


        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
            image = sample['image']
            label = sample['label']
        else:
            image = np.expand_dims(image, axis=0)

        sample['case_name'] = self.sample_list[idx].strip('\n')
        name = self.sample_list[idx].strip('\n')
        size = image.shape

        # return sample
        # image = np.expand_dims(image, axis=0)
        return image, label, np.array(size), name

#####################obtain data infor
class SynapseTrainInform:
    """ To get statistical information about the train set, such as mean, std, class distribution.
        The class is employed for tackle class imbalance.
    """

    def __init__(self, data_dir='', classes=9, train_set_file="",
                 inform_data_file="", normVal=1.10):
        """
        Args:
           data_dir: directory where the dataset is kept
           classes: number of classes in the dataset
           inform_data_file: location where cached file has to be stored
           normVal: normalization value, as defined in ERFNet paper
        """
        self.data_dir = data_dir
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(1, dtype=np.float32)
        self.std = np.zeros(1, dtype=np.float32)
        self.train_set_file = train_set_file
        self.inform_data_file = inform_data_file

    def compute_class_weights(self, histogram):
        """to compute the class weights
        Args:
            histogram: distribution of class samples
        """
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))

    def readWholeTrainSet(self, fileName, train_flag=True):
        """to read the whole train set of current dataset.
        Args:
        fileName: train set file that stores the image locations
        trainStg: if processing training or validation data
        
        return: 0 if successful
        """
        global_hist = np.zeros(self.classes, dtype=np.float32)

        no_files = 0
        min_val_al = 0
        max_val_al = 0
        with open(fileName, 'r') as textFile:
            # with open(fileName, 'r') as textFile:
            for line in textFile:
                # we expect the text file to contain the data in following format
                # <RGB Image> <Label Image>
                line_arr = line.split()
                data_path = ((self.data_dir).strip() + '/' + line_arr[0].strip()).strip() + ".npz"
                data = np.load(data_path)
                img, label = data['image'], data['label']

                unique_values = np.unique(label)
                max_val = max(unique_values)
                min_val = min(unique_values)

                max_val_al = max(max_val, max_val_al)
                min_val_al = min(min_val, min_val_al)

                if train_flag == True:
                    hist = np.histogram(label, self.classes, [0, self.classes - 1])
                    global_hist += hist[0]

                    self.mean[0] += np.mean(img[:, :])
                    # self.mean[1] += np.mean(img[:, :, 1])
                    # self.mean[2] += np.mean(img[:, :, 2])

                    self.std[0] += np.std(img[:, :])
                    # self.std[1] += np.std(img[:, :, 1])
                    # self.std[2] += np.std(img[:, :, 2])

                else:
                    print("we can only collect statistical information of train set, please check")

                if max_val > (self.classes - 1) or min_val < 0:
                    print('Labels can take value between 0 and number of classes.')
                    print('Some problem with labels. Please check. label_set:', unique_values)
                    print('Label Image ID: ' + data_path)
                no_files += 1

        # divide the mean and std values by the sample space size
        self.mean /= no_files
        self.std /= no_files

        # compute the class imbalance information
        self.compute_class_weights(global_hist)
        return 0

    def collectDataAndSave(self):
        """ To collect statistical information of train set and then save it.
        The file train.txt should be inside the data directory.
        """
        print('Processing training data')
        return_val = self.readWholeTrainSet(fileName=self.train_set_file)

        print('Pickling data')
        if return_val == 0:
            data_dict = dict()
            data_dict['mean'] = 0 #self.mean
            data_dict['std'] = 1 #self.std
            data_dict['classWeights'] = self.classWeights
            print(self.mean, '#',self.std,'#',self.classWeights)
            pickle.dump(data_dict, open(self.inform_data_file, "wb"))
            return data_dict
        return None


if __name__ == '__main__':
    cal_mean_std = SynapseTrainInform(data_dir='/data/gpxu/Synapse/train_npz', classes=9,
                        train_set_file="/data/gpxu/Synapse_trans/TransUNet/lists/lists_Synapse/train.txt",
                        inform_data_file= '/home/gpxu/vess_seg/vess_efficient/dataset/inform/synapse.pkl'
                        )
    obtain_mean_std = cal_mean_std.collectDataAndSave()
    # F=open('/home/gpxu/vess_seg/vess_efficient/dataset/inform/camvid_inform.pkl','rb')
    # content=pickle.load(F)      
    # print(content)