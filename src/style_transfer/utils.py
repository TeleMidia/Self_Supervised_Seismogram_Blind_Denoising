
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random as rand
from tqdm import tqdm
import os
import tensorflow as tf


exp_chart_folder = None
model_weights_folder = None
dict_chart_data = None


class CustomMetric:
    def __init__(self, name="none"):
        self.buffer = []
        self.name = name
    def feed(self, batch_y, predictions):
        if self.name == "PSNR":
            self.buffer = np.concatenate((self.buffer, tf.image.psnr(batch_y, predictions, 
                            max_val=1.0).numpy()), axis=None) 
    def result(self):
        return np.mean(self.buffer)

    def reset_states(self):
        self.buffer = []


def check_experiment_folders():
    global exp_chart_folder, model_weights_folder
    if exp_chart_folder is None or model_weights_folder is None:
        return False
    return True

def create_experiment_folders(exp_id):
    global exp_chart_folder, model_weights_folder
    exp_chart_folder = os.path.join(exp_id, "chart_data")
    model_weights_folder = os.path.join(exp_id, "model_weights")
    if not os.path.exists(exp_chart_folder):
        os.makedirs(exp_chart_folder)
    if not os.path.exists(model_weights_folder):
        os.makedirs(model_weights_folder)
    return 

def load_chart_data():
    assert check_experiment_folders()
    global exp_chart_folder, dict_chart_data
    path =  os.path.join(exp_chart_folder, "data.txt")
    if os.path.exists(path):
        with open(path, "r") as file:
            dict_chart_data = eval(file.readline())
    else:
        dict_chart_data = {}
        dict_chart_data["step"] = []
        dict_chart_data["train_loss"] = []
        dict_chart_data["train_eval"] = []
        dict_chart_data["valid_eval"] = []
        dict_chart_data["valid_eval_2"] = []
    return

def update_chart_data(step, train_loss, train_eval, valid_eval, valid_eval_2):
    assert check_experiment_folders()
    global exp_chart_folder,dict_chart_data
    assert dict_chart_data is not None
    path =  os.path.join(exp_chart_folder, "data.txt")

    dict_chart_data["step"].append(step)
    dict_chart_data["train_loss"].append(train_loss)
    dict_chart_data["train_eval"].append(train_eval)
    dict_chart_data["valid_eval"].append(valid_eval)
    dict_chart_data["valid_eval_2"].append(valid_eval_2)

    if os.path.exists(path):
        os.remove(path) 
    with open(path, "w") as file:
        file.write(str(dict_chart_data))
        
    return 

def annot_max(ax, x,y, op="min"):

    if op=="min":
        xmax = x[np.argmin(y)]
        ymax = y.min()
    else:
        xmax = x[np.argmax(y)]
        ymax = y.max()

    text= "epoch={}, result={:.4f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=1)
    arrowprops=dict(arrowstyle="->")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

    



def draw_chart(titles):
    global dict_chart_data

    if len(dict_chart_data["step"]) == 0:
        return

    fig, axs = plt.subplots(3, figsize=(15,15))
    axs[0].plot(dict_chart_data["step"], dict_chart_data["train_loss"], linewidth=2, color="orange", label=titles[0])
    axs[0].legend(frameon=False, loc='upper center', ncol=1)

    axs[1].plot(dict_chart_data["step"], dict_chart_data["train_eval"], linewidth=2, color="red", label=titles[1])
    axs[1].plot(dict_chart_data["step"], dict_chart_data["valid_eval"], linewidth=2, color="blue", label=titles[2])
    axs[1].legend(frameon=False, loc='upper center', ncol=2)
    annot_max(axs[1], np.asarray(dict_chart_data["step"]), np.asarray(dict_chart_data["valid_eval"]) )

    axs[2].plot(dict_chart_data["step"], dict_chart_data["valid_eval_2"], linewidth=2, color="green", label=titles[3])
    axs[2].legend(frameon=False, loc='upper center', ncol=1)
    annot_max(axs[2], np.asarray(dict_chart_data["step"]), np.asarray(dict_chart_data["valid_eval_2"]), op="max")

    plt.show()



def load_dataset(file_path):

    lines = open(file_path, "r").readlines()
    
    total_images = len(lines)
    print("Loading", total_images, "images ...")

    img_size = 300

    dataset_X = np.zeros((total_images, img_size, img_size, 1), dtype=np.float32)
    dataset_Y = np.zeros((total_images, img_size, img_size, 1), dtype=np.float32)

    for index, line in tqdm(enumerate(lines)):
        line = line.replace('\n', '')

        paths = line.split(";") 

        img_x = np.array(Image.open(paths[0]))
        img_y = np.array(Image.open(paths[1]))
    
        if img_x is None or img_y is None:
            print("Corrupted dataset!")
            return None, None
        

        dataset_X[index,:,:,0] = img_x
        dataset_Y[index,:,:,0] = img_y
    
    return dataset_X, dataset_Y

def show_samples(dataset_x, dataset_y, begin=0, end=1):
    quant = end - begin
    fig, axs = plt.subplots(quant, figsize=(15,15), frameon=True)
    for index in range(begin,end):
        axs[index].axis('off')
        axs[index].imshow(np.concatenate((dataset_x[index,:,:,0],dataset_y[index,:,:,0]), axis=1), cmap="bwr", vmin=0, vmax=1)
        
    plt.show()
   

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches