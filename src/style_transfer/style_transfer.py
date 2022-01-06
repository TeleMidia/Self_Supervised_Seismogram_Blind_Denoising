import os
import sys
import scipy.io
import scipy.misc
from nst_utils import *
import numpy as np

import cv2
import random
from tqdm import tqdm

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


model_global = None
sess_global = None



def set_config1(config):
    global min_box_w, max_box_w, min_offset, max_offset, max_iterations

   
def compute_content_cost(a_C, a_G):
 
    # obtendo as dimensões do tensor a_G 
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape a_C and a_G 
    a_C_unrolled = tf.reshape(a_C,[n_H*n_W,n_C])
    a_G_unrolled = tf.reshape(a_G,[n_H*n_W,n_C])
    
    # Calcule a função de custo
    J_content = (1/(4*n_H*n_W*n_C))*tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled)))
    
    
    return J_content

def gram_matrix(A):
    
    GA = tf.matmul(A,A,transpose_b=True)
    
    return GA

def compute_layer_style_cost(a_S, a_G):

    
    # Obtendo as dimensões de a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Resahepe dos tensores (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.reshape(tf.transpose(a_S),[n_C, n_H*n_W])
    a_G = tf.reshape(tf.transpose(a_G),[n_C, n_H*n_W])

    # Calculando as matrizes Gram
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Calculando a perda
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS,GG)))*(1/(4*(n_C**2)*( (n_H*n_W)**2 )))
    
    return J_style_layer


STYLE_LAYERS = [
    ('conv1_1', 0.1),
    ('conv2_1', 0.1),
    ('conv3_1', 2.0),
    ('conv4_1', 1.0),
    ('conv5_1', 1.0)]

def compute_style_cost(sess, model, STYLE_LAYERS):
    J_style = 0
    
    for layer_name, coeff in STYLE_LAYERS:

        #Obtendo o tensor atual 
        out = model[layer_name]

        #Obtendo a ativação do tensor
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # Calculando o custo
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # adicionando o coeficiente ao custo
        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 80):

    J = alpha*J_content + beta*J_style
    
    return J

def model_nn(sess, model, train_step, J, J_content, J_style, input_image, num_epochs = 100):
    
    # inicializando as variaveis
    sess.run(tf.global_variables_initializer())
   
    
    # Run the noisy input image (initial generated image) through the model. Use assign(). 
    sess.run(model['input'].assign(input_image))
   
    
    for i in tqdm(range(num_epochs)):
    
        #Rode o "train_step" para minimizar o custo total
        sess.run(train_step)
        
        #Computar a imagem gerada rodando o model['input']
        generated_image = sess.run(model['input'])

        #Printar informaç˜oes 
        #if i%1000 == 0:
        #    Jt, Jc, Js = sess.run([J, J_content, J_style])
        #    print("Iteration " + str(i) + " :")
        #    print("total cost = " + str(Jt))
        #    print("content cost = " + str(Jc))
        #    print("style cost = " + str(Js))
            
    
    # salvando a última imagem 
    generated_image = restore_image(generated_image)
    
    return np.squeeze(generated_image)



def print_feature_map(sess_global, model_global, layer_name, sufix):
    feature_maps = sess_global.run(model_global[layer_name])
    print("Saída do tensor:",feature_maps.shape)

    folder_name = layer_name+sufix
    for c in range(feature_maps.shape[-1]):

        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        file_name = folder_name+"/"+str(c)+".jpg"
        if os.path.exists(file_name):
            os.remove(file_name)

        cv2.imwrite(file_name, feature_maps[0, :, :, c]) 

        plt.imshow(feature_maps[0, :, :,c], cmap="gray")
        plt.pause(0.1)


def run_style_tranfer(STYLE_W, content_image, style_image, num_epochs=100, lr=2.0, output_gray=True):
    
    global model_global, sess_global
    
    print("Params:")
    if STYLE_W is not None:
        STYLE_LAYERS = STYLE_W
        print(STYLE_LAYERS)
    
    print("lr", lr)
    print("num_epochs", num_epochs)
    
    if model_global is None:

        # Reset the graph
        tf.reset_default_graph()

        #Intanciando a sessao
        sess_global = tf.InteractiveSession()

        model_global = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
    
    #print("loading images ...")
    content_image = reshape_and_normalize_image(content_image)
    #print("content image loaded")
    style_image = reshape_and_normalize_image(style_image)
    #print("style image loaded")

    generated_image = generate_noise_image(content_image)

    # Assign da imagem de conteúdo na entrada da rede VGG-19.  
    sess_global.run(model_global['input'].assign(content_image))

    #-----------------------------
    #print_feature_map(sess_global, model_global, 'conv1_2', 'signal')
    #print_feature_map(sess_global, model_global, 'conv2_2', 'signal')
    #print_feature_map(sess_global, model_global, 'conv3_4', 'signal')
    #print_feature_map(sess_global, model_global, 'conv4_2', 'signal')
   

    #Obtendo o tensor te saida conv4_2
    out = model_global['conv4_2']

    #saída de ativação do tensor  conv4_2
    a_C = sess_global.run(out)

    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out

    # Compute the content cost
    J_content = compute_content_cost(a_C, a_G)

    # Assign the input of the model to be the "style" image 

    sess_global.run(model_global['input'].assign(style_image))

    # Compute the style cost
    J_style = compute_style_cost(sess_global, model_global, STYLE_LAYERS)

    J = total_cost(J_content, J_style)

    # define optimizer (1 line)
    optimizer = tf.train.AdamOptimizer(lr)

    # define train_step (1 line)
    train_step = optimizer.minimize(J)


     # inicializando as variaveis
    sess_global.run(tf.global_variables_initializer())
   
    
    # Run the noisy input image (initial generated image) through the model. Use assign(). 
    sess_global.run(model_global['input'].assign(generated_image))

    #print("initializing style tranfer process")
    final_img = model_nn(sess_global, model_global, train_step, J, J_content, J_style,   generated_image, num_epochs = num_epochs)
  

         

    return final_img



def gen_mask(shape, config=0):
    boxes_x_list = []
    mask_image = np.ndarray(shape=shape, dtype=np.uint8)
    mask_image[:,:] = 0.7
    cursor_1 = 5
    cursor_2 = 5

    min_box_w = 0
    max_box_w = 0
    min_offset = 0
    max_offset = 0
    max_iterations = 0

    if config == 0:
        min_box_w = 5
        max_box_w = 80
        min_offset = 35
        max_offset = 100
        max_iterations=5
    else:
        min_box_w = 5
        max_box_w = 15
        min_offset = 100
        max_offset = 250 
        max_iterations = 3
    
    iterations = random.randint(1, max_iterations)
    
    while(cursor_2 < shape[1] and iterations > 0):
        rand_offset = random.randint(min_offset, max_offset)
        rand_box_w = random.randint(min_box_w,max_box_w)
        
        cursor_1 = cursor_2 + rand_offset
        cursor_2 = cursor_1 + rand_box_w

        if cursor_1 > shape[1] or cursor_2 > shape[1]:
            break
    
        mask_image[:,cursor_1:cursor_2] = 1

        boxes_x_list.append((cursor_1, cursor_2))
        
        iterations = iterations -1
        
    return mask_image, boxes_x_list


def generate_ugly_sismo(good_img_path, ugly_img_path, mask_list):
    
    gen_image_list = []
    
    for mask in mask_list:
        mask_image = mask[0]
        
        content_img = cv2.imread(good_img_path, 0)
        content_img = cv2.resize(content_img, (400,300), interpolation=cv2.INTER_AREA)
        content_img_masked = np.multiply(content_img, mask_image)
        #content_img_masked = cv2.cvtColor(content_img_masked, cv2.COLOR_GRAY2RGB)

        #imshow(content_img_masked, cmap="gray", vmin=0, vmax=255)

        style_img = cv2.imread(ugly_img_path, 0)
        #style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
        style_img = cv2.resize(style_img, (400,300), interpolation=cv2.INTER_AREA)

        gen_image = run_style_tranfer(content_image=content_img, style_image=style_img)
        #gen_image = run_style_tranfer(content_image=content_img_masked, style_image=style_img)
        gen_image_list.append(gen_image)
        
    return gen_image_list

def analyze_region(region):
    #print("shape:", region.shape)
    #min = np.amin(region)
    #print("min", min)
    #max = np.amax(region)
    #print("max", max)
    mean = np.mean(region)
    #print("mean", mean)
    return mean


def center_image(image, boxes_x, margin=10):

    centered_img = np.ndarray(shape=image.shape)
    centered_img[:,:] = 0
    aux_img = np.ndarray(shape=image.shape)
    aux_img[:,:] = 0

    bounding_boxes_list=[]
   
    for box_x in boxes_x:
        mean = analyze_region(image[:, box_x[0]:box_x[1]])
        centered_img[:, (box_x[0]):(box_x[1])] = image[:, (box_x[0]):(box_x[1])] - mean
        centered_img = np.where((centered_img > -40) & (centered_img < -10), 0 , centered_img)

        #calcule left border
        #aux_img[:, (box_x[0]-margin):(box_x[0])] = image[:, (box_x[0]-margin):(box_x[0])] - mean
        #aux_img = np.where((aux_img > -80) & (aux_img < -30), 0 , aux_img)
        #centered_img = centered_img + aux_img
                
        #cacule right border
        #aux_img[:,:] = 0
        #aux_img[:, (box_x[1]):(box_x[1]+margin)] = image[:, (box_x[1]):(box_x[1]+margin)] - mean
        #aux_img = np.where((aux_img > -80) & (aux_img < -30), 0 , aux_img)
        #centered_img = centered_img + aux_img
        
    return centered_img

def save_annotation(index, image_path, boxes_x, width):

    filename = image_path.split(".")[0]+".txt" 

    file = open(filename, "w")

    for box in boxes_x:
        xmin = box[0]/float(width)
        xmax = box[1]/float(width)

        file.write(str(xmin)+",0,"+str(xmax)+",1.0\n")

    file.close()

def save_simogram(index, ori_img_path, img_data):

    filename = ori_img_path.split("/")[-1].split(".")[0]+"_aug_"+str(index)+".jpg"

    ann_folder = "annotations/" 

    cv2.imwrite(ann_folder+filename, img_data)


def generate_sismograms(good_image_path, ugly_image_path, qtdy=10, output_shape=(300,400), config=0):

    mask_list = []

    set_config1(config)

    for i in range(0, qtdy):
        plt.pause(.1)
        mask, boxes_x = gen_mask(shape=(300,400))
        mask_list.append((mask,boxes_x))
        #imshow(mask, cmap="gray")
        save_annotation(i, good_image_path, boxes_x, 400)   

    gen_image_list = generate_ugly_sismo(good_img_path=good_image_path, ugly_img_path=ugly_image_path, mask_list=mask_list)

    centered_img_list = []

    for index, gen_image in enumerate(gen_image_list):

        imshow(gen_image, cmap="gray", vmin=0, vmax=255)
        plt.pause(.1)
        '''
        boxes_x = mask_list[index][1]
        centered_img = center_image(gen_image, boxes_x)
        centered_img_list.append(centered_img)
        '''



    '''
    content_img = cv2.imread(good_image_path, 0)
    content_img = cv2.resize(content_img, (400,300), interpolation=cv2.INTER_AREA)

    final_image_list = []

    for index, gen_image in enumerate(gen_image_list):
        centered_img = centered_img_list[index]
        final_image = content_img + centered_img
        
        if final_image.shape != output_shape:
            final_image = cv2.resize(final_image, output_shape, interpolation=cv2.INTER_AREA)

        final_image_list.append(final_image)
        save_simogram(index, good_image_path, final_image)
        #imshow(final_image, cmap="gray", vmin=0, vmax=255)
        #plt.pause(.1)

    '''

    #return final_image_list