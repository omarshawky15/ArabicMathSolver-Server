import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sympy as sp

all_labels = {54: '8', 2: 'jeem', 53: '7', 49: '3', 37: 'sum', 0: 'alif', 47: '1', 1: 'ba', 33: 'neq', 51: '5', 3: 'dal', 6: 'sen', 20: '[', 55: '9', 39: 'times', 13: 'nun', 42: 'lim', 15: 'waw', 57: 'Sec', 18: '(', 17: '-', 23: '}', 8: 'ta', 36: 'sqrt', 10: 'qaf', 30: 'int', 45: 'tg', 29: 'infty', 31: 'leq', 35: 'sigma', 28: 'csc', 26: 'geq', 19: ')', 41: 'cos', 25: 'div', 7: 'sad', 46: '0', 43: 'log', 14: 'ha', 32: 'lt', 52: '6', 22: '{', 56: 'cot', 27: 'gt', 44: 'sin', 21: ']', 9: 'ayn', 5: 'zay', 40: 'Larr', 24: '+', 16: 'ya', 12: 'mim', 38: 'theta', 50: '4', 4: 'ra', 11: 'lam', 34: 'phi', 48: '2'}

# Reading data from disk

def resize(image,dim =(32,32) ) :
  old_size = image.shape[:2]

  ratio = float(dim[0])/max(old_size)
  new_size = tuple([max(int(x*ratio),1) for x in old_size])
  image = cv2.resize(image, (new_size[1], new_size[0]))
  delta_w = dim[0] - new_size[1]
  delta_h = dim[1] - new_size[0]
  top, bottom = delta_h//2, delta_h-(delta_h//2)
  left, right = delta_w//2, delta_w-(delta_w//2)
  color = [256, 256]
  new_im = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
      value=color)
  return new_im
def check_similarity(image,image_list): 
  for i in range(len(image_list)) : 
    if (image_list[i].flatten() ==image.flatten()).all() : 
      return i 
  return -1
def preprocess_image (image,dim=(32,32)):
  #_, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
  if(image.shape != dim):
      image =resize(image,dim)
  return image 
def read_data(data_directory):
  dim =(32,32)
  all_labels = {}
  image_data =[]
  image_label_num =[]
  image_label_letter=[]
  image_filename =[]
  # data
  for filedir in os.listdir(data_directory):
    folder_label = filedir.split('_')
    all_labels[int(folder_label[0])-1] =folder_label[1]
    for filename in os.listdir(data_directory+'/'+filedir):
      data_image = cv2.imread(data_directory+'/'+filedir+'/'+filename, cv2.IMREAD_GRAYSCALE)
      data_image = preprocess_image(data_image)
      image_data.append(data_image)
      image_filename.append(filename)
      image_label_num.append(folder_label[0])
      image_label_letter.append(folder_label[1])
      continue
    continue 
  image_data = np.array(image_data).reshape((-1,32,32,1))
  image_label_num = np.array(image_label_num,dtype=np.int32) -1
  image_label_letter = np.array(image_label_letter)
  image_filename = np.array(image_filename)
  return all_labels,image_data,image_label_num,image_label_letter,image_filename

# Split data and getting one hot representation of its labels

# train / test split 
def split_data(all_labels,image_data,image_label_num,image_label_letter,image_filename,percent) :
  image_train_data,image_test_data,image_train_filename,image_test_filename =train_test_split(image_data,image_filename,
                                    stratify=image_label_num,
                                    random_state = 0
                                    ,shuffle=True,
                                    test_size=percent)
  image_train_label_num,image_test_label_num,image_train_label_letter,image_test_label_letter =train_test_split(image_label_num,image_label_letter,
                                    stratify=image_label_num,
                                    random_state = 0,
                                    shuffle=True,
                                    test_size=percent)
  return image_train_data,image_test_data,image_train_label_num,image_test_label_num,image_train_label_letter,image_test_label_letter,image_train_filename,image_test_filename
def to_categorical(image_label_num,num_classes):
    return tf.keras.utils.to_categorical(image_label_num,num_classes=num_classes)

def load_model(path):
    model_to_load = tf.keras.models.load_model(r'./resources/model.h5')
    return model_to_load


# Segmentation of Equations

def crop_contour(out, mask):
    (y, x) = np.where(mask == 0)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = out[topy:bottomy+1, topx:bottomx+1]
    return out

def crop_image(path):
    im = cv2.imread(path)
    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # img = cv2.Canny(img, 100, 200)
    image_copy = im.copy()  
    # cv2.imshow(img)  
    img = ~img
    ctrs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0], reverse=True)
    crop_results = []
    rect_results = []
    for ctr in ctrs:
        mask = np.ones_like(img)
        cv2.drawContours(mask, ctr, -1, 0, thickness=10, lineType=cv2.LINE_AA)
        out = np.full(img.shape, 255)
        out[mask == 0] = ~img[mask == 0]
        out = crop_contour(out, mask)
        out = preprocess_image(out.astype('float32'))
        # cv2_imshow(out)
        x1, y1, w, h = cv2.boundingRect(ctr)
        x2 = w+x1
        y2 = h+y1      
        rect_results.append((x1,y1,x2,y2))
        crop_results.append(out)
    return np.array(crop_results),np.array(rect_results)

def labels_to_eqn(labels) : 
  equation_str = ''
  for ch in reversed(labels) :
    equation_str += str(all_labels[ch]) +' '
  return equation_str 

# Symbol Info Class

class Symbol_Info:

    def __init__(self, label, x1, y1, x2, y2):
        self._label = label
        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2
        
    @property
    def x1(self):
        return self._x1

    @x1.setter
    def x1(self, x1):
        if not isinstance(x1, int) and not isinstance(x1, float):
            raise TypeError("x1 must be set to a int, found type", type(x1))
        self._x1 = x1

    @property
    def x2(self):
        return self._x2

    @x2.setter
    def x2(self, x2):
        if not isinstance(x2, int) and not isinstance(x2, float):
            raise TypeError("x2 must be set to a int, found type", type(x2))
        self._x2 = x2

    @property
    def y1(self):
        return self._y1

    @y1.setter
    def y1(self, y1):
        if not isinstance(y1, int) and not isinstance(y1, float):
            raise TypeError("y1 must be set to a int, found type", type(y1))
        self._y1 = y1

    @property
    def y2(self):
        return self._y2

    @y2.setter
    def y2(self, y2):
        if not isinstance(y2, int) and not isinstance(y2, float) :
            raise TypeError("y2 must be set to a int, found type", type(y2))
        self._y2 = y2

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        if not isinstance(label, str) :
            raise TypeError("label must be set to a str, found type", type(label))
        self._label = label

# Educated Parsing Methods
def updateSymbol(update_list, symbol_list, j, new_label):
    new_x1 = 9223372036854775807
    new_y1 = 9223372036854775807
    new_x2= -9223372036854775807
    new_y2= -9223372036854775807
    for i in update_list :   
      new_x1 = min(i.x1,new_x1)
      new_y1 = min(i.y1,new_y1)
      new_x2 = max(i.x2,new_x2)
      new_y2 = max(i.y2,new_y2)
    symbol_list[j] = Symbol_Info(new_label, new_x1, new_y1, new_x2, new_y2)

def parseLine(symbol, symbol_list, i, remove_symbols_indexes):
    if i < len(symbol_list) - 1:    # Handle equal mark
        s1 = symbol_list[i + 1]
        if s1.label == '-' and abs(s1.x1 - symbol.x1) < 30 and abs(s1.x2 - symbol.x2) < 30:
            updateSymbol([symbol, s1], symbol_list, i, '=')
            remove_symbols_indexes.append(i + 1)
            return

    if i < len(symbol_list) - 2:    # Handle division mark
        s1 = symbol_list[i + 1]
        s2 = symbol_list[i + 2]
        if s1.label == '0' and s2.label == '0' and \
            symbol.x1 < s1.x1 < symbol.x2 and symbol.x1 < s2.x1 < symbol.x2 and \
            s1.y1 < symbol.y1 and s2.y1 > symbol.y2:
            updateSymbol([symbol, s1, s2], symbol_list, i, 'div')
            remove_symbols_indexes.append(i + 1)
            remove_symbols_indexes.append(i + 2)
            return


def parseDot(symbol, symbol_list, i, remove_symbols_indexes):
    # Todo
    pass


def parsePosNeg(symbol, symbol_list, i, remove_symbols_indexes):
    s1 = symbol_list[i + 1]
    if (symbol.label == '+' and s1.label == '-') or (symbol.label == '-' and s1.label == '+'):
        s_x_center = symbol.x1 + (symbol.x2 - symbol.x1) / 2
        s1_x_center = s1.x1 + (s1.x2 - s1.x1) / 2
        if abs(s_x_center - s1_x_center) < 30:
            updateSymbol([symbol, s1], symbol_list, i, '+-')
            remove_symbols_indexes.append(i + 1)


def educated_parse(symbol_list):
    remove_symbols_indexes = []

    for i in range(len(symbol_list)):
        symbol = symbol_list[i]

        if symbol.label == '-':   # Handle line symbols (equal, minus, division, fraction)
            parseLine(symbol, symbol_list, i, remove_symbols_indexes)

        if symbol.label == '0':   # Handle dot symbols (baa, taa, thaa, ..)
            parseDot(symbol, symbol_list, i, remove_symbols_indexes)

        if i < len(symbol_list) - 1:    # Handle +- or -+
            parsePosNeg(symbol, symbol_list, i, remove_symbols_indexes)

    for index in remove_symbols_indexes:
        symbol_list.pop(index)
    return symbol_list

# Convert to Eqn
def toEqn(symbol_list):
    variables = ['alif', 'ba', 'ta', 'tha', 'jeem', 'ha', 'kha', 'dal', 'dhal', 'ra', 'zay', 'sen', 'shin', 'sad', 'dad', 'ta', 'za', 'ayn', 'ghayn', 'fa', 'qaf', 'kaf', 'lam', 'mim', 'nun', 'ha', 'waw', 'ya']
    nums = ['0','1','2','3','4', '5', '6' , '7' , '8' , '9']
    curr_english_symbol = 'a'
    equation = ""
    mapped_symbols = {}

    i = len(symbol_list) - 1
    while i >= 0:
        symbol = symbol_list[i]
        if (i < len(symbol_list) - 1) and isUpper(symbol, symbol_list[i + 1]) and \
            (symbol_list[i].label in variables or symbol_list[i].label in nums) and \
             (symbol_list[i+1].label in variables or symbol_list[i+1].label in nums)  :    # Handle power
            
            equation += '^' + symbol_list[i].label
            i -= 1
            continue

        if symbol.label in variables:   # Normal variable, for ex: sen

            if symbol.label not in mapped_symbols:
                mapped_symbols[symbol.label] = curr_english_symbol
                curr_english_symbol = chr(ord(curr_english_symbol) + 1)
           
            if i < len(symbol_list) - 1 and symbol_list[i+1].label in nums :
              equation +='*'
            equation += mapped_symbols[symbol.label]  
            i -= 1
            continue
        elif symbol.label.isnumeric():
            curr_num = symbol.label
            i -= 1
            while (i >= 0 and symbol_list[i].label.isnumeric()):
                curr_num += symbol_list[i].label 
                i -= 1
            equation += curr_num[::-1]  # Reversed numeber, ex: "21" --> "12"
        else:
            equation += symbol.label
            i -= 1

    return equation, mapped_symbols
        
def isUpper(power, s1):
    s1_y_center = s1.y1 + (s1.y2 - s1.y1) / 2
    s1_x_center = s1.x1 + (s1.x2 - s1.x1) / 2
    return power.y2 < s1_y_center and power.x2 < s1_x_center

# Labels to Symbol_info and classify
def classify(model, images, most_probable_k  =3) :
  eqn_predictions =  model.predict(images)
  most_probable  = eqn_predictions.argsort(axis =1)[:,-most_probable_k:]
  return most_probable

def labels_to_symbols(rects,pred_labels):
  symbols = []
  length = len(pred_labels)-1
  for i in range(length+1) : 
    x1,y1,x2,y2=rects[length-i]
    symbols.append(Symbol_Info(all_labels[pred_labels[length-i]],x1,y1,x2,y2))
  return symbols

def print_labels(images,most_probable_pred):
  length = len(most_probable_pred)-1
  for i in range(length+1) : 
    print([all_labels[w] for w in most_probable_pred[length-i]][::-1])
    print('predicted :', all_labels[most_probable_pred[length-i][-1]])
    # cv2.imshow(images[length-i])
    print()

# Solve Polynomial Equation
def solve(equation, mapped_symbols):
    sp_vars = []
    for _, eng_letter in mapped_symbols.items():
        sp_vars.append(sp.Symbol(eng_letter))

    return sp.solve(sp.sympify(equation), sp_vars)
