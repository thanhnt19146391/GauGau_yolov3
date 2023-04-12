""" 
load yolov3 model and perform object detection
based on https://github.com/experiencor/keras-yolo3
"""
import numpy as np
from numpy import expand_dims
from tensorflow import keras
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from keras.models import load_model
from keras.utils import load_img
from keras.utils import img_to_array

from matplotlib import pyplot
from matplotlib.patches import Rectangle
 
class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1
    
    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        
        return self.score
    
def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    boxes = []  
    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh
    
    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            
            if(objectness.all() <= obj_thresh): continue
            
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            
            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
            
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            
            boxes.append(box)
    return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h

        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    
    intersect = intersect_w * intersect_h
    
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
   
    return float(intersect) / union

def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].classes[c] == 0: continue
            
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

# load and prepare an image
def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename)
    print(f'Type of input image: {type(image)}')
    
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape)
    print(f'Type of image after load: {type(image)}')

    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height

def ThanhNT_load_image_pixels(mat, shape):
    image = Image.fromarray(mat)
    print(f'Type of input image: {type(image)}')

    # Original size
    width, height = image.size
    
    # resize to required size
    resized_image = image.resize(shape)
    
    # convert to numpy array
    image = img_to_array(resized_image)

    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    
    return image, width, height

# My custom function load image from video
# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
    # enumerate all possible labels
        for i in range(len(labels)):
        # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
                # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores

# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='white')
        # draw the box
        ax.add_patch(rect)
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        pyplot.text(x1, y1, label, color='white')
    # show the plot
    pyplot.show()

def ThanhNT_draw_boxes(src, v_boxes, v_labels, v_scores):
    mat = src.copy()
    
    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        
        # create the shape
        mat = cv2.rectangle(img = mat, pt1 = (x1, y1), pt2 = (x2, y2), color = (255, 255, 255), thickness = 1)
        
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        mat = cv2.putText(
            img = mat, 
            text = label, 
            org = (x1, y1), 
            fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale = 1, 
            color = (255, 255, 255), 
            thickness = 1)

    return mat

def Experiencor_main():
    # load yolov3 model
    model = load_model('model.h5')
    
    # define the expected input shape for the model
    input_w, input_h = 416, 416
    
    # define our new photo
    photo_filename = 'The_million_march_man.jpg'
    
    # load and prepare image
    image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))
    
    # make prediction
    yhat = model.predict(image)
    
    # summarize the shape of the list of arrays
    print([a.shape for a in yhat])
    
    # define the anchors
    anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
    
    # define the probability threshold for detected objects
    class_threshold = 0.6

    boxes = list()
    for i in range(len(yhat)):
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
    # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
    # suppress non-maximal boxes
    do_nms(boxes, 0.5)
    # define the labels
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    # get the details of the detected objects
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
    # summarize what we found
    for i in range(len(v_boxes)):
        print(v_labels[i], v_scores[i])
    # draw what we found
    draw_boxes(photo_filename, v_boxes, v_labels, v_scores)

class ThanhNT_GUI(tk.Tk):
    def __init__(self, frames = None, *arg, **kwargs):
        """ Backend """
        # load yolov3 model
        self.model = load_model('model.h5')

        # define the expected input shape for the model
        self.input_w, self.input_h = 416, 416
        self.input_shape = (self.input_w, self.input_h)

        # define the labels
        self.labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
        "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

        # define the anchors
        self.anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
        
        # define the probability threshold for detected objects
        self.class_threshold = 0.6
                
        self.frames = frames

        self.max_index = len(frames)

        self.frame_index = -1

        """ Frontend """
        # Init Tk
        tk.Tk.__init__(self, *arg, **kwargs)

        # Configure window size as maximum
        self.state('zoomed')

        self.frame_label = tk.Label(
            master = self
        )
        self.frame_label.grid(column = 0, row = 0)

        self.update_frame()

    def createPhotoImage(self, mat):
        img = Image.fromarray(mat)
        w, h = img.size
        dst = ImageTk.PhotoImage(image = img)
        return dst

        
    def update_frame(self):
        # self.frame_index = (self.frame_index + 1) % self.max_index
        self.frame_index = (self.frame_index + 1) % 10
        print(f'Frame {self.frame_index}')
        frame = self.frames[self.frame_index]

        # load and prepare image
        image, image_w, image_h = ThanhNT_load_image_pixels(mat = frame, shape = self.input_shape)

        # make prediction
        yhat = self.model.predict(image)

        # summarize the shape of the list of arrays
        print([a.shape for a in yhat])

        boxes = list()
        for i in range(len(yhat)):
            # decode the output of the network
            boxes += decode_netout(yhat[i][0], self.anchors[i], self.class_threshold, self.input_h, self.input_w)
        
        # correct the sizes of the bounding boxes for the shape of the image
        correct_yolo_boxes(boxes, image_h, image_w, self.input_h, self.input_w)
        
        # suppress non-maximal boxes
        do_nms(boxes, 0.5)
        
        # get the details of the detected objects
        v_boxes, v_labels, v_scores = get_boxes(boxes, self.labels, self.class_threshold)

        # draw what we found
        dst_frame = ThanhNT_draw_boxes(frame, v_boxes, v_labels, v_scores) 
        
        self.photoImage = self.createPhotoImage(dst_frame)
        self.frame_label.config(image = self.photoImage)
        self.after(ms = 1000, func = self.update_frame)

    


def ThanhNT_main():
    """
    ThanhNT: load video and detect realtime
    """

    

    # define our new video
    video_filename = 'elderly.mp4'

    cap = cv2.VideoCapture(video_filename)
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret: 
           frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           frames.append(frame) 
        else:
            break

    print(f'Number of frame: {len(frames)}')

    app = ThanhNT_GUI(frames = frames)
    app.mainloop()
    

    

if __name__ == '__main__':
    # Experiencor_main()
    ThanhNT_main()
   

    