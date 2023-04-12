# Gau Gau

## Download 

yolov3.weights : https://pjreddie.com/media/files/yolov3.weights

## Initialize virtual environment on the commmand line
Create virtual envirnoment
```
py -m venv .venv
```
Upgarde pip
```
py -m pip install --upgrade pip
```
Set permission for program
```
Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser
```
Install packages from requirements.txt
```
pip install -r .\requirements.txt
```
Freeze 
```
python -m pip freeze > requirements.txt
```

## Installaion
Tensorflow
```
pip install tensorflow
```

## Some problems:
* Problem 1: ModuleNotFoundError: No module named 'keras.layers.merge'.

keras.layers.merge -> keras.layers

* Problem 2: ImportError: cannot import name 'load_img' from 'keras.preprocessing.image'

* Problem 3: ImportError: Could not import PIL.Image. The use of `load_img` requires PIL.

* Problem 4: Slowly

* Problem 5: RecursionError: maximum recursion depth exceeded while calling a Python object

Init Tk
```
tk.Tk.__init__(self, *arg, **kwargs)
```

## Reference:
https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/

https://github.com/experiencor/keras-yolo3