# How to use this project?
## 1) Create the environment

```sh
conda env create -f environment.yml
```

## 2) Activate the environment
```sh
conda activate hand-env
```

## 3) Select a camera
Change the camera index at line 336 in `steering_wheel.py` to chose a different camera:
```python
cap = cv2.VideoCapture(0)  # Put 1 for external webcam
```

## 3) Run the script
```sh
python steering_wheel.py
```
