import paddlex as pdx
from paddlex import transforms as T
import matplotlib.pyplot as plt

%matplotlib inline


eval_transforms = T.Compose([
    T.Resize(target_size=(1088, 1920), interp='CUBIC'), 
    T.Normalize(mean=[0.40158695, 0.43556893, 0.507324], std=[0.19307534, 0.19843009, 0.2915112])])

model = pdx.load_model('output/faster_rcnn_r50_fpn/best_model')
image_name =  'VOC/JPEGImages/260.jpeg'
result = model.predict(image_name)
pred = pdx.det.visualize(image_name, result, threshold=0.5, save_dir=None)
pred = pred[:, :, ::-1]  # 2RGB
plt.figure(figsize=(10, 10))
plt.imshow(pred)