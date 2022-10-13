import torch
import cv2
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

img = cv2.imread('test.jpg')
results = model(img)
results.crop("person")

result = results.pandas().xyxy[0].to_numpy()
result = [item for item in result if item[6]=='person']
#어떤 데이터가 오는지 몰라서 print
print(result)
print(results.xyxy[0][0][0].item())

### 1번 ###



for a,b in enumerate(result):
    tmp_img = cv2.imread('test.jpg')
    print(tmp_img.shape)
    cropped = tmp_img[int(a[0][1]):int(a[0][3]), int(a[0][0]):int(a[0][2])]
    print(cropped.shape)
    cv2.rectangle(tmp_img, (int([b][0][0].item()), int([b][0][1].item())), (int(results.xyxy[b][0][2].item()), int(results.xyxy[b][0][3].item())), (255,255,255))
    cv2.imwrite('result.png', tmp_img)





