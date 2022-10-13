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

### 2번 ### cv.imwrite에서 f스트링을 안쓰니까 사진이 한장이 나왔다.
#모든 사람의 crop된 사진이 나왔지만 이미 rectangle이 된 사진으로 crop을 해서 사각형들이 보인다.



tmp_img = cv2.imread('test.jpg')
for a in (result):
    cropped = tmp_img[int(a[1]):int(a[3]), int(a[0]):int(a[2])]
    cv2.imwrite('result.png', cropped)
    cv2.rectangle(tmp_img, (int(a[0]), int(a[1])), (int(a[2]), int(a[3])), (0,255,255))


    cropped = tmp_img[int(a[1]):int(a[3]), int(a[0]):int(a[2])]
    cv2.imwrite(f"{a}.png", cropped)
cv2.imwrite('result.png', tmp_img)





