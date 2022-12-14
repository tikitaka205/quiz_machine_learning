import torch
import cv2
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
imgs = ['https://teamsparta.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F396579ac-a81f-4073-9409-248ba51bc5ea%2FUntitled.jpeg?table=block&id=e70a6c23-18cd-4bc6-8c46-b585967dfd56&spaceId=83c75a39-3aba-4ba4-a792-7aefe4b07895&width=1470&userId=&cache=v2']  # batch of images

results = model(imgs)

print(results.xyxy[0], results.xyxy[0][0][0].item())  # img1 predictions (tensor)
print(results.pandas().xyxy[0])  # img1 predictions (pandas)

tmp_img = cv2.imread('test.jpg')
cv2.rectangle(tmp_img, (int(results.xyxy[0][0][0].item()), int(results.xyxy[0][0][1].item())), (int(results.xyxy[0][0][2].item()), int(results.xyxy[0][0][3].item())), (0,0,255))
print(results.xyxy[0][0][0].item())
print(results.xyxy[0][0][1].item())

cv2.imwrite('result.png', tmp_img)