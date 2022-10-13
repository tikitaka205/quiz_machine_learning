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

###5번###



tmp_img = cv2.imread('test.jpg')
tmp_img2 = cv2.imread('test.jpg')

for c,d in enumerate(result):
    cropped = tmp_img[int(d[1]):int(d[3]), int(d[0]):int(d[2])] #행, 열
    cv2.imwrite(f"crop{c}.png", cropped) #자른걸 이미지 다시 불러주겠다.
    cv2.rectangle(tmp_img2, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0,255,255))
cv2.imwrite('rectangle.png', tmp_img2)

"""
float값 소수 int는 실수다. 실수로 해야 인식을 한다.
직사각형 그려진걸 다시 가져와서 자르기때문에 위아래 순서도 중요하고
원본을 가져와서 직사각형 그려준다거ㅏ 바뀐데이터의 변수를 저장해주는 거다.
포문안에 cv2.imwrite("rectangle.png", tmp_img2) 넣으면 다섯번이나 진행된다. for문에서 빼서 불필요한 일을 하지말자
드디어 마무리.
"""





