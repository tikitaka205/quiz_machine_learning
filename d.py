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

### 4번 ### 결과 : 모든사람에게 직사각형 사진, 직사각형 표시된 사진이 없어도 crop할 수 있다. 깨끗한 사람 사진



tmp_img = cv2.imread('test.jpg')
tmp_img2 = cv2.imread('test.jpg')

for c,d in enumerate(result):
    cropped = tmp_img[int(d[1]):int(d[3]), int(d[0]):int(d[2])] #행, 열
    cv2.imwrite(f"crop{c}.png", cropped)

for a,b in enumerate(result):
    cv2.rectangle(tmp_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0,255,255))
    cv2.imwrite("rectangle.png", tmp_img) #따로 저장 완료
"""
    cropped가 또 설정을 다시해서 잘리는건지 선을 보고 자르는건지
    선을 보고 자르면 사진 하나씩 넣어줘야하고 사진인식 아니다?.. 사진 직사각형 인식이 아니다 다른 개념이다.
    결국 사진 하나하나 넣어줘야 할 것 같은데? 사진 하나하나 넣기보다는 사진을 가지고
    네모를 안그려도 잘라준다.
    for문 순서가 바뀌면 또 rectangle 있는 사진이 crop된다 이걸해결하기위해 변수를 두개로 해도 괜찮을 듯
    rec 함수안에 넣고 하면 사진 하나당 직사각형이 하나씩 추가가 된다.
    a,b마지막에 사진이름을 f스트링을 사용해서 다 이름을 바꿔주면 직사각형 하나씩 된 사진 여러개를 저장한다. 하나로 저장하면 데이터가 쌓여서 마지막 한개의 사진이 저장된다
"""





