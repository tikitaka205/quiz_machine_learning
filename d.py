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

#의도한대로 사람마다의 직사각형과 사람마다의 사진이 나왔다
#하지만 이중 for문의 시간복잡도 그리고 더 좋은 코드가 있을 것 같다.
for a,b in enumerate(result):
    tmp_img = cv2.imread('test.jpg')
    cv2.rectangle(tmp_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0,255,255))
    cv2.imwrite(f"{a}.png", tmp_img) 
    for c in range(int(a)+1):
        cropped = tmp_img[int(b[1]):int(b[3]), int(b[0]):int(b[2])] #행, 열
        cv2.imwrite(f"crop{a}.png", cropped)
"""
    tmp_img for문 밖에 놓으면 사진 하나 늘어날때 마다 직사각형이 하나씩 늘어난다. 직사각형 그린 사진을 또 사용하는 느낌
    cropped가 또 설정을 다시해서 잘리는건지 선을 보고 자르는건지
    선을 보고 자르면 사진 하나씩 넣어줘야하고 사진인식 아니다?..
    결국 사진 하나하나 넣어줘야 할 것 같은데?
    네모를 안그려도 잘라준다.
    이중 for문 시간복잡도 좋지않다
"""




