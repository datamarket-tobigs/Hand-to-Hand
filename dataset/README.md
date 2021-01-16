## TFRecord 생성하기_create_tfrecords.py
### 데이터 준비
```
data
|   │
|   ├── Image
|   │     └── 영상 별 Image 폴더
|   │           └── .png
|   └── label.txt
```
#### 1. 이미지 폴더  
![image](https://user-images.githubusercontent.com/39791467/104814357-c2e03180-5851-11eb-929f-7119dfafc4ed.png)  
![image](https://user-images.githubusercontent.com/39791467/104814367-d68b9800-5851-11eb-92c8-8265d6d7b618.png)
#### 2. 한 줄씩 이미지 keypoint 저장한 keypoint txt  
![image](https://user-images.githubusercontent.com/39791467/104814459-531e7680-5852-11eb-8356-07a0558e9143.png)
#### 3. 이미지 개수와 label.txt의 줄 수가 같아야 함  

### TFRecord 생성 후
```
data
|   │
|   ├── Image
|   │     └── 영상 별 Image 폴더
|   │           └── .png
|   ├── label.txt
|   ├── tfrecords
|   │     └── .tfrecord
```

## TFRecord 읽기_get_tfrecords.py
### 데이터 확인하기
아래와 같이 이미지와 이미지 위에 keypoint를 찍은 데이터 3개를 확인해 볼 수 있다.  
![image](https://user-images.githubusercontent.com/39791467/104814584-1f901c00-5853-11eb-897c-f2bef03e4818.png)

## 데이터 출처
[AI Hub 수어 영상](https://www.aihub.or.kr/aidata/7965)
