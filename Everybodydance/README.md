# Everybody dance now 🧏🏻‍♀️

## 1. Getting Started
- tensorflow==2.4
- cuda_11.0
- scipy==1.2.0
- ! pip install dominate 

* * *

## 2. Dataset
- [수어 영상 데이터](https://www.aihub.or.kr/aidata/7965) 를 활용함.
> <img src='https://user-images.githubusercontent.com/28949182/105356632-fc8cb000-5c36-11eb-89ba-d90fcaee96c2.png' width='70%'/> 

- 데이터는 사전 전처리 과정을 거쳐 이미지와 json쌍으로 mapping 시킴. [참고](https://github.com/Tobigs-team/hand-to-hand/tree/master/preprocessing)
> <img src='https://user-images.githubusercontent.com/28949182/105357856-c3553f80-5c38-11eb-8c2c-872700ceedf3.png' width='70%'/>
* * *

## 3. Dataset preparation
### Train dataset
```
# train dataset 만들기
! python graph_train.py \
--keypoints_dir '/workspace/Image-team1/my_data/json' \
--frames_dir '/workspace/Image-team1/my_data/sign' \
--save_dir '/workspace/Image-team1/my_data/save' \
--spread 3502 4000 1 \
--facetexts
```

### Test dataset
```
#test dataset 만들기
! python graph_avesmooth.py \
--keypoints_dir '/workspace/Image-team1/my_data/json' \
--frames_dir '/workspace/Image-team1/my_data/sign' \
--save_dir '/workspace/Image-team1/my_data/save_test' \
--spread 4000 4996 1 \
--facetexts
```

> 참고사항 <br/>
 json 파일에 "hand_left_keypoints_2d":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 이런부분은 없어야 함.
* * *

## 4. Training
### global stage
```
# train a model at 512x256 resolution
! python train_fullts.py \
--name 'MY_model_Dance_global' \
--dataroot '/workspace/Image-team1/my_data/save' \
--checkpoints_dir '/workspace/Image-team1/my_data/checkpoint' \
--loadSize 512 \
--no_instance \
--no_flip \
--tf_log \
--label_nc 6
```

### local stage
```
! python train_fullts.py \
--name 'MY_model_Dance_local' \
--dataroot '/workspace/Image-team1/my_data/save' \
--checkpoints_dir '/workspace/Image-team1/my_data/checkpoint' \
--load_pretrain '/workspace/Image-team1/my_data/checkpoint/MY_model_Dance_local' \
--netG local \
--ngf 32 \
--num_D 3 \
--resize_or_crop none \
--no_instance \
--no_flip \
--tf_log \
--label_nc 6
```

### face GAN
```
# face GAN
!python train_fullts.py \
--name 'MY_model_Dance_face2' \
--dataroot '/workspace/Image-team1/my_data/save'  \
--load_pretrain '/workspace/Image-team1/my_data/checkpoint/MY_model_Dance_face' \
--checkpoints_dir '/workspace/Image-team1/my_data/checkpoint' \
--face_discrim \
--face_generator \
--faceGtype global \
--niter_fix_main 10 \
--netG local \
--ngf 32 \
--num_D 3 \
--resize_or_crop none \
--no_instance \
--no_flip \
--tf_log \
--label_nc 6
```
* * *
## 5. Testing
### global stage 
```
# test model at 512x256 resolution
! python test_fullts.py \
--name 'MY_model_Dance_global' \
--dataroot '/workspace/Image-team1/my_data/save_test' \
--checkpoints_dir '/workspace/Image-team1/my_data/checkpoint'\
--results_dir '/workspace/Image-team1/my_data/save_test/result' \
--loadSize 512 \
--no_instance \
--how_many 10000 \
--label_nc 6
```

### local stage
```
# test model at 1024x512 resolution
! python test_fullts.py \
--name 'MY_model_Dance_local' \
--dataroot '/workspace/Image-team1/my_data/save_test' \
--checkpoints_dir '/workspace/Image-team1/my_data/checkpoint' \
--results_dir '/workspace/Image-team1/my_data/save_test/result' \
--netG local \
--ngf 32 \
--resize_or_crop none \
--no_instance \
--how_many 10000 \
--label_nc 6
```

### face stage
```
# test model at 1024x512 resolution with face GAN
!python test_fullts.py \
--name 'MY_model_Dance_face' \
--dataroot '/workspace/Image-team1/my_data/save_test' \
--checkpoints_dir '/workspace/Image-team1/my_data/checkpoint'\
--results_dir '/workspace/Image-team1/my_data/save_test/result' \
--face_generator \
--faceGtype global \
--netG local \
--ngf 32 \
--resize_or_crop none \
--no_instance \
--how_many 10000 \
--label_nc 6
```

* * *

## 6. Result
<img src ='https://user-images.githubusercontent.com/28949182/105358394-76259d80-5c39-11eb-8f5d-47d7d62409a6.png' width='70%'></img>

* * *
## 7. Reference
- https://github.com/carolineec/EverybodyDanceNow 
* * *
## 8. More
- [Web Page](https://hand-to-hand.kro.kr/public/index.html)

