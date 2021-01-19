# 🧏🏻‍♀️Everybody dance now (with sign)

## 데이터셋
  * [수어데이터셋 - AI hub](https://www.aihub.or.kr/aidata/7965)
  * <img src = https://user-images.githubusercontent.com/28949182/104993948-019b0500-5a67-11eb-85b9-b7e878a200d0.png width='70%'></img><br/>

* * *

## 환경
  ```
  scipy==1.2.0
  tensorflow==2.4.0
  cuda_11.0
  ! pip install dominate 
  ```
* * *

## 코드 실행 방법
  ### 1. dataset 만들기 
  ```
  # train dataset 만들기
  ! python graph_train.py \
  --keypoints_dir '/workspace/Image-team1/my_data/json' \
  --frames_dir '/workspace/Image-team1/my_data/sign' \
  --save_dir '/workspace/Image-team1/my_data/save' \
  --spread 1 4000 1 \
  --facetexts
  ```

 ``` 
 #test dataset 만들기
  ! python graph_avesmooth.py \
  --keypoints_dir '/workspace/Image-team1/my_data/json' \
  --frames_dir '/workspace/Image-team1/my_data/sign' \
  --save_dir '/workspace/Image-team1/my_data/save_test' \
  --spread 4000 4996 1 \
  --facetexts
  ```
* * *

  ### 2. Training
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
  
  ```
  # Followed by a "local" stage model with 1024x512 resolution.
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
  ### 3. Testing
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
  
## 참고사항
  *  오류 목록
  * ! python graph_train.py 실행시 다음과 같은 부분이 포함된 파일은 제거하여야함.
      1. "hand_right_keypoints_2d":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
      2. "hand_left_keypoints_2d":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 

* * *
## 그외 코드 실행 방법은
   *  https://github.com/carolineec/EverybodyDanceNow 이곳을 참고
   
* * *
## 모델 구조
<img src=https://user-images.githubusercontent.com/28949182/104292626-66eb7500-5500-11eb-9faf-eb6f698e3b03.png width="70%"></img><br/> 
<img src=https://user-images.githubusercontent.com/28949182/104294419-7b307180-5502-11eb-8245-46366faf9b48.png width="70%"></img><br/> 

* * *
## result
 <img src =https://user-images.githubusercontent.com/28949182/104993955-065fb900-5a67-11eb-8f8f-7c63a30527dc.png width='70%'></img>

* * * 

## more
  * [hand to hand homepage](https://hand-to-hand.kro.kr/public/index.html)
 
