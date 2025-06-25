[희연 코드]   
코드를 직접 돌려보고 싶으면 path3_only_autodrive.py 파일을 사용!  

### 0625
- 라이다 설정: interval time:0.3   Ypos: 1.65   Channel: 45      minimapChannel: -     max_distance: 110     lidar_position: turret
- astar 실행할 때만 라이더 값 가져오게 변경
   - 기존 코드: 0.3초마다 가져와서 수시로 맵을 갱신함.
   - 0.3초 안에 전차가 움직인다해도 어차피 주변 환경은 크게 바뀌는 것이 없으니 0.3초마다 데이터를 받아오면 중복 데이터가 심해서 필요 없다고 판단
   - 장점: 동일 장애물의 중복으로 받아오는게 줄어듬 + 전차 이동시 장애물을 훨씬 더 잘 피함
- 파일 이름 변경
  - path2_only_autodrive.py -> path3_only_autodrive.py

### 062?
- 라이다 설정: interval time:0.3   Ypos: 1.65   Channel: 45      minimapChannel: -     max_distance: 110     lidar_position: turret
- 주변 제외 나머지 구역은 초기화하는 함수 추가: clamp_range 함수, initialize_maze 함수
  - what: 전차 현 위치를 실시간으로 받아와서 그 주위를 제외하고는 맵을 초기화한다.
  - when: init 함수 3번 실행할 때 마다, 위 방식으로 초기화
  - 이렇게 한 이유: 1. 맵을 초기화를 안 하면 장애물이 꽉 찼다고 떠서 전차가 갈 길을 잃는다
                   2. 그래서 초기화를 하는 코드가 한 줄 있었는데, 라이다 데이터를 받아올 때마다 맵의 전체를 초기화해서 비효율적이라고 판단

### 0619
- 라이다 설정: interval time:0.3   Ypos: 1.65   Channel: 45      minimapChannel: -     max_distance: 110     lidar_position: turret
- info 함수에서 라이다 가져오는 조건값 수정
- info 함수와, info 함수 내에서 사용하는 함수들 전부 위로 올림.(위치만 바꿈)
- path 3로 변경
- 예시맵: 아래 사진에서 전차가 부딫힘 거의 없이 잘 감
  - 우측 상단에 돌에 살짝 스친 것 외에 잘 감
  ![image](https://github.com/user-attachments/assets/1a534303-da85-44be-8d63-98ccb4c3db91)


### 0618
- 포격팀과 통일하기로 한 라이다 설정에 맞추기
- interval time:0.3   Ypos: 1.65   Channel: 45      minimapChannel: 6     max_distance: 110     lidar_position: turret
- Ypos: 1.65에 맞춰서 가져오는 라이더 설정 변경

### 0617
- 포격팀과 통일하기로 한 라이다 설정에 맞추기
- interval time:0.3   Ypos: 1   Channel: 12      minimapChannel: 6     max_distance: 50     lidar_position: turret
- path2_only_autodrive.py 돌려볼 때 interval time 은 0.5로 일단 돌려보기는 함.

[변경된 함수]
- split_by_distance 함수 수정:
  - ![aebc8f91-59bd-40df-bae4-83d502921cb0](https://github.com/user-attachments/assets/bb5f76d5-6041-4946-9e5c-1fc90e10c01f)
  - 위 사진은 지형만 감지한 라이더의 x,z 를 그래프로 그렸을 때, 지형은 좌표가 멀리 떨어지는 경우가 많아 한 객체로 분류하기 어렵다.
  - ![image.png](attachment:07933bcb-61cc-4f69-bd0f-2ceea183881b:image.png)  
  (이렇게 분류됨)
  - vertical_angle 값으로 그룹으로 묶은 뒤 그 안에서 각 좌표의 거리기반으로 나누는 방식으로 바꿈
  - 멀리 떨어져 있는 객체는 하나의 unique 한 line_group을 못 가지게 하고 -1로
  - ![5a32d72b-a373-44a8-863c-0d438c9de415.png](attachment:2a6fa960-62ad-4f0a-9b36-e66dba31cd15:5a32d72b-a373-44a8-863c-0d438c9de415.png)   
  - 위처럼 개선됨
-  detect_obstacle_and_hill 함수 수정:
  - return문 들여쓰기. (for 문 안에 있어서 다 계산하기도 전에 함수가 종료됨)
    - 복사 붙여넣기 하는 과정에서 실수로 들여쓰기 된 것 같음... 다음에 주의
- 



---
### 0614  
- path2_only_autodrive  
- path2 기능 추가  
- 주행만 가능  
- 라이다로 장애물, 지형 구분 가능

목적지 도달 후 astar 함수 호출 횟수 출력함_ path1에서는 78이었으나 path 2에서는 42로 감소  
<img width="505" alt="스크린샷 2025-06-14 175550" src="https://github.com/user-attachments/assets/d03d2f28-fae1-42c0-8f04-61da745c090c" />  

![download](https://github.com/user-attachments/assets/c7bb2221-7cec-40e7-8e49-621a9f675396)    

---

### 0613
- lidar에서 x,z 값 실시간으로 받아서 장애물 감지
- 라이더 설정은 이렇게  
![image](https://github.com/user-attachments/assets/55c9dc3c-15cd-4f49-8870-a6515fdeb3eb)
- 이렇게 해도 돌아감.  
  ![image](https://github.com/user-attachments/assets/3db97b88-7fe4-460a-81fb-78138a917f23)  


- split_by_distance: 거리 기반으로 좌표 값 계산해서 가까운 것끼리 한 객체로 묶음.  
![image](https://github.com/user-attachments/assets/8204b0e1-fd75-49b6-8519-693b2bc32283)  
- detect_obstacle_and_hill: 각도 계산을 해서 언덕과 장애물 구분 함수  
- map_obstacle: 감지한 장애물을 맵에 반영  

- 작은 벽을 인식을 못해서 데이터 값이 작더라도 직선이면 인식하도록 바꿈.  

--- 
- path2: 2번 이동후 path 재계산하는 기능
---
### 0605
- 추가기능: (시작점 -> 목적지) 도달 시간, 이동거리, 충돌횟수 추가
  - tracking mode 누른 시점부터 시간 셈   
- 기존기능: path2 + 기홍님 휴리스틱 코드
![image](https://github.com/user-attachments/assets/4c03bdf1-6218-462d-81d1-f1dfc649bab6)  
![image](https://github.com/user-attachments/assets/a1f592ad-21b7-4fcb-b93a-0af7b07a79d0)

---
### 0604
- 기능: path2 + 기홍님 휴리스틱 코드  
- 장애물을 완만하게 잘 피해감. 그러나 실행할 때마다 결과가 약간 다르다.  
- ![download](https://github.com/user-attachments/assets/d16b9d8a-7957-435d-9a4d-9717efb6739c) 

---
### 0602 오후
- 기능: path2 + 각도 크게 틀 때 멈추기
- 각도가 크게 꺾일 때 멈췄다 가기 (diff가 30이상이면 멈췄다감)
- 근데 장애물이 있을 때는 잘 못감
- path 2 기능이랑 각도 크면 멈추기 기능은 같이 합치면 성능이 최악.
결론: 둘이 같이 못 쓴다. path2

![after](https://github.com/user-attachments/assets/dc47757b-bc28-49c1-87c5-471653aa096a)

---
### 0602 오전
- 0602 아침에 기홍님 코드를 base로 내 코드(path2)와 합친 것.
- 장애물 잘피함

![download](https://github.com/user-attachments/assets/fb23b9f5-0f6c-4c81-96b9-08974f115c67)
![download](https://github.com/user-attachments/assets/01018496-14ee-4dcd-b5db-2cfa3ae8ca26)
---
[김기홍님 코드]    
### 0602 아침
- 제일 기본 코드
- 아침에 올라온 코드로 돌려봤을때  
- 한번 이동 후 path 재계산  
![download](https://github.com/user-attachments/assets/73195860-40e0-4275-8ca1-af134ebd6b88)
>>>>>>> jhy
