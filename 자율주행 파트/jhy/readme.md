코드를 직접 돌려보고 싶으면 0602.py 파일을 사용!

[희연 코드]  
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
