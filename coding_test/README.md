# 코테 정리
- 거의 대부분 주어진 변수로 규칙성을 찾아야함
- 그냥 뭘 더 생각하지말고 문제 숙지잘하고 문제 따라가면 답이 있음

### 문제를 천천히 한줄씩 한줄씩 해독하는게 더 편할때가 있음
- 진짜 안풀리는 경우에는 한줄씩 한단계씩 생각하면서 풀기
- 뭔가 갯수를 세야하는 문제는 거의 70% 변수=0 또는 1에서 업데이트해야함. for문
- list를 str로 풀어줄때는 ''.join(a) for a in answer

### runtime 줄이기
- runtime 줄일 때는 무조건 값을 딱 정해주면 속도 빨라짐 / input 값이 많다면 종류별로 분리해서 저장해주는 것도 방법  
ex. set(range(5)) --> {0,1,2,3,4}
- list보다는 dictionary로 풀기

### 구현
- 벌집은 6배씩 커진다
ex.num+=6*count

### set
1) remove vs. discard(예외처리가능)
2) 빈 set 만들때 S.clear | S = set()
3) add vs. update(다수의 값이 input으로 들어올떄 ex. list)

### defaultdict
1) dictionary랑 비슷하지만 없는 key값도 불러올 수 있음 (없는 key 값은 0)
2) from collections import defaultdict
3) 문자의 개수 셀 때 유용함!

### 조합 구하기 
- 들어오는 값을 2개 조합으로 만들어라 
- 각 변수는 값 하나하나를 가지고 옴.. for 구문을 2번 쓸 필요 X
'''from itertools import combinations
for a, b in combinations(x, 2):
  a,b,c=f1
  c,d,f=f2'''

### 정렬문제
- list 정렬은 [].sort(reverse=True): 내림차순 // 값이 2개라면 튜플로 묶어서 append + [].sort(key=lambda x: x[0])
- dict 정렬은 sorted(my_dict.items()) # key 기준 오름차순
- dict sorted(my_dict.items(), key=lambda item=item[0], reverse=True) #내림차순 여기서는 꼭 key값을 정해주자!!
- 데이터 프레임 정렬은 df.sort_values('column')
