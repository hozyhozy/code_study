# 코테 정리
- 거의 대부분 주어진 변수로 규칙성을 찾아야함
- runtime 줄일 때는 무조건 값을 딱 정해주면 속도 빨라짐 / input 값이 많다면 종류별로 분리해서 저장해주는 것도 방법  
ex. set(range(5)) --> {0,1,2,3,4}
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

# 조합 구하기 
- 들어오는 값을 2개 조합으로 만들어라 
- 각 변수는 값 하나하나를 가지고 옴.. for 구문을 2번 쓸 필요 X
from itertools import combinations
for a, b in combinations(x, 2):
  a,b,c=f1
  c,d,f=f2
