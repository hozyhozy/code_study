# 코테 정리
- 주어진 변수로 무조건 규칙성을 찾아야함
- runtime 줄일 때는 무조건 값을 딱 정해주면 속도 빨라짐 
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

