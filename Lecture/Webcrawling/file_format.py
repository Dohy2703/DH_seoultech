msg = 1234 

# print('{}'.format(msg))
# print('|{}|'.format(msg))
print('|{:10}|'.format(msg))
print('|{:>10}|'.format(msg))
print('|{:<10}|'.format(msg))
print('|{}|'.format(msg))
print()

print('|{:0>10,}|'.format(msg)) #|000001,234|
print('|{:*>10,}|'.format(msg)) #|*****1,234
 
print()
print('- ' * 50)

import random  #난수발생 로또 적용 set셋 소트, 
import math    #수학함수 - 통계,수학지식
import time
import datetime

print('장화홍련')
time.sleep(2)
print('춘하추동')
time.sleep(2)
print('겨울나라')
time.sleep(1)

print(datetime.datetime.now()) #년 월 일 시 분 초
dt = datetime.datetime.now() 
print( dt.strftime('날짜 %Y년-%m월-%d일'))
print( dt.strftime('시간 %H시-%M분-%S초'))

print()
print('- ' * 50)
