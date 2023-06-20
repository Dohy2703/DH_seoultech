'''
To use font, you should download it
albumentation : https://github.com/albumentations-team/albumentations.git

* you should change these parameters *

  from CapstoneDesign_0_booknames import booklist   # you can change book names

  bookcover = glob.glob('{your directory which have images}/*.jpg')
  bground = glob.glob('{your directory which have background images}/*.jpg')

  save_path_images = '{your directory to save image}'
  save_path_labels = '{your directory to save label}'
  
  font = '{set font}'
'''

import cv2
import random as rd
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import albumentations as A
import math
import glob
from CapstoneDesign_0_booknames import booklist

book_list = booklist(show=False) # Book name list

bookcover = glob.glob('/home/kdh/Desktop/224x224/*.jpg')
bground = glob.glob('/home/kdh/Desktop/background/*.jpg')
font = ImageFont.truetype("./malgun.ttf", 24)  # font

background = cv2.imread(bground[rd.randint(0, len(bground)-1)])
background = cv2.resize(background,(640,640))

book_center_point = []
book_side_points = []

label = background.copy()

aug_cnt = 0
kor_list = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
eng_list_s = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
              'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

str1, str2, str3, str4, str5 = '', '', '','',''
center_point = []
side_points = []

# Random generate
def mk_rd_num():
    return rd.randint(0, 9)

def mk_rd_kor():
    global kor_list
    return kor_list[rd.randint(0, len(kor_list) - 1)]

def mk_rd_eng_s():
    global eng_list_s
    return eng_list_s[rd.randint(0, len(eng_list_s) - 1)]

def mk_rd_eng_l():
    global eng_list_s
    return eng_list_s[rd.randint(0, len(eng_list_s) - 1)].upper()

def mk_rd_100():
    return rd.randint(0, 100)

def mk_rd_255():
    return rd.randint(0, 255)

def label_color():  # set booklabel paper color
    bg_col = [0, 0, 0]
    for i in range(len(bg_col)) :
        bg_col[i] = (255 - rd.randint(0, 50)) * rd.randint(0, 1)
    return bg_col

def make_background(show=False) : # generate background
    global background
    # def rd_num_255():
    #     return rd.randint(0, 255)
    # background = np.full((640, 640, 3), (rd_num_255(), rd_num_255(), rd_num_255()), np.uint8)
    background = cv2.imread(bground[rd.randint(0, len(bground) - 1)])
    background = cv2.resize(background, (640, 640))
    if show :
        cv2.imshow('bg', background)

def make_label(draw_center=False, draw_side=False, print_word = False, show=False) :
    global background
    global label
    global font
    global center_point
    global side_points

    font = ImageFont.truetype("./malgun.ttf", 24)  # font
    label = background.copy()
    str1 = str(mk_rd_num())+str(mk_rd_num())+str(mk_rd_num())+'.'+str(mk_rd_num())+str(mk_rd_num())
    str2 = str(mk_rd_kor())+str(mk_rd_num())+str(mk_rd_num())+str(mk_rd_kor())
    str3 = str(mk_rd_eng_s())+'.'+str(mk_rd_num())
    str4 = str(mk_rd_eng_l())+str(mk_rd_eng_l())+str(mk_rd_num())+str(mk_rd_num())+str(mk_rd_num())+str(mk_rd_num()) \
        +str(mk_rd_num())+str(mk_rd_num())
    str5 = str(rd.randint(1, 9))+'0'+'0'

    bg_color = 200  # bookcover color
    init_p = [100, 100]  # left upper initial position

    label[init_p[0] + 0:init_p[0] + 200, init_p[1] + 0:init_p[1] + 200, :] = bg_color    # 흰색 테두리 색 조금 실제처럼 만들기

    bg_col = label_color()  # 라벨 밑면 색 만들기
    for i in range(3) :     # 라벨 밑면 색 만들기( ex) 100 테두리 )
        label[init_p[0] + 200:init_p[0] + 300, init_p[1] + 0:init_p[1] + 200, i] = bg_col[i]

    label_pil = Image.fromarray(label)  # 한글 사용을 위해 PILLOW 라이브러리 이용
    draw = ImageDraw.Draw(label_pil)

    init_x = 30; init_y = 30   # 라벨에서부터 글자가 떨어진 거리
    draw.text((init_p[0] + init_x+20, init_p[1] + init_y), str1, font=font, fill='#000')
    draw.text((init_p[0] + init_x+20, init_p[1] + 25+init_y), str2, font=font, fill='#000')

    if rd.randint(0, 9)>=8 :   # 확률적으로 3번째 줄에 c.2 이런거 출력
        draw.text((init_p[0] + init_x+20, init_p[1] + 50+init_y), str3, font=font, fill='#000')
    draw.text((init_p[0] + init_x+10, init_p[1] + 120+init_y), str4, font=font, fill='#000')
    font = ImageFont.truetype("./malgun.ttf", 70)
    # draw.text((init_x+20, 170+init_y), str5, font=font, fill='#FFF', stroke_width=1)

    rand_num = rd.randint(1, 9)
    draw.text((init_p[0] + init_x, init_p[1] + 165+init_y), str(rand_num), font=font,
              fill=(bg_color, bg_color, bg_color), stroke_width=1)
    draw.text((init_p[0] + init_x+50, init_p[1] + 165+init_y), '0', font=font,
              fill=(bg_color, bg_color, bg_color), stroke_width=1)
    draw.text((init_p[0] + init_x+100, init_p[1] + 165+init_y), '0', font=font,
              fill=(bg_color, bg_color, bg_color), stroke_width=1)

    label = np.array(label_pil)

    center_point = [init_p[0]+100, init_p[1]+150]
    side_points = [[init_p[1]+200, init_p[0]], [init_p[1], init_p[0]], [init_p[1], init_p[0]+300], [init_p[1]+200, init_p[0]+300]]
    # 오른쪽 위 [x1, y1], 왼쪽 위 [x2, y2], 왼쪽 아래 [x3, y3], 오른쪽 아래 [x4, y4]


    if draw_center : # 센터 포인트 표시
        cv2.circle(label, (init_p[0]+100, init_p[1]+150), 5, (0,0,255), -1)
    if draw_side : # 사이드 포인트 표시
        cv2.circle(label, (init_p[1], init_p[0]), 5, (0,0,255), -1)
        cv2.circle(label, (init_p[1]+200, init_p[0]), 5, (0, 0, 255), -1)
        cv2.circle(label, (init_p[1], init_p[0]+300), 5, (0, 0, 255), -1)
        cv2.circle(label, (init_p[1]+200, init_p[0]+300), 5, (0, 0, 255), -1)
    if print_word :  # 글자 표시
        print(str1)
        print(str2)
        print(str3)
        print(str4)
        print(str5)
    if show :
        cv2.imshow('label', label)

# def make_book(draw_points=False) :
#     global aug_cnt
#     global label
#     global side_points
#     global center_point
#
#     print(side_points[0][0])
#     book_points = [[side_points[0][0]-5 , side_points[0][1]+50],
#                    [side_points[1][0]-5 , side_points[1][1]-200],
#                    [side_points[2][0]+5 , side_points[2][1]-200],
#                    [side_points[3][0]+5 , side_points[3][1]+50]]
#
#     label[book_points[1][1]:book_points[3][1], book_points[1][0]:book_points[2][0], :] = 0
#     print(book_points[1][1], book_points[3][1])
#     print(book_points[1][0], book_points[2][0])
#
#     if draw_points :
#         cv2.circle(label, (book_points[0][0], book_points[0][1]), 5, (255, 0, 0), -1)  # 오른쪽 위에 점
#         cv2.circle(label, (book_points[1][0], book_points[1][1]), 5, (255, 0, 0), -1)  # 왼쪽 위에 점
#         cv2.circle(label, (book_points[2][0], book_points[2][1]), 5, (255, 0, 0), -1)  # 왼쪽 아래 점
#         cv2.circle(label, (book_points[3][0], book_points[3][1]), 5, (255, 0, 0), -1)  # 오른쪽 아래 점
#
#     cv2.imshow('book', label)


def change_size(show=False, apply=True, draw_circle=False, resize_x=100, draw_book=False) :  # 1-2. 크기 조절
    global aug_cnt
    global label
    global background
    global side_points
    global center_point
    global bookcover
    global book_side_points
    global book_center_point
    global book_list

    aug_cnt += 1  # aug_cnt + 1

    init_p2 = [400+rd.randint(0,50), 300+rd.randint(-200, 200)]

    label_rs = background.copy()

    # resize_x = 100
    resize_y = int(resize_x*1.5)
    label_size = cv2.resize(label[side_points[1][0]:side_points[3][1], side_points[1][1]:side_points[3][0]], (resize_x, resize_y))

    side_points = [[init_p2[1]+0, init_p2[0]+resize_y], [init_p2[1]+0, init_p2[0]+0],
                      [init_p2[1]+resize_x, init_p2[0]+0], [init_p2[1]+resize_x, init_p2[0]+resize_y]]
    center_point = [init_p2[1] + int(resize_x/2), init_p2[0] + int(resize_y/2)]

    book_upper_h = int(resize_x*4)+rd.randint(0,100)
    book_lower_h = int(resize_x*3/5)+rd.randint(0,10)
    book_width = rd.randint(0,10)  # +int(resize_x/10)

    book_side_points = [[side_points[0][0]-book_width , side_points[0][1]+book_lower_h],
                   [side_points[1][0]-book_width , side_points[1][1]-book_upper_h],
                   [side_points[2][0]+book_width , side_points[2][1]-book_upper_h],
                   [side_points[3][0]+book_width , side_points[3][1]+book_lower_h]]

    # 단색 채우기
    # label_rs[book_points[1][1]:book_points[3][1], book_points[1][0]:book_points[2][0], 0] = mk_rd_255()
    # label_rs[book_points[1][1]:book_points[3][1], book_points[1][0]:book_points[2][0], 1] = mk_rd_255()
    # label_rs[book_points[1][1]:book_points[3][1], book_points[1][0]:book_points[2][0], 2] = mk_rd_255()

    bookcover_ = cv2.imread(bookcover[rd.randint(0, len(bookcover)-1)])
    bookcover_ = cv2.resize(bookcover_, (640, 640))
    label_rs[book_side_points[1][1]:book_side_points[3][1], book_side_points[1][0]:book_side_points[2][0], :] = \
        bookcover_[book_side_points[1][1]:book_side_points[3][1], book_side_points[1][0]:book_side_points[2][0], :] # 책 표지 붙이기

    # 책에 글자 쓰기
    label_pil = Image.fromarray(label_rs)  # 한글 사용을 위해 PILLOW 라이브러리 이용
    draw = ImageDraw.Draw(label_pil)

    init_x = int(resize_y/5)-book_width; init_y = int(resize_y/10)   # 라벨에서부터 글자가 떨어진 거리
    font_size = int(resize_x*5/10)

    font_rd = rd.randint(0,9)
    if font_rd <= 2 :
        font = ImageFont.truetype("./malgun.ttf", font_size)
    elif font_rd <=6 :
        font = ImageFont.truetype("./nanum/NanumBarunGothic.ttf", font_size)
    else :
        font = ImageFont.truetype("./nanum/NanumGothicCoding-Bold.ttf", font_size)
    black_or_white = rd.randint(0,1)

    # print(int(sum((bookcover_[50][0])+sum(bookcover_[60][0]))/2))

    if int(sum((bookcover_[0][0])+sum(bookcover_[10][0]))/2)>=600 or int(sum((bookcover_[50][0])+sum(bookcover_[60][0]))/2)>600:
        fill_ = '#'+str(rd.randint(0,3))+str(rd.randint(0,3))+str(rd.randint(0,3))
    elif int(sum((bookcover_[0][0])+sum(bookcover_[10][0]))/2)<100 or int(sum((bookcover_[50][0])+sum(bookcover_[60][0]))/2)<100:
        fill_ = '#' + str(rd.randint(7, 9)) + str(rd.randint(7, 9)) + str(rd.randint(7, 9))
    elif black_or_white == 0:
        fill_ = '#' + str(rd.randint(7, 9)) + str(rd.randint(7, 9)) + str(rd.randint(7, 9))
    else :
        fill_ = '#'+str(rd.randint(0,3))+str(rd.randint(0,3))+str(rd.randint(0,3))
    book_name = book_list[rd.randint(0, len(book_list)-1)]
    book_idx = 0
    blank = 0
    for item in book_name :
        book_idx += 1
        # if book_idx > 6 :
        #     break
        if book_side_points[1][1]+init_y+blank+font_size >= init_p2[0] :
            break
        draw.text((book_side_points[1][0] + init_x, book_side_points[1][1] + init_y+blank), str(item), font=font, fill=fill_)
        if item == ' ' :
            blank += int(font_size*0.5)
        else :
            blank += int(font_size)

    # draw.text((book_side_points[1][0] + init_x, book_side_points[1][1] + init_y+font_size+5), '나', font=font, fill=fill_)
    # draw.text((book_side_points[1][0] + init_x, book_side_points[1][1] + init_y+2*font_size+5), '의', font=font, fill=fill_)
    # # draw.text((book_side_points[1][0] + init_x, book_side_points[1][1] + init_y), '다', font=font, fill=fill_)
    # # draw.text((book_side_points[1][0] + init_x, book_side_points[1][1] + init_y), '라', font=font, fill=fill_)

    label_rs = np.array(label_pil)

    # 책 테두리  (x, y) (x, y) book_points[1][0] = x
    book_b = mk_rd_100()
    book_g = mk_rd_100()
    book_r = mk_rd_100()

    side_points[0], side_points[2] = side_points[2], side_points[0]
    book_side_points[0], book_side_points[2] = book_side_points[2], book_side_points[0]

    cv2.line(label_rs, (book_side_points[1][0], book_side_points[1][1]), (book_side_points[2][0], book_side_points[1][1]), (book_b,book_g,book_r), 2)
    cv2.line(label_rs, (book_side_points[1][0], book_side_points[1][1]), (book_side_points[1][0], book_side_points[3][1]), (book_b,book_g,book_r), 2)
    cv2.line(label_rs, (book_side_points[1][0], book_side_points[3][1]), (book_side_points[2][0], book_side_points[3][1]), (book_b,book_g,book_r), 2)
    cv2.line(label_rs, (book_side_points[2][0], book_side_points[1][1]), (book_side_points[2][0], book_side_points[3][1]), (book_b,book_g,book_r), 2)

    label_rs[init_p2[0]+0:init_p2[0]+resize_y, init_p2[1]+0:init_p2[1]+resize_x, :] = label_size # bookcover attach

    if draw_circle :
        cv2.circle(label_rs, (center_point[0], center_point[1]), 5, (0, 0, 255), -1)
        cv2.circle(label_rs, (side_points[0][0], side_points[0][1]), 5, (255, 0, 0), -1)
        cv2.circle(label_rs, (side_points[1][0], side_points[1][1]), 5, (0, 255, 0), -1)
        cv2.circle(label_rs, (side_points[2][0], side_points[2][1]), 5, (0, 0, 255), -1)
        cv2.circle(label_rs, (side_points[3][0], side_points[3][1]), 5, (255, 255, 255), -1)
    if show :
        cv2.imshow('size', label_rs)
    if apply :
        label = label_rs
    if draw_book :
        # cv2.circle(label_rs, (center_point[0], center_point[1]), 5, (0, 0, 255), -1)
        cv2.circle(label_rs, (book_side_points[0][0], book_side_points[0][1]), 5, (255, 0, 0), -1)
        cv2.circle(label_rs, (book_side_points[1][0], book_side_points[1][1]), 5, (0, 255, 0), -1)
        cv2.circle(label_rs, (book_side_points[2][0], book_side_points[2][1]), 5, (0, 0, 255), -1)
        cv2.circle(label_rs, (book_side_points[3][0], book_side_points[3][1]), 5, (255, 255, 255), -1)

def change_blur(show=False, apply=False, dice=0):
    if dice > 3 :  # 30%
        return None
    else :
        apply = True

    global aug_cnt
    global label
    global background
    global side_points
    global center_point
    aug_cnt += 1  # aug_cnt + 1

    label_blur = label.copy()
    if dice == 0 :
        kernel_s = 5
    else :
        kernel_s = 7

    gauss = cv2.GaussianBlur(label_blur, (kernel_s, kernel_s), 0)

    if show :
        cv2.imshow('blur', gauss)
    if apply :
        label = label_blur

# def change_noise(apply=False, std=10):
#     global aug_cnt
#     global label
#     global background
#     global side_points
#     global center_point
#     aug_cnt += 1  # aug_cnt + 1
#
#     label_noise = label.copy()
#
#     height, width, _ = label_noise.shape
#
#     for i in range(height):
#         for a in range(width):
#             make_noise = np.random.normal() # 랜덤함수를 이용하여 노이즈 적용 0~1
#             # percent_ = 0 if rd.randint(0, 100)<97 else 1
#             percent_ = 1
#             if percent_ :
#                 label_noise[i][a][0] = std * make_noise + label_noise[i][a][0]
#                 label_noise[i][a][1] = std * make_noise + label_noise[i][a][1]
#                 label_noise[i][a][2] = std * make_noise + label_noise[i][a][2]
#
#     if apply :
#         label = label_noise

def change_bright(show=False, apply=True) :   # 1-1. brightness noise
    global aug_cnt
    global label
    aug_cnt += 1 # aug_cnt + 1

    label_bright = label.copy()
    limit, always_apply, p = 0.2, False, 1
    transform = A.Compose([A.RandomBrightnessContrast(limit, always_apply, p)], p=1)
    label_bright = transform(image=label_bright)['image']
    if show :
        cv2.imshow('bright'+str(aug_cnt), label_bright)
    if apply :
        label = label_bright


def change_rot(draw_circle=False, show=False, apply=True, draw_book=False) :  # rotation
    global center_point
    global side_points
    global label
    global aug_cnt
    global book_side_points
    aug_cnt += 1

    label_rot = label.copy()

    rotate_angle = rd.randint(1, 90) * (1 - 2*rd.randint(0,1))   # -90~90
    M = cv2.getRotationMatrix2D((center_point[0], center_point[1]), rotate_angle, 1)
    label_rot = cv2.warpAffine(label_rot, M, (640, 640))

    original_angle_0 = -math.degrees(math.atan2(side_points[0][1]-center_point[1], side_points[0][0]-center_point[0]))
    original_angle_1 = -math.degrees(math.atan2(side_points[1][1]-center_point[1], side_points[1][0]-center_point[0]))
    original_angle_2 = -math.degrees(math.atan2(side_points[2][1]-center_point[1], side_points[2][0]-center_point[0]))
    original_angle_3 = -math.degrees(math.atan2(side_points[3][1]-center_point[1], side_points[3][0]-center_point[0]))

    book_org_ang_0 = -math.degrees(
        math.atan2(book_side_points[0][1]-center_point[1], book_side_points[0][0]-center_point[0]))
    book_org_ang_1 = -math.degrees(
        math.atan2(book_side_points[1][1] - center_point[1], book_side_points[1][0] - center_point[0]))
    book_org_ang_2 = -math.degrees(
        math.atan2(book_side_points[2][1] - center_point[1], book_side_points[2][0] - center_point[0]))
    book_org_ang_3 = -math.degrees(
        math.atan2(book_side_points[3][1] - center_point[1], book_side_points[3][0] - center_point[0]))
    # cv2.line(label_rot, (book_side_points[0][0],book_side_points[0][1]), center_point, (255,0,0), 5)

    # print(book_org_ang_0, book_org_ang_1, book_org_ang_2, book_org_ang_3)

    rotate_r_book_upper = math.sqrt((book_side_points[0][0]-center_point[0])**2 + (book_side_points[0][1]-center_point[1])**2)
    rotate_r_book_lower = math.sqrt((book_side_points[2][0]-center_point[0])**2 + (book_side_points[2][1]-center_point[1])**2)

    book_side_0_0 = center_point[0]+int(rotate_r_book_upper*math.cos(math.radians(rotate_angle+book_org_ang_0)))
    book_side_0_1 = center_point[1]-int(rotate_r_book_upper*math.sin(math.radians(rotate_angle+book_org_ang_0)))
    book_side_1_0 = center_point[0] + int(rotate_r_book_upper * math.cos(math.radians(rotate_angle + book_org_ang_1)))
    book_side_1_1 = center_point[1] - int(rotate_r_book_upper * math.sin(math.radians(rotate_angle + book_org_ang_1)))
    book_side_2_0 = center_point[0] + int(rotate_r_book_lower * math.cos(math.radians(rotate_angle + book_org_ang_2)))
    book_side_2_1 = center_point[1] - int(rotate_r_book_lower * math.sin(math.radians(rotate_angle + book_org_ang_2)))
    book_side_3_0 = center_point[0] + int(rotate_r_book_lower * math.cos(math.radians(rotate_angle + book_org_ang_3)))
    book_side_3_1 = center_point[1] - int(rotate_r_book_lower * math.sin(math.radians(rotate_angle + book_org_ang_3)))

    if book_side_0_0 < 0:
        book_side_0_1 = int(-(book_side_0_1 - book_side_3_1) * book_side_3_0 / (book_side_0_0 - book_side_3_0)  + book_side_3_1)
        book_side_0_0 = 0
    if book_side_1_0 < 0:
        book_side_1_1 = int(-(book_side_1_1 - book_side_2_1) * book_side_2_0 / (book_side_1_0 - book_side_2_0) + book_side_2_1)
        book_side_1_0 = 0
    if book_side_0_0 > 640 :
        book_side_0_1 = int((book_side_0_1 - book_side_3_1) * (640-book_side_3_0) / (book_side_0_0 - book_side_3_0) + book_side_3_1)
        book_side_0_0 = 640
    if book_side_1_0 > 640:
        book_side_1_1 = int( -(book_side_2_1 - book_side_1_1) * (640-book_side_2_0) / (book_side_1_0 - book_side_2_0) + book_side_2_1)
        book_side_1_0 = 640

    book_side_points = [[book_side_0_0, book_side_0_1],
                        [book_side_1_0, book_side_1_1],
                        [book_side_2_0, book_side_2_1],
                        [book_side_3_0, book_side_3_1]]


    # print(book_side_points)  [x, y]
    if draw_book:
        cv2.circle(label_rot, (book_side_points[0][0], book_side_points[0][1]), 5, (255, 0, 0), -1)
        cv2.circle(label_rot, (book_side_points[1][0], book_side_points[1][1]), 5, (0, 255, 0), -1)
        cv2.circle(label_rot, (book_side_points[2][0], book_side_points[2][1]), 5, (0, 0, 255), -1)
        cv2.circle(label_rot, (book_side_points[3][0], book_side_points[3][1]), 5, (255, 255, 255), -1)


    rotate_r = math.sqrt((side_points[0][0]-center_point[0])**2 + (side_points[0][1]-center_point[1])**2)

    rotate_side_left_lower = [center_point[0]+int(rotate_r)*math.cos(math.radians(rotate_angle+original_angle_0)),
                               center_point[1]-int(rotate_r)*math.sin(math.radians(rotate_angle+original_angle_0))]
    #
    rotate_side_left_upper = [center_point[0]+int(rotate_r)*math.cos(math.radians(rotate_angle+original_angle_1)),
                               center_point[1]-int(rotate_r)*math.sin(math.radians(rotate_angle+original_angle_1))]

    rotate_side_right_upper = [center_point[0]+int(rotate_r)*math.cos(math.radians(rotate_angle+original_angle_2)),
                               center_point[1]-int(rotate_r)*math.sin(math.radians(rotate_angle+original_angle_2))]
    # right_upper
    rotate_side_right_lower = [center_point[0]+int(rotate_r)*math.cos(math.radians(rotate_angle+original_angle_3)),
                               center_point[1]-int(rotate_r)*math.sin(math.radians(rotate_angle+original_angle_3))]
    # side_points [x, y]
    side_points = [[int(rotate_side_right_upper[0]), int(rotate_side_right_upper[1])],
                   [int(rotate_side_left_upper[0]), int(rotate_side_left_upper[1])],
                   [int(rotate_side_left_lower[0]), int(rotate_side_left_lower[1])],
                   [int(rotate_side_right_lower[0]), int(rotate_side_right_lower[1])]]

    side_points[0], side_points[2] = side_points[2], side_points[0]

    if draw_circle :
        cv2.circle(label_rot, (side_points[0][0], side_points[0][1]), 5, (255, 0, 0), -1)
        cv2.circle(label_rot, (side_points[1][0], side_points[1][1]), 5, (0, 255, 0), -1)
        cv2.circle(label_rot, (side_points[2][0], side_points[2][1]), 5, (0, 0, 255), -1)
        cv2.circle(label_rot, (side_points[3][0], side_points[3][1]), 5, (255, 255, 255), -1)
    if show :
        cv2.imshow('rot', label_rot)
    if apply :
        label = label_rot

def change_pers(show=False, apply=True) : # Perspective
    global label
    global aug_cnt
    aug_cnt += 1

    label_pers = label.copy()

    pts1 = np.float32(side_points)
    pts2 = np.float32([[400, 100], [200, 100], [100, 500], [500, 500]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    label_pers = cv2.warpPerspective(label_pers, M, (640, 640))
    if show :
        cv2.imshow('pers', label_pers)
    if apply :
        label = label_pers

# def make_noise(std, image_):
#     height, width, _ = image_.shape
#     for i in range(int(height/5)):
#         for a in range(int(width/5)):
#             make_noise = np.random.normal() # 랜덤함수를 이용하여 노이즈 적용 0~1
#             percent_ = 0 if rd.randint(0, 100)<97 else 1
#             if percent_ :
#                 if 5*i+4 <= height and 5*a+4 <= width:
#                     image_[5*i:5*i+4][5*a:5*a+4][0] = std * make_noise + 100
#                     image_[5*i:5*i+4][5*a:5*a+4][1] = std * make_noise + 100
#                     image_[5*i:5*i+4][5*a:5*a+4][2] = std * make_noise + 100
#                 else :
#                     pass
#     return image_

def show_points():
    global label
    cv2.circle(label, (book_side_points[0][0], book_side_points[0][1]), 5, (255, 0, 0), -1)
    cv2.circle(label, (book_side_points[1][0], book_side_points[1][1]), 5, (255, 0, 0), -1)
    cv2.circle(label, (book_side_points[2][0], book_side_points[2][1]), 5, (255, 0, 0), -1)
    cv2.circle(label, (book_side_points[3][0], book_side_points[3][1]), 5, (255, 0, 0), -1)

    cv2.circle(label, (side_points[0][0], side_points[0][1]), 5, (0, 0, 255), -1)
    cv2.circle(label, (side_points[1][0], side_points[1][1]), 5, (0, 0, 255), -1)
    cv2.circle(label, (side_points[2][0], side_points[2][1]), 5, (0, 0, 255), -1)
    cv2.circle(label, (side_points[3][0], side_points[3][1]), 5, (0, 0, 255), -1)

    cv2.imshow('pts', label)


def save_result(filename, train=False, valid=False, test=False) :
    global bground

    if train :
        save_path_images = '/home/kdh/Desktop/new_dataset/train/images/'
        save_path_labels = '/home/kdh/Desktop/new_dataset/train/labels/'
    elif valid :
        save_path_images = '/home/kdh/Desktop/new_dataset/valid/images/'
        save_path_labels = '/home/kdh/Desktop/new_dataset/valid/labels/'
    elif test :
        save_path_images = '/home/kdh/Desktop/new_dataset/test/images/'
        save_path_labels = '/home/kdh/Desktop/new_dataset/test/labels/'
    else :
        return None

    background_img = False
    if train :   # random background
        random_back = rd.randint(0,9)
        if random_back == 7 :
            background_img = True
    if background_img :
        random_background = cv2.imread(bground[rd.randint(0, len(bground) - 1)])
        random_background = cv2.resize(random_background, (640, 640))
        cv2.imwrite(save_path_images + str(filename) + '.jpg', random_background)
        f = open(save_path_labels + str(filename) + '.txt', 'w')
        f.write(' ')
        f.close()
        return None

    cv2.imwrite(save_path_images + str(filename) + '.jpg', label)
    f = open(save_path_labels + str(filename) + '.txt', 'w')
    f.write('0 ' + str(book_side_points[0][0] / 640) + ' ' + str(book_side_points[0][1] / 640) + ' '
            + str(book_side_points[1][0] / 640) + ' ' + str(book_side_points[1][1] / 640) + ' '
            + str(book_side_points[2][0] / 640) + ' ' + str(book_side_points[2][1] / 640) + ' '
            + str(book_side_points[3][0] / 640) + ' ' + str(book_side_points[3][1] / 640) + ' '
            + str(book_side_points[0][0] / 640) + ' ' + str(book_side_points[0][1] / 640))
    f.write('\n')
    f.write('1 ' + str(side_points[0][0] / 640) + ' ' + str(side_points[0][1] / 640) + ' '
            + str(side_points[1][0] / 640) + ' ' + str(side_points[1][1] / 640) + ' '
            + str(side_points[2][0] / 640) + ' ' + str(side_points[2][1] / 640) + ' '
            + str(side_points[3][0] / 640) + ' ' + str(side_points[3][1] / 640) + ' '
            + str(side_points[0][0] / 640) + ' ' + str(side_points[0][1] / 640))
    f.close()

if __name__ == '__main__' :
    for i in range(50):
        make_background()
        make_label(show=False, draw_center=False)
        change_size(show=False, draw_circle=False, resize_x=rd.randint(3,7)*10, draw_book=False)
        change_blur(dice=rd.randint(0,9))  # 30%
        change_bright(show=False)  # brightness change
        change_rot(show=True, draw_circle=False, draw_book=False)
        # change_pers(show=False)
        save_result(i, test=True)
        # show_points()  # show points
        # cv2.waitKey(0) # show picture
