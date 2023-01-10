import os
import numpy as np
from PIL import Image
import math


image_path = "C:/Users/ehy84/Desktop/YUJIN/2021-2/CV/CV_Final/images/" # 원본 이미지 디렉토리
opencv_canny_path = "C:/Users/ehy84/Desktop/YUJIN/2021-2/CV/CV_Final/images/cv2_test/" # cv2로 생성한 이미지 디렉토리
Thresholding_path = "C:/Users/ehy84/Desktop/YUJIN/2021-2/CV/CV_Final/images/Thresholding/" # Thresholding시 픽셀 값 설정 방법 다르게 생성한 이미지들 디렉토리
filter_path = "C:/Users/ehy84/Desktop/YUJIN/2021-2/CV/CV_Final/images/filter/" # filter 다르게 생성한 이미지들 디렉토리
image_name = ["ho1.jpg", "ho2.jpg", "ho3.jpg", "im2.png", "im6.png"] # 원본 이미지 이름 
edge_name = ["15_5_0.jpg", "15_5_1.jpg", "15_5_2.jpg", "15_5_3.jpg", "15_5_4.jpg", "100_10_0.jpg", "100_10_1.jpg", "100_10_2.jpg", "100_10_3.jpg" ,"100_10_4.jpg"] # 생성한 이미지 이름 

CHANNEL = 1

# 이미지 회색조 변환
def gray_scale(Image): 
    len_row = len(Image)
    len_col = len(Image[0])
    len_channel = 3 

    # 결과값으로 반환될 리스트
    result = [[[0 for c in range(CHANNEL)] for i in range(len_col)] for j in range(len_row)]

    for i in range(len_row):
        for j in range(len_col):
            value = 0
            #NTSC RGB to Grayscale 변환 공식 이용(png), PIL은 채널이 RGB 순서로 되어있기 때문에 아래와 같은 조건문 사용
            for c in range(len_channel):
                if c == 0:
                    value = value + 0.299 * Image[i][j][c] 
                elif c == 1:
                    value = value + 0.587 * Image[i][j][c]
                elif c == 2:
                    value = value + 0.114 * Image[i][j][c]
            result[i][j][0] = value # 결과 리스트에 저장 
    
    return result

# 차원 줄이기(grayscale 이미지의 shape은 (H, W)이고 아래 함수들에서 동작하는 이미지의 shape은 (H, W, C) 이다.)
def squeeze(Image):
    len_row = len(Image)
    len_col = len(Image[0])

    result = [[0 for i in range(len_col)] for j in range(len_row)] # 결과값 

    for i in range(len_row):
        for j in range(len_col):
            for c in range(CHANNEL):
                value = Image[i][j][c] # Channel 차원의 value 받아오기 
            result[i][j] = value # 2차원 이미지에 저장 
    return result

# zero padding function
def zero_padding(Image):
    len_row = len(Image) # 행의 길이
    len_col = len(Image[0]) # 열의 길이

    zero_pad = [[0 for k in range(CHANNEL)] for i in range(len_col + 2)] # 이미지의 top과 bottom에 추가할 zero padding

    result_img = [[[0 for k in range(CHANNEL)] for i in range(0, len_col)] for j in range(len_row)] # 결과값으로 나갈 이미지
    
    # 결과값으로 나갈 이미지에 원래 이미지를 복사
    for i in range(len_row):
        for j in range(len_col):
            for c in range(CHANNEL):
                result_img[i][j][c] = Image[i][j][c]

    for i in range(0, len_row):
        result_img[i].insert(0, [0,0,0]) # 각 행의 맨 왼쪽에 zero padding
        result_img[i].append([0,0,0]) # 각 행의 맨 왼쪽에 zero padding

    result_img.insert(0, zero_pad) # 이미지의 top에 zero padding 추가
    result_img.append(zero_pad) # 이미지의 bottom에 zero padding 추가 

    return result_img

# 2d convolution function 
def conv2d(Image, Filter):

    len_row_img = len(Image) # 이미지 행의 길이
    len_col_img = len(Image[0]) # 이미지 열의 길이

    len_row_fil = len(Filter) # 필터 이미지 행의 길이
    len_col_fil = len(Filter[0]) # 필터 이미지 열의 길이

    row_out = len_row_img - len_row_fil + 1 # convolution 연산 결과로 나올 이미지 리스트의 행 길이
    col_out = len_col_img - len_col_fil + 1 # convolution 연산 결과로 나올 이미지 리스트의 열 길이

    out = [[[0 for k in range(CHANNEL)] for i in range(col_out)] for j in range(row_out)] # convolution 연산 결과로 나올 이미지 리스트

    # convolution 연산 반복문
    for i in range(row_out):
        for j in range(col_out):
            for c in range(CHANNEL): # RGB 채널마다 계산 
                channel_val = 0 # 결과 이미지에 들어갈 값 0으로 초기화
                for k in range(len_row_fil):
                    for z in range(len_col_fil):
                        channel_val = channel_val + Image[i + k][j + z][c] * Filter[k][z][c] # convolution 연산
                out[i][j][c] = channel_val # 연산 수행 결과를 결과 이미지에 대입
    return out

def gaussian_smoothing(Image):
    Image = zero_padding(Image) # 가장자리 처리, 이미지 사이즈 변화를 막기 위한 zero padding 
    Gaussain_Filter = [[[1/16 , 1/16, 1/16], [1/8 , 1/8, 1/8], [1/16 , 1/16, 1/16]],
                       [[1/8 , 1/8, 1/8], [1/4 , 1/4, 1/4], [1/8 , 1/8, 1/8]],
                       [[1/16 , 1/16, 1/16], [1/8 , 1/8, 1/8], [1/16 , 1/16, 1/16]]]
    result = conv2d(Image, Gaussain_Filter)
    return result
 
# partial derivative wrt x                        
def derivation_x(Image):
    Filtered = gaussian_smoothing(Image)
    Filtered = zero_padding(Filtered) # x축 방향 기울기를 구할 이미지에 zero padding 추가

    basic_gradient_Filter = [[[0 for k in range(CHANNEL)], [0 for k in range(CHANNEL)], [0 for k in range(CHANNEL)]],
                            [[-1 for k in range(CHANNEL)], [0 for k in range(CHANNEL)], [1 for k in range(CHANNEL)]],
                            [[0 for k in range(CHANNEL)], [0 for k in range(CHANNEL)], [0 for k in range(CHANNEL)]]] # x축 방향 basic_gradient_Filter

    sobel_Filter = [[[-1 for k in range(CHANNEL)], [0 for k in range(CHANNEL)], [1 for k in range(CHANNEL)]],
                    [[-2 for k in range(CHANNEL)], [0 for k in range(CHANNEL)], [2 for k in range(CHANNEL)]],
                    [[-1 for k in range(CHANNEL)], [0 for k in range(CHANNEL)], [1 for k in range(CHANNEL)]]] # x축 방향 sobel_Filter

    scharr_Filter = [[[-3 for k in range(CHANNEL)], [0 for k in range(CHANNEL)], [3 for k in range(CHANNEL)]],
                    [[-10 for k in range(CHANNEL)], [0 for k in range(CHANNEL)], [10 for k in range(CHANNEL)]],
                    [[-3 for k in range(CHANNEL)], [0 for k in range(CHANNEL)], [3 for k in range(CHANNEL)]]] # x축 방향 scharr_Filter

    return conv2d(Filtered, sobel_Filter) # Image와 filter convolution 연산 결과 반환

# partial derivative wrt y
def derivation_y(Image):
    Filtered = gaussian_smoothing(Image)
    Filtered = zero_padding(Filtered) # y축 방향 기울기를 구할 이미지에 zero padding 추가 

    basic_gradient_Filter = [[[0 for k in range(CHANNEL)], [-1 for k in range(CHANNEL)], [0 for k in range(CHANNEL)]],
                            [[0 for k in range(CHANNEL)], [0 for k in range(CHANNEL)], [0 for k in range(CHANNEL)]],
                            [[0 for k in range(CHANNEL)], [1 for k in range(CHANNEL)], [0 for k in range(CHANNEL)]]] # y축 방향 basic_gradient_Filter

    sobel_Filter =[[[-1 for k in range(CHANNEL)], [-2 for k in range(CHANNEL)], [-1 for k in range(CHANNEL)]],
                    [[0 for k in range(CHANNEL)], [0 for k in range(CHANNEL)], [0 for k in range(CHANNEL)]],
                    [[1 for k in range(CHANNEL)], [2 for k in range(CHANNEL)], [1 for k in range(CHANNEL)]]] # y축 방향 sobel_Filter

    scharr_Filter =[[[-3 for k in range(CHANNEL)], [-10 for k in range(CHANNEL)], [-3 for k in range(CHANNEL)]],
                    [[0 for k in range(CHANNEL)], [0 for k in range(CHANNEL)], [0 for k in range(CHANNEL)]],
                     [[3 for k in range(CHANNEL)], [10 for k in range(CHANNEL)], [3 for k in range(CHANNEL)]]] # y축 방향 scharr_Filter

    return conv2d(Filtered, sobel_Filter) # Image와 filter convolution 연산 결과 반환

# magnitude, orientation function
def mag_or(Image):
    der_v_x = derivation_x(Image) # x축 방향으로 구한 기울기값
    der_v_y = derivation_y(Image) # y축 방향으로 구한 기울기값

    len_row = len(der_v_x) # 행의 길이
    len_col = len(der_v_x[0]) # 열의 길이

    out_mag = [[[0 for k in range(CHANNEL)] for i in range(len_col)] for j in range(len_row)] # magnitude 저장할 결과 값
    out_ori = [[[0 for k in range(CHANNEL)] for i in range(len_col)] for j in range(len_row)] # suppression시 orientation을 이용해 어느 방향 픽셀들을 비교할 것인지를 저장할 결과 값 

    for i in range(0, len_row):
        for j in range(0, len_col):
            for c in range(CHANNEL):
                out_mag[i][j][c] = ((der_v_x[i][j][c]) ** 2 + (der_v_y[i][j][c]) ** 2) ** 0.5 # 각 픽셀에 대해 magnitude 값을 구함
    
    for i in range(0, len_row):
        for j in range(0, len_col):
            for c in range(CHANNEL):
                theta = np.arctan2(der_v_x[i][j][c], der_v_y[i][j][c]) 
                theta = theta * 180 / np.pi # radian --> degree 변환    

                if (theta > -22.5 and theta <= 22.5) or (theta <= -157.5 and theta > 157.5):
                    out_ori[i][j][c] = 0 # non-max suppression 시 수직 방향 비교
                
                elif (theta > 22.5 and theta <= 67.5) or (theta <= -112.5 and theta > -157.5):
                    out_ori[i][j][c] = 45 # non-max suppression 시 북동, 남서 대각선 방향 비교
                
                elif (theta > 67.5 and theta <= 112.5) or (theta <= -67.5 and theta > -112.5) : 
                    out_ori[i][j][c] = 90 # non-max suppression 시 수평 방향 비교
                
                else:
                    out_ori[i][j][c] = 135 # non-max suppression 시 북서, 남동 대각선 방향 비교
    
    return out_mag, out_ori 

# Nonmax suppression
def NMS(Image):
    mag, ori = mag_or(Image) # Magnitude Image 초기화

    len_row = len(mag)
    len_col = len(mag[0])
   
    mag_pad = zero_padding(mag) # 가장 자리 값을 처리하기 위한 zero padding된 이미지 생성

    # nonmax suppression  
    for i in range(1, len(mag_pad) -1): # zero padding이 되어있으므로 range 별도 설정
        for j in range(1, len(mag_pad[0]) - 1): # zero padding이 되어있으므로 range 별도 설정
            for c in range(CHANNEL):
                ori_val = ori[i-1][j-1][c] # 픽셀이 어느 방향 픽셀들을 비교해야 할지 기록해놓은 값
                if ori_val == 0: # 수직 3픽셀 비교
                    comp_val_1, comp_val_2 = mag_pad[i][j + 1][c], mag_pad[i][j - 1][c] # 비교할 위 아래 값 저장
                    
                elif ori_val == 45: # 우상향 대각선 비교 
                     comp_val_1, comp_val_2 = mag_pad[i - 1][j + 1][c], mag_pad[i + 1][j - 1][c] # 비교할 값 저장

                elif ori_val == 90: # 수평 비교 
                     comp_val_1, comp_val_2 = mag_pad[i - 1][j][c], mag_pad[i + 1][j][c] # 비교할 값 저장

                else: # 좌상향 대각선 비교 
                     comp_val_1, comp_val_2 = mag_pad[i - 1][j - 1][c], mag_pad[i + 1][j + 1][c] # 비교할 값 저장

                if mag_pad[i][j][c] > comp_val_1 and mag_pad[i][j][c] > comp_val_2: # 비교 결과 해당 방향 최대값인 경우
                    pass # 픽셀 값을 살린다.
                else: # 그 이외의 경우
                    mag_pad[i][j][c] = 0 # 0으로 만듬


    # Nonmax suppression 처리가 된 이미지를 원래 크기의 이미지로 복사시킴.                   
    for i in range(0, len_row):
        for j in range(0, len_col):
            for c in range(0, CHANNEL):
                mag[i][j][c] = mag_pad[i+1][j+1][c]
    
    return mag

def Canny(Image, high, low):
    result_img = NMS(Image) # Nonmax suppression 처리가 된 이미지 생성
    Image_pad = zero_padding(result_img) # 가장자리 값들을 처리하기 위해 zero padding 된 이미지 생성

    len_row = len(result_img)
    len_col = len(result_img[0])

    len_row_pad = len(Image_pad)
    len_col_pad = len(Image_pad[0])

    Image_bool = [[[0 for k in range(CHANNEL)] for i in range(len_col_pad)] for j in range(len_row_pad)] # 값이 threshold high, low 사이에 위치해있을시 살릴지 말지 이어진 픽셀을 보고 결정하기 위해 생존 여부를 기록해 놓는 리스트

    for i in range(1, len_row_pad -1): # zero padding이 되어있으므로 range 별도 설정
        for j in range(1, len_col_pad - 1): # zero padding이 되어있으므로 range 별도 설정
            for c in range(CHANNEL):
                if Image_pad[i][j][c] < low: # threshold low 보다 낮을 경우 
                    Image_pad[i][j][c] = 0 # 0으로 처리한다.
                    Image_bool[i][j][c] = 0 # 생존 여부 기록 리스트에 0(죽음) 기록

                elif Image_pad[i][j][c] >= low and Image_pad[i][j][c] <= high: # 값이 high와 low 사이에 있을 경우
                    zero_bool = True # 생존 여부 판단을 위한 boolean 변수, True : 죽음, False : 생존
                    for m in range(-1, 2, 2): # 자신 기준 8 방향 픽셀의 생존 여부 기록 리스트를 확인한다.
                        if Image_bool[i][j+m][c] == 1: # 자신 기준 위 아래 픽셀의 생존 여부 판단
                            zero_bool = False
                        
                        elif Image_bool[i+m][j][c] == 1:  # 자신 기준 양 옆 픽셀의 생존 여부 판단
                            zero_bool = False
                        
                        elif Image_bool[i+m][j-1][c] == 1: # 자신 기준 왼쪽 열 위 아래 픽셀의 생존 여부 판단
                            zero_bool = False
                        
                        elif Image_bool[i+m][j+1][c] == 1:  # 자신 기준 오른쪽 열 위 아래 픽셀의 생존 여부 판단
                            zero_bool -False

                    if zero_bool == True: # 이어지지 않았다고 판단됐을 경우
                        Image_pad[i][j][c] = 0 # 0으로 처리한다.
                        Image_bool[i][j][c] = 0 # 생존 여부 기록 리스트에 0(죽음) 기록

                    else: # 이어져 있다고 판단됐을 경우(약한 엣지)
                        # 픽셀 값 그대로 사용할 시 Image_pad[i][j][c] 에 대해 안 건드리면 됨.
                        #Image_pad[i][j][c] = low 
                        Image_pad[i][j][c] = 122.5 # 약한 엣지의 경우 122.5 부여
                        Image_bool[i][j][c] = 1 # 생존 여부 기록 리스트에 1(생존) 기록  

                else: # threshold high 보다 높은 경우(강한 엣지)
                    # 픽셀 값 그대로 사용할 시 Image_pad[i][j][c] 에 대해 안 건드리면 됨.
                    #Image_pad[i][j][c] = high 
                    Image_pad[i][j][c] = 255 # 강한 엣지의 경우 255 부여
                    Image_bool[i][j][c] = 1 # 생존 여부 기록 리스트에 1(생존) 기록
    
    # Thresholding 처리된 이미지(가장자리 값 처리위해 zero padding 되어있던 상태)를 원 크기 이미지에 복사시킨다.
    for i in range(0, len_row):
        for j in range(0, len_col):
            for c in range(0, CHANNEL):
                result_img[i][j][c] = Image_pad[i+1][j+1][c]
        
    return result_img

# 성능 평가를 위한 PSNR 
def PSNR(Image1, Image2):
    len_row = len(Image1)
    len_col = len(Image1[0])
    mse = 0 # MSE를 저장할 value
    for i in range(len_row):
        for j in range(len_col):
            mse = mse + (Image1[i][j] - Image2[i][j]) ** 2
    mse = mse / (len_row * len_col)
    max_val = 255.0 # 픽셀이 가질 수 있는 최대값 - 최소값(255 - 0)
    PSNR = 20 * math.log10(max_val / (mse ** 0.5)) # PSNR 수식에 값 대입 
    return PSNR

#LoG
def LoG(Image):
    Image = gaussian_smoothing(Image) # gaussian smoothing 
    Image = zero_padding(Image) # 가장자리 처리, 이미지 사이즈 변화를 막기 위한 zero padding 
    Laplacian_filter = [[[0 for i in range(CHANNEL)], [1 for i in range(CHANNEL)], [0 for i in range(CHANNEL)]],
                       [[1 for i in range(CHANNEL)], [-4 for i in range(CHANNEL)], [1 for i in range(CHANNEL)]],
                       [[0 for i in range(CHANNEL)], [1 for i in range(CHANNEL)], [0 for i in range(CHANNEL)]]] # Laplacian filter
    result = conv2d(Image, Laplacian_filter)
    return result

def main():
    #edge image 생성 
    for i in range(5): # 원본 이미지 5장 모두에 대해서
        img1 = Image.open(image_path + image_name[i]) # 이미지 로드
        img1 = np.array(img1) # 이미지 numpy array 변환
        img1 = img1.tolist() # numpy array를 list로 변환
        img1 = gray_scale(img1) # gray scaling 

        x_der = LoG(img1) # Canny edge detector 실행
        x_der = squeeze(x_der) # 이미지 저장을 위한 차원 축소
        x_der = np.array(x_der) # 리스트를 numpy array로 변환 
        x_der = Image.fromarray((x_der * 1).astype(np.float32)).convert('L') # numpy array를 PIL.Image화 
        x_der.save(image_path + "LoG_" + str(i) + ".JPG") # 이미지 저장
    
    # PSNR 계산 
    '''
    psnr = 0 # psnr 평균을 내기 위해 만든 변수

    for i in range(10):
        img1 = Image.open(opencv_canny_path + edge_name[i]) # 기준이 되는 cv2로 생성한 이미지 로드
        img1 = np.array(img1) 
        img1 = img1.tolist() 
        
        img2 = Image.open(filter_path + "sobel/" + edge_name[i]) # 비교할 이미지의 세부 디렉토리 입력 후 이미지 로드
        img2 = np.array(img2)
        img2 = img2.tolist()

        psnr = psnr + PSNR(img1, img2) # psnr 계산 후 저장
    
    print("PSNR of sobel : ", psnr / 10) # 평균 값 print
    '''

if __name__ == '__main__':
    main()
    


