import numpy as np
import cv2
import os
import glob

'''
2019104035 장서연
2022 - 1 로봇센서데이터처리 중간고사 대체과제
'''
class MyCalib:
    def __init__(self):
        # 체커보드로, 카메라 캘리브레이션을 수행하는 클래스 입니다.
        self.CHECKERBOARD = (6,8)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.calib_images = glob.glob('../calib/*.jpg')
        self.objpoints = []
        self.imgpoints = []

        self.objp = np.zeros((1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)
        self.objp[0,:,:2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)

        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None

    def run_calibration(self, mode):
        '''
        휴대전화 카메라로 찍은 해상도 높은 이미지를 사용하고 있기 때문에, 그냥 캘리브레이션 함수를 적용해주게 되면 시간이 오래 걸립니다.
        그래서 mode 변수를 추가하여 calibration 옵션을 넣었습니다.

        1. mode == 1로 설정시 제가 찾은 pre-defined calib coefficient matrix를 불러옵니다.
        2. mode == 2 또는 다른 숫자로 설정시 run_calibration() function을 수행하여 이미지로 부터 calibration 매트릭스를 구합니다.
        '''

        if (mode == 1):
            self.mtx = np.array([[2.66888569e+03, 0.00000000e+00, 1.99903739e+03],[0.00000000e+00, 2.67671159e+03, 1.44211074e+03], [0.00000000e+00 ,0.00000000e+00, 1.00000000e+00]])
            self.dist = np.array([[ 1.75402080e-01, -7.48159879e-01 , 2.86709827e-04 ,-1.64777692e-03, 1.26106460e+00]])
            self.rvecs = (np.array([[-0.36972014], [-0.60928317],[ 1.2972393 ]]), 
                                    np.array([[-0.45617329],[-0.59331868],[ 1.2566169 ]]), 
                                    np.array([[ 0.1735061 ],[ 0.27998858],[-1.54732321]]),
                                    np.array([[-0.54084815],[ 0.57465703],[-1.19234384]]), 
                                    np.array([[-0.26475728],[ 0.30446651],[ 1.54904813]]))
            
            self.tvecs = (np.array([[ 0.77999127],[-2.92899164],[17.51990434]]), 
                                    np.array([[ 3.20470194],[-3.59188578],[11.70767985]]), 
                                    np.array([[-2.59491562],[ 2.38296388],[16.67693034]]), 
                                    np.array([[-3.49741424],[ 2.29262768],[22.62296227]]), 
                                    np.array([[ 3.75438944],[-3.21979857],[12.77994864]]))
        else:
            for idx, cimg in enumerate(self.calib_images):
                print("Processing Calibration images ...", idx)
                src = cv2.imread(cimg)
                img = cv2.cvtColor(src.copy(), cv2.COLOR_BGR2GRAY)

                ret, corners = cv2.findChessboardCorners(img, self.CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

                if ret == True:
                    self.objpoints.append(self.objp)
                    corners2 = cv2.cornerSubPix(img, corners, (11,11),(-1,-1), self.criteria)
                    self.imgpoints.append(corners2)
                    
            _, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, img.shape[::-1], None, None)
            # 각각의 체커 이미지로 부터 구한 카메라 intrinsic 행렬, 렌즈 왜곡 계수, Rotation 벡터, Translation 벡터등을 멤버변수로 대입해 넣습니다.

    def printCalibMat(self):
        '''
        Calibration 매트릭스 출력용 함수 입니다. 

        '''
        print("Camera matrix : \n") # intrinsic 카메라 행렬을 의미합니다.
        print(self.mtx)
        print("dist : \n") # 렌즈 왜곡 계수입니다.
        print(self.dist) 
        print("rvecs : \n") # 캘리브레이션 이미지로부터 유래한 Rotation (회전) 행렬입니다.
        print(self.rvecs) 
        print("tvecs : \n") # 캘리브레이션 이미지로부터 유래한 Translation (이동) 행렬입니다.
        print(self.tvecs) 

if __name__ == "__main__":
    mycalib = MyCalib()
    mycalib.run_calibration(0)
    mycalib.printCalibMat()
