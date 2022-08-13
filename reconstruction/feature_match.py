import cv2
import numpy as np
from calibration import *
import open3d as o3d

'''
for Feature Matching & 3D reconstruction
2019104035 장서연 2022 - 1 로봇센서데이터처리 중간고사 대체과제
'''

class MyFeatureMatch:
    def __init__(self, calib_mode):

        # 이미지 로드 하는 부분
        self.left_src = cv2.imread("../stereo/l_img2.jpg") # 캘리브레이션 되지 않은 원본 left 이미지를 받아오는 부분입니다.
        self.right_src = cv2.imread("../stereo/r_img2.jpg") # 캘리브레이션 되지 않은 원본 right 이미지를 받아오는 부분입니다.

        self.left_img = None  # 캘리브레이션 된 이후 left 이미지를 받아오는 부분입니다.
        self.right_img = None # 캘리브레이션 된 이후 right 이미지를 받아오는 부분입니다.

        self.img_h, self.img_w = self.left_src.shape[:2] # 이미지 의 세로, 가로 길이를 받아오는 부분입니다.

        # 캘리브레이터 정의 
        self.calibrator = MyCalib()

        # 캘리브레이션 수행
        # 휴대전화 카메라로 찍은 해상도 높은 이미지를 사용하고 있기 때문에, 그냥 캘리브레이션 함수를 적용해주게 되면 시간이 오래 걸립니다.
        # 그래서 mode 변수를 추가하여 calibration 옵션을 넣었습니다.
        # 1. calib_mode == 1로 설정시 제가 찾은 pre-defined calib coefficient matrix를 불러옵니다.
        # 2. calib_mode == 0 또는 다른 숫자로 설정시 run_calibration() function을 수행하여 이미지로 부터 calibration 매트릭스를 구합니다.
        self.calibrator.run_calibration(mode=calib_mode) 


        # 캘리브레이션 결과로 나온 카메라 intrinsic Matrix의 0행 0열에 해당하는 focal length 값은 자주 사용하는 값이므로 따로 focal_length 라는 멤버 변수로 지정해 줍니다.
        self.focal_length = self.calibrator.mtx[0][0] 

        
        # detector 정의
        self.detector = cv2.SIFT_create() # feature detector 로는 SIFT 알고리즘을 사용합니다.

        # matcher 정의
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params) 
        # Feature Matcher 로는 FlannBasedMatcher 를 사용하고, 교수님께서 사용하신 파라미터를 그대로 적용했습니다.

    def calibStereoImage(self):
        '''
        캘리브레이션 실행 결과로 나온 intrinsic 행렬, 왜곡 계수 행렬을 이용하여 왜곡된 이미지를 펴주는 함수에 해당합니다.
        '''
        print("Now Undistort Stereo Images...")
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.calibrator.mtx, self.calibrator.dist,(self.img_w,self.img_h),1,(self.img_w,self.img_h))

        # newcameramtx 행렬로 카메라 매트릭스를 수정해줍니다.
        self.calibrator.mtx = newcameramtx

        self.left_img = cv2.undistort(self.left_src, self.calibrator.mtx, self.calibrator.dist, None, newcameramtx) # 왼쪽 이미지 왜곡 펴주기
        self.right_img = cv2.undistort(self.right_src, self.calibrator.mtx, self.calibrator.dist, None, newcameramtx) # 오른쪽 이미지 왜곡 펴주기

        x, y, w, h = roi
        self.left_img = self.left_img[y:y+h, x:x+w]
        self.right_img = self.right_img[y:y+h, x:x+w]

        # focal_length를 수정해줍니다. 또 이미지를 roi 대로 자르면서 크기가 수정되었기 때문에 img_h,w 변수값도 수정해줍니다.
        self.img_h, self.img_w = self.left_img.shape[:2]
        self.focal_length = self.calibrator.mtx[0][0]


    def extractKeypoint(self, match_threshold):
        '''
        SIFT Detector 가 스테레오 이미지를 비교하여 feature 를 뽑아내고, 리턴값으로 스테레오 이미지의 피처 쌍들을 반환합니다. 
        match_threshold : 유의미한 피처들만 뽑아내기 위한 임계값, 일반적으로 0.8 을 사용합니다.
        0.8 보다 작아지면 더욱 엄격하게 feature 를 제한하고, 0.8 보다 크게 하면 잘못 매칭된 feature 들이 많이 포함되게 됩니다.
        '''
        print("Now Extracting KeyPoints...")
        keypoint_l, desc_l = self.detector.detectAndCompute(self.left_img, None)
        keypoint_r, desc_r = self.detector.detectAndCompute(self.right_img, None)
        matches = self.matcher.knnMatch(desc_l, desc_r, k=2)

        pts_l = []
        pts_r = []

        for i, (l_match, r_match) in enumerate(matches): 
            if l_match.distance < match_threshold * r_match.distance: 
                kpl = keypoint_l[l_match.queryIdx].pt
                kpr = keypoint_r[l_match.trainIdx].pt
                pts_l.append(kpl)
                pts_r.append(kpr) 
            
        return pts_l, pts_r


    def draw3Dpoints(self, pc_points, pc_colors):
        '''
        3d 포인트 시각화를 위한 함수입니다.
        '''
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_points)
        pcd.colors = o3d.utility.Vector3dVector(pc_colors)
        o3d.visualization.draw_geometries([pcd])


    def detectNmatch(self, threshold, V_OFFSET, U_OFFSET, base_line):
        '''
        V_OFFSET: 두 스테레오 이미지의 v 방향 픽셀 좌표 값 차이 오프셋 변수입니다. 이 값을 작게 두어야 같은 높이에 있는 이미지 상의 키포인트들을 사용하게 됩니다.
        X_OFFSET : 왼쪽 스테레오 이미지의 u 방향 픽셀 좌표 - 오른쪽 스테레오 이미지의 v 방향 픽셀 좌표입니다. 이 값이 너무 작거나 음수일 경우 잘못된 feature 쌍들을 구한 것이므로
        base_line: 두 스테레오 이미지가 카메라 좌표 상에서 얼마만큼 떨어져 있는지를 나타내는 값입니다. 단위는 mm 입니다.
        '''
        pc_points = [] # open3D의 point cloud 에 들어갈 points 좌표입니다.
        pc_colors = [] # 그에 맞는 색깔들이 들어가는 좌표입니다.

        pts_l, pts_r = self.extractKeypoint(threshold) 
        # theshold 범위내의 feature 쌍 들을 1차로 선별해 줍니다.

        for i in range(len(pts_l)):
            v_offset = pts_r[i][1] - pts_l[i][1] # image coordinate에서, v 좌표 픽셀 갯수의 차이
            u_offset = pts_l[i][0] - pts_r[i][0] # image coordinate에서, u 좌표 픽셀 갯수의 차이

            if(abs(v_offset) < V_OFFSET and u_offset > U_OFFSET ):
                # 만족하는 오프셋 내에 들어오는 feature 쌍들을 선별해, 3d reconstruction 해주는 부분입니다.
            
                print("Now Find Feature and 3D RECONSTRUCTION")
                
                # triangulation 에 기초하여, world 좌표계에서 물체까지의 거리를 구하는 부분 입니다.
                Z = (base_line * self.focal_length) / (u_offset)

                # 아래는 이미지, 카메라, 월드 좌표계 관계를 통한 3d 좌표 X, Y 를 찾는 과정입니다.
                '''
                img_coord = K * camera_coord
                camera_coord = K_inv * img_coord ...(1)

                camera_coord = 1 / Z * [R|T] * world_coord
                world_coord = Z * camera_coord (rotation, translation 없다고 가정시) ...(2)

                (1) 과 (2) 를 연립하면,
                world_coord = Z * K_inv * img_coord
                '''
                K_inv = np.linalg.inv(self.calibrator.mtx)

                img_coord = np.array([pts_l[i][0], pts_l[i][1], 1])
                camera_coord = np.dot(K_inv, img_coord)
                world_coord = camera_coord * Z
                pc_points.append(np.array(world_coord, dtype=np.float64)) # 3차원 포인트 float type 추가

                # OpenCV 는 (B, G, R) 좌표를 사용하지만, Open3D 는 (R, G, B ) 좌표계를 사용하기 때문에, 이 좌표계를 변환해 주어야 합니다.
                color_bgr2rgb = np.array(self.left_img[int(pts_l[i][1]), int(pts_l[i][0])] / 255.0 ) # 0~1 사이로의 스케일링
                color_bgr2rgb = color_bgr2rgb[...,::-1] # 색상공간 변화
                pc_colors.append(np.array(color_bgr2rgb))

                '''
                아래는 feature point 주변에서의 부분적인 stereo match 를 하기 위한 코드입니다.
                feature point 만을 3D 로 바꾸면, 점들이 매우 sparse 하게 찍혀서 물체를 알아보기 어렵습니다.
                feature point 주변의 매우 작은 영역은 feature point 간의 disparity 만큼 거리를 두고 떨어져있을 가능성이 높습니다.
                그 patch 내의 점들에 모두 스테레오 매칭을 적용하여 3차원 점으로 바꿀 수 있다면, 더욱 dense한 3d reconstruction이 가능할 것입니다.
                '''
                print("Now Find DisPatch and 3D RECONSTRUCTION")
                img_Lpatch = self.localStereoMatch(pts_l[i][0], pts_l[i][1], 50, 10, self.left_img) # 왼쪽 패치 이미지
                img_Rpatch = self.localStereoMatch(pts_r[i][0], pts_r[i][1], 50, 10, self.right_img) # 오른쪽 패치 이미지.

                img_Dispatch = abs(img_Lpatch - img_Rpatch) 
                '''
                원래대로라면 두 이미지간 스테레오 매칭 (가령, stereoBM 같은) 을 수행해 비슷한 픽셀을 찾아야 하지만,
                매우 작은 영역끼리 비교하는 것이라 비슷한 값이 많게되고, 그래서 스테레오 매칭이 제대로 이루어지지 않을 가능성이 높습니다.
                그대신 왼쪽 이미지에서 오른쪽 이미지의 픽셀 rgb 값을 뺍니다. 이 값을 저는 img_Dispatch 로 명명했습니다.
                만약 서로 비슷한 픽셀 값을 가진다면 매우 작은 값이 구해질 것이고, 서로 다른 픽셀 값을 가진다면 비교적 큰 값이 구해질 것입니다.
                '''

                # cv2.imshow("patch_L", np.array(img_Lpatch, dtype=np.uint8))
                # cv2.imshow("patch_R", np.array(img_Rpatch, dtype=np.uint8))
                # cv2.imshow("img_Dispatch", np.array(img_Dispatch, dtype=np.uint8))

                # cv2.waitKey()
                # cv2.destroyAllWindows()

                for patch_v in range(img_Dispatch.shape[0]):
                    for patch_u in range(img_Dispatch.shape[1]):
                        if np.all(img_Dispatch[patch_v,patch_u] < 10): # DisPatch 값이 임계치보다 작은 값이라면,
                            # 그 픽셀 들에 대해서도 triangulation 을 통한 3d reconstruction 을 수행해 줍니다.
                            Z = (150 * self.focal_length) / (u_offset)
                            K_inv = np.linalg.inv(self.calibrator.mtx)
                            local_u, local_v = pts_l[i][0] - 50 + patch_u, pts_l[i][1] - 10 + patch_v

                            if local_u < self.img_w and local_v <self.img_h:
                                img_coord = np.array([local_u, local_v, 1])
                                camera_coord = np.dot(K_inv, img_coord)
                                world_coord = camera_coord * Z
                                pc_points.append(np.array(world_coord, dtype=np.float64))


                                color_bgr2rgb = np.array(self.left_img[int(local_v), int(local_u)] / 255.0 )
                                color_bgr2rgb = color_bgr2rgb[...,::-1]
                                pc_colors.append(color_bgr2rgb)

        return pc_points, pc_colors

    def localStereoMatch(self, pointx, pointy, margin_x, margin_y, image):
        '''
        image: 원본 이미지를 받아오는 파라미터 입니다.
        pointx, pointy : 원본이미지에서 키포인트 좌표에 해당하는 점입니다.
        margin_x, margin_y: 키포인트로 부터 얼마만큼의 patch 영역을 떼올지 지정하는 파라미터입니다.
        '''
        pointx = int(pointx)
        pointy = int(pointy)

        new_img = np.zeros((2*margin_y, 2*margin_x, 3), dtype=np.int16)

        x_range = range(pointx - margin_x, pointx + margin_x + 1)
        y_range = range(pointy - margin_y, pointy + margin_y + 1)
 
        if x_range[0] < 0:
            x_range = range(0, pointx + margin_x + 1)
            
        elif x_range[-1] > self.img_w - 1:
            x_range = range(pointx - margin_x, self.img_w)

        if y_range[0] < 0:
            y_range = range(0, pointy + margin_y + 1)

        elif y_range[-1] > self.img_h - 1:
            y_range = range(pointy - margin_y, self.img_h)

        print("got point from DisPatch\n y: ", y_range[-1] - y_range[0], "x: ", x_range[-1]- x_range[0])
        
        new_img[0 : y_range[-1]-y_range[0], 0:x_range[-1]- x_range[0]] = image[y_range[0] : y_range[-1], x_range[0] : x_range[-1]]
        
        return new_img
    
if __name__ == "__main__":
    mfnm = MyFeatureMatch(calib_mode = 0) # mode 1 == fast, mode 0 == slow
    mfnm.calibStereoImage()
    pc_points, pc_colors= mfnm.detectNmatch(threshold=0.8, V_OFFSET= 20, U_OFFSET=140, base_line= 150)
    mfnm.draw3Dpoints(pc_points, pc_colors)
