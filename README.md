# Dispatch:**Dis**parity **patch** for Dense stereo matching
2022 RSDP Mid-term project
**Dis**parity **patch** for Dense stereo matching

## What is ‘DisPatch’ ?
I introduce a module called **DisPatch** designed to enable Dense 3D reconstruction with only two stereo images. It simply operates with the following two functions.

![](./img/1.png)

- **Keypoint 3D reconstruction**: It is a function that matches the keypoints of stereo images found by SIFT and converts them into 3D coordinate systems.

- **Local Patch 3D reconstruction**: This function extracts very small patches around the Keypoint of the two stereo images and then performs stereo matching partially on these patches. Using this, dense stereo matching with improved accuracy could be performed.

## How to run DisPatch?
```
Ubuntu 20.04
Python 3.8.10
OpenCV 4.5.5
Open3d 0.13.0 
```

If you download the file (midterm_3Dreconstruction), you can see that there are three folders inside.
```
cd reconstruction
python feature_match.py
```
1) Camera calibration
2) Stereo image undistortion
3) Keypoint 3d reconstruction
4) Local Patch 3D reconstruction 

After executing in order, you can even check the process of floating a point cloud on a 3D space with Open3D.

## Folder 
- calib: Folder containing 15 checkerboard calibration images
- stereo: Folders containing stereo images
- reconstruction: Folder where calibration.py and feature_matching.py exist


## Camera Calibration
Get the checkerboard image in ./calib and perform calibration. The method used at this time is Zhengyou Zhang's method.

## Stereo Image Undistortion
OpenCV's undistort function was used.

## SIFT, Knn Matcher Keypoint Extraction & 3D reconstruction
After searching only key points with SIFT and KnnMatcher, 3D reconstruction is performed.
![](./img/22.png)

## Local Patch 3D reconstruction - Dispatch algorithm to use Dense Matching
Apply the Dispatch algorithm designed for Dense Matching.

![](./img/24_disp.png)
![](./img/25_disp.png)
![](./img/26_disp.png)
![](./img/27_disp.png)
![](./img/28_disp.png)
![](./img/29_disp.png)

## Result
![](./img/30_disp.png)
![](./img/31_disp.png)
