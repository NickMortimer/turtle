import xarray as xr
import rasterio as rio
from drone import P4rtk
import cv2 




img = "T:/drone/cal/DJI_0023.JPG"

dewarp =  [3662.860000000000,3656.260000000000,-20.920000000000,-6.990000000000,-0.262478000000,0.115395000000,0.000433033000,0.000161658000,-0.044218100000]
#dewarp = #[3706.080000000000,3692.930000000000,-34.370000000000,-34.720000000000,-0.271104000000,0.116514000000,0.001092580000,0.000348025000,-0.040583200000]
UtmCode = 32749
crs = f'epsg:{int(UtmCode)}'
drone =P4rtk(dewarp,crs)
data = cv2.imread(img)
h,  w = data.shape[:2] 
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(drone.K,drone.distCoeffs,(w,h),1,(w,h))
dst = cv2.undistort(data, drone.K,drone.distCoeffs , None, newcameramtx)
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
cv2.imwrite('T:/drone/cal/DJI_0023_calibresult.jpg',dst,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
mapx, mapy = cv2.initUndistortRectifyMap(drone.K,drone.distCoeffs,None,None,(w,h), cv2.CV_32FC1)
dst =cv2.remap(data, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderValue= 0, borderMode=cv2.BORDER_REFLECT) 
cv2.imwrite('T:/drone/cal/pairs/DJI_0023_aaaaaaaacalibresult2.jpg',dst,[int(cv2.IMWRITE_JPEG_QUALITY), 100])