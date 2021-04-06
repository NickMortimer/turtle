import numpy as np
import pandas as pd
import ast
import cv2
from shapely.geometry import point
from shapely.geometry import Polygon
import geopandas as gp
CCDSIZE =0.0132/5472
class P4rtk:
    def __init__(self,dewarpdata,crs,imagewidth=5472,imageheight=3648,pixelsize=CCDSIZE):
        self.imagewidth =imagewidth
        self.imageheight = imageheight
        self.pixelsize = pixelsize
        self.crs = crs
        self.dewarp(dewarpdata)
        
    def setdronepos(self,easting,northing,altitude,gimblepitch,gimbleroll,gimbleyaw):
        omga = np.deg2rad(gimblepitch)
        phi = np.deg2rad(gimbleroll)
        k = np.deg2rad(gimbleyaw)
        self.yaw = gimbleyaw
        self.pitch = omga
        self.roll = phi
        rx = np.array([[1.,0,0],
                       [0,np.cos(omga),-np.sin(omga)],
                       [0,np.sin(omga),np.cos(omga)]])
        ry = np.array([[np.cos(phi),0,np.sin(phi)],
                       [0,1,0],
                       [-np.sin(phi),0,np.cos(phi)]])
        rz = np.array([[np.cos(k),-np.sin(k),0],
                       [np.sin(k),np.cos(k),0],
                       [0,0,1]])
        self.R2 = rx.dot(ry).dot(rz)
        self.Translation = np.array([0,0,-altitude])
        self.R= np.array([[np.cos(k)*np.cos(phi),                                 -np.sin(k)*np.cos(phi),np.sin(phi),0],
                        [np.cos(k)*np.sin(omga)*np.sin(phi)+np.sin(k)*np.cos(omga),np.cos(k)*np.cos(omga)-np.sin(k)*np.sin(omga)*np.sin(phi),-np.sin(omga)*np.cos(phi),0],
                        [np.sin(k)*np.sin(omga)-np.cos(k)*np.cos(omga)*np.sin(phi),np.sin(k)*np.cos(omga)*np.sin(phi)+np.cos(k)*np.sin(omga),np.cos(omga)*np.cos(phi),-altitude],
                        [0,0,0,1]])
        self.Rinverse = np.linalg.inv(self.R)
        self.easting = easting
        self.northing = northing
        self.altitude = altitude
        self.tocamm =    self.Km @ self.R 
        self.tocamp =    self.Kp @ self.R
        

    def realwordtocamera(self,easting,northing,evevation=0):
        cam = self.tocamm.dot([easting-self.easting,northing-self.northing,evevation,1])
        cam = (cam/cam[-1])/self.pixelsize
        return cam[:-1]
        
    def cameratorealworld(self,x,y):
        #x=int(x)
        #y=int(y)
        #camcoord =np.array([self.map1[y,x]*self.pixelsize,self.map2[y,x]*self.pixelsize,1])
        camcoord = np.array([x*self.pixelsize,y*self.pixelsize,1])
        tp =self.R[0:3,0:3].T.dot(self.Translation)
        pground =self.R[0:3,0:3].T.dot(self.Kminverse).dot(camcoord)
        #scale focal length to plain
        pground = (pground *tp[2]/pground[2]) -tp
        pground[0] =pground[0] +self.easting        
        pground[1] =  pground[1]+self.northing
        return pground
        
    # get the sensor orientation in North-East-Down coordinates
    # pose is a yaw/pitch/roll tuple of angles measured for the DLS
    # ori is the 3D orientation vector of the DLS in body coordinates (typically [0,0,-1])
    def get_orientation(self):
        """Generate an orientation vector from yaw/pitch/roll angles in radians."""
        yaw, pitch, roll = self.pose
        c1 = np.cos(-yaw)
        s1 = np.sin(-yaw)
        c2 = np.cos(-pitch)
        s2 = np.sin(-pitch)
        c3 = np.cos(-roll)
        s3 = np.sin(-roll)
        Ryaw = np.array([[c1, s1, 0], [-s1, c1, 0], [0, 0, 1]])
        Rpitch = np.array([[c2, 0, -s2], [0, 1, 0], [s2, 0, c2]])
        Rroll = np.array([[1, 0, 0], [0, c3, s3], [0, -s3, c3]])
        R = np.dot(Ryaw, np.dot(Rpitch, Rroll))
        return R        
        
    #print(cam)

        ##"DewarpData": " 2020-05-20;3706.080000000000,3692.930000000000,-34.370000000000,-34.720000000000,-0.271104000000,0.116514000000,0.001092580000,0.000348025000,-0.040583200000",
    #https://github.com/dronemapper-io/dji-dewarp

    def dewarp(self,dewarpdata):
        self.K = np.array([[dewarpdata[0],0,(self.imagewidth/2)+dewarpdata[2]],
                [0,dewarpdata[1],(self.imageheight/2)+dewarpdata[3]],
                [0,0,1]])
        self.Kp = np.array([[dewarpdata[0],0,(self.imagewidth/2)+dewarpdata[2],0],
                        [0,dewarpdata[1],(self.imageheight/2)+dewarpdata[3],0],
                        [0,0,1,0]])
        self.Km = np.array([[dewarpdata[0]*self.pixelsize,0,(self.imagewidth/2+dewarpdata[2])*self.pixelsize,0],
                        [0,dewarpdata[1]*self.pixelsize,(self.imageheight/2+dewarpdata[3])*self.pixelsize,0],
                        [0,0,1,0]])
        self.Kminverse = np.linalg.inv(self.Km[0:3,0:3])
        self.distCoeffs = dewarpdata[4:]
        
    def calculateposition(self,x,y):
        def to_real_wrold(index,altitude,focallen):
            return (index/focallen)*altitude
        x = x-self.K[0,2]
        y= +  y- self.K[1,2] 
        x = to_real_wrold(x,self.altitude,self.K[0,0])
        y = to_real_wrold(y,self.altitude,self.K[1,1])
        xx = (x * np.cos(self.yaw) +  y * np.sin(np.deg2rad(-self.yaw)))+ self.easting 
        yy = (-x * np.sin(self.yaw)  +  y  * np.cos(np.deg2rad(-self.yaw)))+ self.northing
        return np.array([xx,yy])

    def dewappoints(self,x,y):
        return self.map1[y,x],self.map2[y,x]

    def realworld(self,x,y,):
        pass
    
    def getimagepolygon(self):
        xw =self.imagewidth/2
        yw=self.imageheight/2
        points =[[-xw,yw],[-xw,-yw],[xw,-yw],[xw,-yw]]
        polydata =[self.cameratorealworld(pos[0],pos[1]) for pos in points]
        return gp.GeoSeries(Polygon(polydata),crs=self.crs)
        
    def jpegtoreal(self,points):
        xy_undistorted = cv2.undistortPoints(points, self.Kp[0:3,0:3], self.distCoeffs, None, self.Kp[0:3,0:3])
        return xy_undistorted.squeeze()

    
if __name__ == '__main__':
    data = np.array([3706.080000000000,3692.930000000000,-34.370000000000,-34.720000000000,-0.271104000000,0.116514000000,0.001092580000,0.000348025000,-0.040583200000])
    drone = P4rtk(data)
    gcp =pd.read_csv("T:/drone/raw/process/labelmematchup_final_pointpos.csv")
    gcp = gcp[gcp.label.isin(['gcp'])]
    gcp.points = gcp.points.apply(ast.literal_eval)
    jpegpoints = np.array(gcp.points.apply(lambda x:np.array(x[0],float)).tolist())
    gcp[['JpegX','JpegY']] =np.floor(jpegpoints)
    corrected =drone.jpegtoreal(jpegpoints)
    gcp[['DewarpX','DewarpY']] =corrected
    for index,row in gcp.iterrows():
        #drone.setdronepos(row.Eastingrtk,row.Northingrtk,row.RelativeAltitude,-5,0,row.GimbalYawDegree)
        drone.setdronepos(row.Eastingrtk,row.Northingrtk,row.EllipsoideHight+14.772+1.5,(90+row.GimbalPitchDegree)*-1,0,row.GimbalYawDegree+10)
        print(f'JPG:{drone.cameratorealworld(row.JpegX,row.JpegY)} warp:{drone.cameratorealworld(row.DewarpX,row.DewarpY)}')
        
    
    roll = 0
    yaw = 90#-71.9
    pitch = 20
    drone.setdronepos(802621.0124,7567239.79844571,110,0,0,170.5)
    
    pixels =[[2078.361603,-9.819796173],
            [2006.374942,670.0401841],
            [1945.114351,1372.331906],
            [1877.123314,2073.95085],
            [1805.55692,2787.531148]]
    dronepos = [[802621.0124,7567239.798],
              [802617.8474,7567220.614],
              [802614.4944,7567200.755],
              [802611.3814,	7567181.235],
              [802608.023,7567161.772]]
    heading = [-170.5,-170,-170.8,-170.5,-170]

    for d,pix,h in zip(dronepos,pixels,heading):
        drone.setdronepos(d[0],d[1],102,0,0,h)
        print(f'warp:{drone.cameratorealworld(pix[0],pix[1])}')
        
    d =[drone.cameratorealworld(pos[0],pos[1]) for pos in pixels ]
    speed =np.power(np.sum(np.power(np.diff(d,axis=0),2))/4,0.5)/2.5
    print(f'warp:{drone.cameratorealworld(2078.361603,-9.819796173)}')
    print(f'warp:{drone.cameratorealworld(2078.361603,-9.819796173)}')
    print(f'warp:{drone.cameratorealworld(2078.361603,-9.819796173)}')
    print(f'warp:{drone.cameratorealworld(2078.361603,-9.819796173)}')
    
    drone.setdronepos(802617.8474,7567220.614,110,0,0,170.5)
    print(f'jpg:{drone.cameratorealworld(2119.67213,109.5081967)}')
    print(f'warp:{drone.cameratorealworld(2078.361603,-9.819796173)}')
    
    #jpg:[ 802612.23434216 7567187.59899552       0.        ]
    #warp:[ 802611.61166394 7567183.89099018       0.        ]
    
#1835.576923	2754.807692	1807.959126	2782.714366	802590.3348	7567129.807	802588.6399	7567190.571



    pos =drone.realwordtocamera(798217.3835,7555078.163)
    print(pos,drone.cameratorealworld(pos[0],pos[1]))
    for heading in range(0,360,10):
        print(f'heading:{heading}')
        drone.setdronepos(0,0,100,0,0,heading)
        for x in range(0,10):
            pos =drone.realwordtocamera(x*10,0)
            print(pos,drone.cameratorealworld(pos[0],pos[1]))

    
    
    
    	# 	cv::Mat cam1;
		# cam1 = cv::Mat::zeros(3, 3, CV_32FC1);
		# cam1.at<float>(0, 2) = 2736 - 10.100000000000;		// cX (5472x3648 Width / 2)
		# cam1.at<float>(1, 2) = 1824 + 27.290000000000;		// cY (5472x3648 Height / 2)
		# cam1.at<float>(0, 0) = 3678.870000000000;			// fx
		# cam1.at<float>(1, 1) = 3671.840000000000;			// fy
		# cam1.at<float>(2, 2) = 1;