import numpy as np
import cv2 as cv



class Unwraper():

    def __init__(self, img) -> None:
    
        # TODO: need to work out how to crop properly here or crop lafter the unwrap
        # self.cropBlock=int((int(img.shape[1])-int(img.shape[0]))/2)
        # img=img[:,self.cropBlock:-self.cropBlock]

        # distance to the centre of the image
        ##########################
        # The 1.28 constant is specific to the PixPro camera!!!!
        self.offset=int(round((img.shape[0]/1.28)))

        #IMAGE CENTER
        Cx = img.shape[0]/2
        Cy = img.shape[1]/2

        #RADIUS OUTER
        R =- Cx

        #DESTINATION IMAGE SIZE
        Wd = int(abs(2.0 * (R / 2) * np.pi))
        Hd = int(abs(R))

        #BUILD MAP
        self.xmap, self.ymap = self.buildMap(Wd, Hd, R, Cx, Cy)
    
    #MAPPING
    def buildMap(self, Wd, Hd, R, Cx, Cy):
                    
        ys=np.arange(0,int(Hd))
        xs=np.arange(0,int(Wd))
        
        rs=np.zeros((len(xs),len(ys)))
        rs=R*ys/Hd
        
        thetas=np.expand_dims(((xs-self.offset)/Wd)*2*np.pi,1)

        map_x=np.transpose(Cx+(rs)*np.sin(thetas)).astype(np.float32)
        map_y=np.transpose(Cy+(rs)*np.cos(thetas)).astype(np.float32)
        return map_x, map_y
    
    #UNWARP
    def unwarp(self, img):
        return cv.remap(img, self.xmap, self.ymap, cv.INTER_NEAREST)
       
