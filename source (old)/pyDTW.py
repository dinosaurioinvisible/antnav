import numpy as np
from numpy import transpose,matmul,sqrt
from matplotlib import pyplot as plt 


class DTW:

    def __init__(self, img_shape):
        self.img_shape = img_shape
        LP, HP = self.get_filters(img_shape)
        self.LP_vert, self.LP_hor = LP
        self.HP_vert, self.HP_hor = HP
    
    def get_filters(self, img_shape):
        # For this benchmark code the width of the image is first and the height second
        (w,h) = img_shape
        # Setup filters as matrizes
        LP_vert = np.zeros((int(h/2),h))
        HP_vert = np.zeros((int(h/2),h))
        # put the right numbers to use
        for i in range(int(h/2)):
            i_colp = 2*i
            # 
            LP_vert[i,i_colp] = 1/sqrt(2)
            LP_vert[i,i_colp+1] = 1/sqrt(2)
            
            HP_vert[i,i_colp] = 1/sqrt(2)
            HP_vert[i,i_colp+1] = -1/sqrt(2)
            
        LP_hor = np.zeros((int(w/2),w))
        HP_hor = np.zeros((int(w/2),w))
        
        for j in range(int(w/2)):
            j_colp = 2*j
            # 
            LP_hor[j,j_colp] = 1/sqrt(2)
            LP_hor[j,j_colp+1] = 1/sqrt(2)
            
            HP_hor[j,j_colp] = -1/sqrt(2)
            HP_hor[j,j_colp+1] = 1/sqrt(2)
        # scaling of matrices by 1/sqrt(2) as shown in 
        # https://www.ijstr.org/final-print/sep2014/Discrete-Wavelet-Transforms-Of-Haars-Wavelet-.pdf
        return (LP_vert,LP_hor),(HP_vert,HP_hor)
        
    def dwt_haar(self, img):
        '''
        For high res use Vertical -> HL (1/2 deg per pixel)
        For low res use Horizontal -> HH (1 deg per pixel or below)
        '''
        # Filter rows
        I_gr = matmul(self.LP_vert, img)
        I_hr = matmul(self.HP_vert, img)
        
        # Filter columns
        LL = matmul(self.LP_hor,transpose(I_gr))
        HH = matmul(self.HP_hor,transpose(I_hr))
        HL = matmul(self.LP_hor,transpose(I_hr))
        LH = matmul(self.HP_hor,transpose(I_gr))
        
        # transpose back
        LL = abs(transpose(LL))
        HH = abs(transpose(HH))
        HL = abs(transpose(HL))
        LH = abs(transpose(LH))
        
        return LL,LH,HL,HH

# # How to use:
# # load grayscale image as np array 
# img = np.random.randint(0,255,(360,360))
# # create filters according to image dimension
# dtw = DTW(img.shape)

# # perform transform
# Approximation,Horizontal,Vertical,Diagonal = dtw.dwt_haar(img)

# # oraginze resulting coefficients for display
# left = np.concatenate((Approximation,Horizontal),axis=0)
# right = np.concatenate((Vertical,Diagonal),axis=0)
# res = np.concatenate((left,right),axis=1)
# # display
# fig,ax =  plt.subplots(nrows=1 )
# ax.imshow(res,extent=[0,1,0,1])
# plt.show()     
