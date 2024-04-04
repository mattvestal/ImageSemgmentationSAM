import numpy as np
import numpy.fft as fft
import scipy.signal
import cv2


###Scaling/demosaicing functions

###fft based scaling (recommended) -> generally better
def fftScale(smallimage,scalex,scaley):
    upscale = np.zeros((scalex,scaley,smallimage.shape[2]),dtype=complex)
    for k in range(smallimage.shape[2]):
        imSlice = smallimage[:,:,k]
        pad = np.zeros((scalex,scaley),dtype=complex)
        freqs = fft.fft2(imSlice)
        freqs = fft.fftshift(freqs)
        offy = int(pad.shape[0]/2) - int(freqs.shape[0]/2)
        offx = int(pad.shape[1]/2) - int(freqs.shape[1]/2)
        for i in range(freqs.shape[0]):
            for j in range(freqs.shape[1]):
                pad[i+offy,j+offx] += freqs[i,j]
        scaled = fft.fftshift(pad)
        scaled = fft.ifft2(scaled)

        upscale[:,:,k] += scaled
    
    return np.real(upscale)



###Bilinear interpolation
def interp(imagemosaic):#,correction,calCurve):
    offsets = [[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]]
    newIm = np.zeros((imagemosaic.shape[0],imagemosaic.shape[1],16))
    for k in range(len(offsets)):
        sp0 = np.zeros_like(imagemosaic)
        off = offsets[k]
        for i in range(off[0],sp0.shape[0],4):
            for j in range(off[1],sp0.shape[1],4):
                sp0[i,j]+=1
        filtered = sp0*imagemosaic

        F = np.array([[1,2,3,4,3,2,1],
                    [2,4,6,8,6,4,2],
                    [3,6,9,12,9,6,3],
                    [4,8,12,16,12,8,4],
                    [3,6,9,12,9,6,3],
                    [2,4,6,8,6,4,2],
                    [1,2,3,4,3,2,1]])
        H = F/16
        Spec0 = scipy.signal.convolve2d(filtered,H,'same')
        
        #print(Spec0.shape)
        newIm[:,:,k]+=Spec0

    #newIm = newIm/calCurve
            

    return newIm

#mask function for SD
def applymask(imagemosaic,mask):
    maskOut = np.zeros_like(imagemosaic)
    for i in range(imagemosaic.shape[2]):
        maskOut[:,:,i]+=imagemosaic[:,:,i]*mask
    return maskOut

###Spectral Difference
def SDInterp(imagemosaic):
    WB = interp(imagemosaic)
    offsets = [[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]]
    newMasks = np.zeros((imagemosaic.shape[0],imagemosaic.shape[1],16))
    output = np.zeros((imagemosaic.shape[0],imagemosaic.shape[1],16))
    for k in range(len(offsets)):
            sp0 = np.zeros_like(imagemosaic)
            off = offsets[k]
            for i in range(off[0],sp0.shape[0],4):
                for j in range(off[1],sp0.shape[1],4):
                    sp0[i,j]+=1

            newMasks[:,:,k] += sp0

    F = np.array([[1,2,3,4,3,2,1],
        [2,4,6,8,6,4,2],
        [3,6,9,12,9,6,3],
        [4,8,12,16,12,8,4],
        [3,6,9,12,9,6,3],
        [2,4,6,8,6,4,2],
        [1,2,3,4,3,2,1]])
    H = F/16

    for k in range(WB.shape[2]):
        Spec0 = WB[:,:,k]
        kbs = np.zeros((imagemosaic.shape[0],imagemosaic.shape[1],16))
        #slide = np.zeros((imagemosaic.shape[0],imagemosaic.shape[1]))
        for k1 in range(newMasks.shape[2]):
            sp0 = newMasks[:,:,k1]
            filtered = sp0*imagemosaic    
            kHat = (Spec0*sp0) - filtered
            kBar = scipy.signal.convolve2d(kHat,H,'same')

            kbs[:,:,k1] += kBar
        
        cbar = WB-kbs
        #print(newMasks[k].shape)
        refmask = np.dstack([newMasks[:,:,k]]*16)
        #print(refmask.shape)
        cbar = cbar*refmask#applymask(cbar,newMasks[:,:,k])
        #print(cbar.shape)
        output+=cbar
    
          
    return output



###Feature extractions
###NDVI
def NDVI(image):
    #out = np.zeros((image.shape[0],image.shape[1]))
    blank = image.copy()
    R = np.median(blank[:,:,0:4],axis=2)
    NIR = np.median(blank[:,:,11:15],axis=2)
    ranged = np.max(blank,axis=2) - np.min(blank,axis=2)
    R = cv2.GaussianBlur(R,(5,5),1)
    NIR = cv2.GaussianBlur(NIR, (5,5),2)
    #R = np.float32(R)
    #NIR = np.float32(NIR)
    #R = cv2.bilateralFilter(R,9,75,75)
    #NIR = cv2.bilateralFilter(NIR,9,75,75)


    veg = (NIR-R) / (R+NIR)
    #veg = veg*ranged

    ndvi = veg>np.mean(veg)+0.1*np.std(veg)
    return ndvi,veg



def midFeat(image):
    mid = image[:,:,3:9]
    midplus = image[:,:,2:8]
    deriv = mid-midplus
    

    feature = np.median(deriv,axis=2)

    return feature


def plasticFind(image,bands):
    b1 = image[:,:,6:-1]
    b2 = bands[6:-1]
    #b2 = [1,2,3,4,5,6,7,8,9,10]
    feat = np.zeros((image.shape[0],image.shape[1]))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            spec = b1[i,j]
            grad = np.polyfit(b2,spec,deg=1)[0]
            feat[i,j]+=grad
    return feat


def vegFind(image,bands):
    b1 = image[:,:,3:7]
    b2 = bands[3:7]
    feat = np.zeros((image.shape[0],image.shape[1]))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            spec = b1[i,j]
            grad = np.polyfit(b2,spec,deg=1)[0]
            feat[i,j]+=grad
    return feat




####Calibration funtions
###Apply homography
def applyHom(Image,H,w,h):
    warped = cv2.warpPerspective(Image, H, (w, h))

    return warped

###Find matched images
###find timestamps and index of high frequency capture that closest matches  low freq
def tsync(lowfreq,highfreq):
    stamps = []
    for i in range(len(lowfreq)):
        ref1 = np.where(highfreq>=lowfreq[i])[0] ###use the higher value as the multispec fov comes first
        ref2 = np.where(highfreq<lowfreq[i])[0]
        if len(ref1)>1:
            refH = ref1[0]
            stamps.append(refH)
        if len(ref2)>1:
            refL = ref2[0]
        
        ref = min(refH, refL)
        #stamps.append(ref)
        
    stamps = np.array(stamps)

    return stamps


def ffield(imageMosaic,light,dark):
    output = (imageMosaic-dark)*np.mean(light-dark)/(light-dark)

    return output



def undistort(imageMosaic,therm,color): ###undistort thermal and multispectral images -- uses calibration files
    THnewMat = np.load('CamMatrices\\Therm\\newCamFull.npy')
    THroi = np.load('CamMatrices\\Therm\\roiFull.npy')
    THmtx = np.load('CamMatrices\\Therm\\mtxFull.npy')
    THdist = np.load('CamMatrices\\Therm\\distFull.npy')

    MSnewMat = np.load('CamMatrices\\MultiSpec\\newcam.npy')
    MSroi = np.load('CamMatrices\\MultiSpec\\roi.npy')
    MSmtx = np.load('CamMatrices\\MultiSpec\\mtx.npy')
    MSdist = np.load('CamMatrices\\MultiSpec\\dist.npy')

    CnewMat = np.load('CamMatrices\\Color\\newCamFull3b5.npy')
    Croi = np.load('CamMatrices\\Color\\roiFull3b5.npy')
    Cmtx = np.load('CamMatrices\\Color\\mtxFull3b5.npy')
    Cdist = np.load('CamMatrices\\Color\\distFull3b5.npy')



    dstTH = cv2.undistort(therm[:,:,0], THmtx, THdist, None, THnewMat)
    x, y, w, h = THroi
    dstTH = dstTH[y:y+h, x:x+w]

    dstMS = cv2.undistort(imageMosaic, MSmtx, MSdist, None, MSnewMat)
    x, y, w, h = MSroi
    dstMS = dstMS[y:y+h, x:x+w]

    dstC = cv2.undistort(color, Cmtx, Cdist, None, CnewMat)
    x, y, w, h = Croi
    dstC = dstC[y:y+h, x:x+w]

    return dstMS, dstTH, dstC
        


#generate full image
def stack(dstMS,imwarp):
    multiSpec = np.zeros_like(dstMS)
    for k in range(multiSpec.shape[2]):
        multiSpec[:,:,k]+=dstMS[:,:,k]*(imwarp!=0)
    #plt.imshow(multiSpec[:,:,-1],'gray')
    fullIm = np.dstack([multiSpec,imwarp])

    return fullIm


##historam stretch
def image_histogram_equalization(image, x, number_bins=2**10):

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = (number_bins-1) * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), x*bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf




#miscelaneous functions
def dsample(imageMosaic,correction,calCurve):
    M = imageMosaic.shape[0]//272
    N = imageMosaic.shape[1]//512
    tiles = [imageMosaic[x:x+M,y:y+N] for x in range(0,imageMosaic.shape[0],M) for y in range(0,imageMosaic.shape[1],N)]
    arr3dbase = np.zeros((272,512,16))
    specs = []
    for i in range(0,len(tiles),512):
        specs.append(tiles[i:i+512])

    for i in range(272):
        for j in range(512):
            NIR = np.ravel(specs[i][j])
            arr3dbase[i,j,:] += NIR
    
    ImageMosaic= np.zeros((arr3dbase.shape[0],arr3dbase.shape[1],arr3dbase.shape[2]-1))
    for i in range(arr3dbase.shape[0]):
        for j in range(arr3dbase.shape[1]):
            spec = arr3dbase[i,j,:]#/calCurve
            ImageMosaic[i,j,:] += np.dot(correction,spec)

    ImageMosaic = ImageMosaic/calCurve

    return ImageMosaic