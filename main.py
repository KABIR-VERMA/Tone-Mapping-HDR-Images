import numpy as np
import copy
import cv2
import math
import scipy.signal
import scipy.interpolate

def bilateral_approximation(data, edge ,sigmaS, sigmaR):
	
    inputHeight,inputWidth  = data.shape
    edgeMax = np.amax(edge)
    edgeMin = np.amin(edge)

    paddingXY = 5
    paddingZ = 5

	# allocate 3D grid
    downsampledWidth = math.floor( ( inputWidth - 1 ) / sigmaS ) + 2 * paddingXY + 1
    downsampledHeight = math.floor( ( inputHeight - 1 ) / sigmaS ) + 2 * paddingXY + 1
    downsampledDepth = math.floor( edgeMax - edgeMin / sigmaR ) + 2 * paddingZ + 1

    gridData = np.zeros( (downsampledHeight, downsampledWidth, downsampledDepth) )
    gridWeights = np.zeros( (downsampledHeight, downsampledWidth, downsampledDepth) )

	# compute downsampled indices
    (jj, ii) = np.meshgrid( range(inputWidth), range(inputHeight) )
    di = np.around( ii / sigmaS)  + paddingXY+1
    dj = np.around( jj / sigmaS ) + paddingXY+1
    dz = np.around( ( edge - edgeMin ) / sigmaR ) + paddingZ+1
    

    for k in range(0, dz.size):
        gridData[ int(di.flat[k]), int( dj.flat[k]), int(dz.flat[k]) ] += data.flat[k]
        gridWeights[ int(di.flat[k]), int( dj.flat[k]), int(dz.flat[k]) ] += 1

	# make gaussian kernel
    kernelWidth = 5
    kernelHeight = 5
    kernelDepth = 5

    (gridX, gridY, gridZ) = np.meshgrid( range(0, int(kernelWidth) ), range(0, int(kernelHeight) ), range(0, int(kernelDepth) ) )
    gridX -= math.floor( kernelWidth / 2 )
    gridY -=  math.floor( kernelHeight / 2 )
    gridZ -= math.floor( kernelDepth / 2 )
    gridRSquared = (gridX * gridX + gridY * gridY ) + ( gridZ * gridZ )
    kernel = np.exp( -0.5 * gridRSquared )

	# convolve
    
    blurredGridWeights = scipy.signal.fftconvolve( gridWeights, kernel, mode='same' )
    blurredGrid = scipy.signal.fftconvolve( gridData, kernel, mode='same' )
	# divide
    blurredGridWeights = np.where( blurredGridWeights == 0 , 0.1, blurredGridWeights) 
    normalizedBlurredGrid = blurredGrid / blurredGridWeights
    normalizedBlurredGrid = np.where( blurredGridWeights < -1, 0, normalizedBlurredGrid ) 

	# upsample
    ( jj, ii ) = np.meshgrid( range(0, inputWidth ), range(0, inputHeight ) )
	# no rounding
    di = ( ii / sigmaS ) + paddingXY +1
    dj = ( jj / sigmaS ) + paddingXY +1
    dz = ( edge - edgeMin ) / sigmaR + paddingZ +1

    return scipy.interpolate.interpn( (range(0,normalizedBlurredGrid.shape[0]),range(0, normalizedBlurredGrid.shape[1]),range(0, normalizedBlurredGrid.shape[2])), normalizedBlurredGrid, (di, dj, dz) )



def gammaCorrection(x):
    if x <= 0.0031308:
        return 12.92 * x
    return 1.055 * x ** (1 / 2.2) - 0.055


def durandanddorsy(img):
    height, width = np.shape(img)[:2]
    img = img/img.max()
    epsilon = 0.000001
    val = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2] + epsilon
    log_val = np.log(val)
    
    
    space_sigma = min(width,height)/16
    range_sigma = (np.amax(log_val)-np.amin(log_val))/10
    imgg = bilateral_approximation(log_val, log_val ,space_sigma, range_sigma)
    gamma = 0.45
    detailed_channel = np.exp(gamma*imgg + np.subtract(log_val,imgg))
    detailed_channel = detailed_channel.astype('float32')
    
    out = np.zeros(img.shape)
    out[:,:,0] = img[:,:,0] * (detailed_channel/val)
    out[:,:,1] = img[:,:,1] * (detailed_channel/val)
    out[:,:,2] = img[:,:,2] * (detailed_channel/val)
    
    gammafun = np.vectorize(lambda t: gammaCorrection(t))
    outt = np.clip(gammafun(out) * 255, 0, 255)
    outt = outt.astype('uint8')
    cv2.imwrite('tonemap.png', outt)
    # cv2.waitKey()
    
def linearScaling(img,scale):
    # img = img/img.max()
    epsilon = 0.0001
    val = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2] + epsilon

    out_val= val+scale
    out = np.zeros(img.shape)
    # np.clip(val,0,1)
    # np.clip(out_val,0,1)    
    out[:,:,0] = img[:,:,0] * (out_val/val)
    out[:,:,1] = img[:,:,1] * (out_val/val)
    out[:,:,2] = img[:,:,2] * (out_val/val)
    # temp = copy.deepcopy(out)
    # np.clip(out,0,1)
    gammafun = np.vectorize(lambda t: gammaCorrection(t))
    
    out = np.clip(gammafun(out) * 255, 0, 255).astype('uint8')
    
    
    cv2.imwrite('linearscale.png', out)
    # cv2.waitKey()

def logScaling(img,scale):
    epsilon = 0.000001
    val = np.log(0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2] + epsilon)
    out_val = val+scale
    out = np.zeros(img.shape)
    out[:,:,0] = img[:,:,0] * (np.exp(out_val)/np.exp(val))
    out[:,:,1] = img[:,:,1] * (np.exp(out_val)/np.exp(val))
    out[:,:,2] = img[:,:,2] * (np.exp(out_val)/np.exp(val))
    
    gammafun = np.vectorize(lambda t: gammaCorrection(t))
    
    out = np.clip(gammafun(out) * 255, 0, 255).astype('uint8')
    
    
    cv2.imwrite('logscale.png', out)
    # cv2.waitKey()


def mapfun(val, low, high):
    return (val - low)/ (high - low)

def equalizeHist(val):
    histogram = np.zeros(256)
    # print(val)
    # val = np.where(val>255, 255, val)
    # val = val*255
    val = val.astype('uint8')
    # print(val)
    for i in val:
        histogram[i] +=1    
    
    #cdf
    # print(histogram)
    cdf = np.zeros(256)
    cdf[0] = histogram[0]
    for i in range(1,256):
        cdf[i] = cdf[i-1]+histogram[i]
    
    inputHeight,inputWidth  = val.shape
    # print(np.min(cdf), inputHeight,inputWidth, cdf)
    cdffun = np.vectorize(lambda t : ((cdf[t]-np.min(cdf))/ inputHeight*inputWidth - np.min(cdf) ) )
    # cdffun = np.vectorize(lambda t : cdf[t])
    g= cdffun(val)

    out = mapfun(g,g.min(),g.max())
    return out



def histogram(img):
    epsilon = 0.00001
    # img = img/img.max()
    L =  0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2] + epsilon
    vall = np.log( L + 1)
    val = mapfun(vall,vall.min(),vall.max())

    out_val = equalizeHist(val*255)
    # out_val = val

    out = np.zeros(img.shape)

    out[:,:,0] = img[:,:,0] * ((out_val)/L)
    out[:,:,1] = img[:,:,1] * ((out_val)/L)
    out[:,:,2] = img[:,:,2] * ((out_val)/L)
    
    # out = out
    # out = val
    gammafun = np.vectorize(lambda t: gammaCorrection(t))
    
    out = np.clip(gammafun(out) * 255, 0, 255).astype('uint8')
    
    
    cv2.imwrite('histequal.png', out)
    # cv2.waitKey()

if __name__ == "__main__":
    img = cv2.imread('images/nave.hdr', -1)
    
    cv2.imwrite('normal.png', img)
    # cv2.waitKey()
    
    linearScaling(img,0.5)
    logScaling(img,0.01)

    histogram(img)

    durandanddorsy(img)
    