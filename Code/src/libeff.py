import cv2
import numpy as np 
import matplotlib.pyplot as plt
import os
from numpy.fft import fft2, ifft2, fftfreq
from libray import *


def on_key(event):
    if True: 
        plt.close()
def pad(img, padsize):
    padVert = int((padsize[1] - img.shape[0]) / 2)
    padHoz = int((padsize[0] - img.shape[1]) / 2)
    img = cv2.copyMakeBorder(img, padVert, padVert, padHoz, padHoz, cv2.BORDER_DEFAULT)
    return cv2.resize(img, padsize[::-1])

def align(imgs, scaleFactors):
    # Initialize ORB detector
    if len(imgs) == 2:
        orb = cv2.ORB_create()
        imgs[1] = cv2.resize(imgs[1], (int(imgs[0].shape[0] * scaleFactors[0]), int(imgs[0].shape[1] * scaleFactors[0]))) 

        imgs[1] = pad(imgs[1], imgs[0].shape[0:2])

        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(cv2.equalizeHist(imgs[0].astype(np.uint8)), None)
        kp2, des2 = orb.detectAndCompute(cv2.equalizeHist(imgs[1].astype(np.uint8)), None)

        method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
        matcher = cv2.DescriptorMatcher_create(method)
        matches = matcher.match(des1, des2, None)
        
        matches = sorted(matches, key=lambda x:x.distance)
        keep = int(len(matches) * 0.9)
        matches = matches[:keep]

        ptsA = np.zeros((len(matches), 2), dtype="float")
        ptsB = np.zeros((len(matches), 2), dtype="float")
        
        # loop over the top matches
        for (i, m) in enumerate(matches):
            # indicate that the two keypoints in the respective images
            # map to each other
            ptsA[i] = kp1[m.queryIdx].pt
            ptsB[i] = kp2[m.trainIdx].pt
            
        # Estimate homography
        H, _ = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 5.0)

        # Warp image
        res1 = cv2.warpPerspective(imgs[1], H, (imgs[1].shape[1], imgs[1].shape[0]))

        return [imgs[0], res1]

    orb = cv2.ORB_create()
    imgs[1] = cv2.resize(imgs[1], (int(imgs[0].shape[0] * scaleFactors[0]), int(imgs[0].shape[1] * scaleFactors[0]))) 
    imgs[2] = cv2.resize(imgs[2], (int(imgs[0].shape[0] * scaleFactors[1]), int(imgs[0].shape[1] * scaleFactors[1]))) 

    imgs[1] = pad(imgs[1], imgs[0].shape[0:2])
    imgs[2] = pad(imgs[2], imgs[0].shape[0:2])

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(cv2.equalizeHist(imgs[0].astype(np.uint8)), None)
    kp2, des2 = orb.detectAndCompute(cv2.equalizeHist(imgs[1].astype(np.uint8)), None)
    kp3, des3 = orb.detectAndCompute(cv2.equalizeHist(imgs[2].astype(np.uint8)), None)

    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(des1, des2, None)

    matcher2 = cv2.DescriptorMatcher_create(method)
    matches2 = matcher2.match(des1, des3, None)
    
    matches = sorted(matches, key=lambda x:x.distance)
    matches2 = sorted(matches2, key=lambda x:x.distance)

    keep = int(len(matches) * 0.9)
    matches = matches[:keep]

    keep2 = int(len(matches2) * 0.9)
    matches2 = matches2[:keep2]

    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    
    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kp1[m.queryIdx].pt
        ptsB[i] = kp2[m.trainIdx].pt

    ptsB2 = np.zeros((len(matches2), 2), dtype="float")
    ptsA2 = np.zeros((len(matches2), 2), dtype="float")

    for (i, m) in enumerate(matches2):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA2[i] = kp1[m.queryIdx].pt
        ptsB2[i] = kp3[m.trainIdx].pt
        

    # Estimate homography
    H, _ = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 5.0)

    # Estimate homography
    H2, _ = cv2.findHomography(ptsB2, ptsA2, cv2.RANSAC, 5.0)

    # Warp image
    res1 = cv2.warpPerspective(imgs[1], H, (imgs[1].shape[1], imgs[1].shape[0]))
    res2 = cv2.warpPerspective(imgs[2], H2, (imgs[1].shape[1], imgs[1].shape[0]))

    return [imgs[0], res1, res2]


def dualDistanceEffs(imgPath, darkPath, flatPath, distance):
    dark = []
    flat = []
    raw =[]
    for i in range(0, len(darkPath)):
        dark.append(cv2.imread(darkPath[i][6:]))
        flat.append(cv2.imread(flatPath[i][6:]))
        raw.append(cv2.imread(imgPath[i][6:],cv2.IMREAD_GRAYSCALE))

    scaleFactors = [(137+distance[0])/(137+distance[1])]
    
    
    raw = align(raw, scaleFactors)
    #Converting all images to grayscale f32 
    for i in range(0,len(dark)):
        dark[i] = cv2.cvtColor(dark[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        flat[i] = cv2.cvtColor(flat[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        raw[i] = raw[i].astype(np.float32)

    #preprocess 
    processed_imgs = []
    for i in range(0, len(dark)):
        reg = (raw[i]-dark[i]) / (flat[i]-dark[i]) 
        processed_imgs.append(reg) #divide be mean in empty space

    #regimage pixels
    for i in range(0,len(dark)):
        width, height = processed_imgs[i].shape
        processed_imgs[i] = processed_imgs[i][50:width-50, 50:height-50]
        meanFree = cv2.mean(processed_imgs[i][50:width-150, 50:100])
        processed_imgs[i] = processed_imgs[i] / meanFree[0]

    res = dualdistance_FP(processed_imgs, distance, 0.5e-7, 2, 12.3e-6)

    return res["thickness"]


    
#try to line images up 
#try smaller distance only (0.3, 1, 3)?
def xray_effects(imgPath, darkPath, flatPath, distance, tik_regs, padType=0, alignment=None, cutoff=0):

    dark = []
    flat = []
    raw =[]
    for i in range(0, len(darkPath)):
        dark.append(cv2.imread(darkPath[i][6:]))
        flat.append(cv2.imread(flatPath[i][6:]))
        raw.append(cv2.imread(imgPath[i][6:],cv2.IMREAD_GRAYSCALE))
        
    scaleFactors = [(137+distance[0])/(137+distance[1]), (137+distance[0])/(137+distance[2])]
    
    if alignment is not None:
        raw = align(raw, scaleFactors)
    #Converting all images to grayscale f32 
    for i in range(0,len(dark)):
        dark[i] = cv2.cvtColor(dark[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        flat[i] = cv2.cvtColor(flat[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        raw[i] = raw[i].astype(np.float32)
        
    #preprocess 
    processed_imgs = []
    for i in range(0, len(dark)):
        reg = (raw[i]-dark[i]) / (flat[i]-dark[i]) 
        processed_imgs.append(reg) #divide be mean in empty space

    #regimage pixels
    for i in range(0,len(dark)):
        width, height = processed_imgs[i].shape
        processed_imgs[i] = processed_imgs[i][50:width-50, 50:height-50]
        meanFree = cv2.mean(processed_imgs[i][50:width-150, 50:100])
        processed_imgs[i] = processed_imgs[i] / meanFree[0]

    #try
    
    match padType:
        case 0:
            results = multidistance_FP(processed_imgs, distance, 2.016e10, 12.3e-6, tik_regs, 0, cutoff)
        case 1:
            results = multidistance_FP(processed_imgs, distance, 2.016e10, 12.3e-6, tik_regs, 1, cutoff)
        case 2:
            results = multidistance_FP(processed_imgs, distance, 2.016e10, 12.3e-6, tik_regs, 2, cutoff)
        case 3:
            results = multidistance_FP(processed_imgs, distance, 2.016e10, 12.3e-6, tik_regs, 3, cutoff)

    I0 = results['I0'] #white line should disappear
    phase = results['phase']
    diff_coeff = results['diff_coeff']

    return [raw[0], processed_imgs[0], I0, phase, diff_coeff]
            
def dualPhaseDarkField(imgPath, darkPath, flatPath, 
                       distance, stride, cutsize = 0, 
                       highPass=True, edgeTaper = True, highPassVal = 0):
    dark = []
    flat = []
    raw =[]
    for i in range(0, len(darkPath)):
        dark.append(cv2.imread(darkPath[i][6:]))
        flat.append(cv2.imread(flatPath[i][6:]))
        raw.append(cv2.imread(imgPath[i][6:],cv2.IMREAD_GRAYSCALE))

    scaleFactors = [(137+distance[0])/(137+distance[1])]
    
    
    raw = align(raw, scaleFactors)
    #Converting all images to grayscale f32 
    for i in range(0,len(dark)):
        dark[i] = cv2.cvtColor(dark[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        flat[i] = cv2.cvtColor(flat[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        raw[i] = raw[i].astype(np.float32)

    #preprocess 
    processed_imgs = []
    for i in range(0, len(dark)):
        reg = (raw[i]-dark[i]) / (flat[i]-dark[i]) 
        processed_imgs.append(reg) #divide be mean in empty space

    #regimage pixels
    for i in range(0,len(dark)):
        width, height = processed_imgs[i].shape
        processed_imgs[i] = processed_imgs[i][50:width-50, 50:height-50]
        meanFree = cv2.mean(processed_imgs[i][50:width-150, 50:100])
        processed_imgs[i] = processed_imgs[i] / meanFree[0]

    dist = distance[1] - distance[0]
    wavelength = 1/2.016e10

    darkField = getDarkField(processed_imgs, stride)

    _, phase = getPhaseEffects(processed_imgs, dist,wavelength, 12.3e-6, 
                               edge_taper=edgeTaper, 
                               highpass=highPass, 
                               highpassVal=highPassVal, 
                               cutsize=cutsize)

    return processed_imgs[0], darkField, phase

def show_video():
    start = True
    folder_path = "./res/Images/Saved/"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        img = cv2.resize(cv2.imread(file_path), (1200,1200))
        cv2.imshow('fig', img)
        if start:
            start = False
            cv2.waitKey(0)
        else:
            key = cv2.waitKey(1000)  # Wait 100ms (~10 FPS); adjust as needed
            if key == 27:  # Press Esc to exit early
                break

    cv2.destroyAllWindows()    


'''

#try
results = multidistance_FP(processed_imgs, distances, (2e10), 12.3e-6, tik_regs)
I0 = results['I0']
phase = results['phase']
diff_coeff = results['diff_coeff']

intensity_eff = processed_imgs[1] - I0
phase_eff = processed_imgs[1] - phase
diff_eff = processed_imgs[1] - diff_coeff

'''


'''
for i in range(0,1):
    raw, preprocess, I0, phase, dark = xray_effects(i, [1.0,2.0,3.0], [1e-3, 1e-4, 1e-3]) #try modulation of tik factor of 10

    fig = plt.figure(figsize=(15, 15)) 

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.subplot(4, 2, 1) # 1 row, 2 columns, 1st position
    plt.imshow(raw, cmap='gray')
    plt.title('Raw')
    plt.axis('off') # Optional: turn off axes

    plt.subplot(4, 2, 2) # 1 row, 2 columns, 2nd position
    plt.imshow(preprocess, cmap='gray')
    plt.title('Preprocesed')
    plt.axis('off')

    plt.subplot(4, 2, 3) # 1 row, 2 columns, 2nd position
    plt.imshow(I0, cmap='gray')
    plt.title('Intensity')
    plt.axis('off')

    plt.subplot(4, 2, 7) # 1 row, 2 columns, 2nd position
    plt.imshow(phase, cmap='gray')
    plt.title('phase')
    plt.axis('off')

    plt.subplot(4, 2, 5) # 1 row, 2 columns, 2nd position
    plt.imshow(dark, cmap='gray')
    plt.title('Diff Coefficiant')
    plt.axis('off')

    plt.savefig(f'./res/Images/Saved/xray_figure_{i}.png', dpi=300, bbox_inches='tight')

    #plt.tight_layout()
    plt.show()

'''



#show_video()



'''
plt.figure(figsize=(20, 20)) 

plt.subplot(4, 2, 1) # 1 row, 2 columns, 1st position
plt.imshow(raw[1], cmap='gray')
plt.title('Raw')
plt.axis('off') # Optional: turn off axes

plt.subplot(4, 2, 2) # 1 row, 2 columns, 2nd position
plt.imshow(processed_imgs[1], cmap='gray')
plt.title('Preprocesed')
plt.axis('off')

plt.subplot(4, 2, 3) # 1 row, 2 columns, 2nd position
plt.imshow(I0, cmap='gray')
plt.title('Intensity')
plt.axis('off')

plt.subplot(4, 2, 7) # 1 row, 2 columns, 2nd position
plt.imshow(phase, cmap='gray')
plt.title('phase')
plt.axis('off')

plt.subplot(4, 2, 5) # 1 row, 2 columns, 2nd position
plt.imshow(diff_coeff, cmap='gray')
plt.title('Diff Coefficiant')
plt.axis('off')

plt.subplot(4, 2, 4) # 1 row, 2 columns, 2nd position
plt.imshow(intensty_eff, cmap='gray')
plt.title('intensity effects')
plt.axis('off')

plt.subplot(4, 2, 8) # 1 row, 2 columns, 2nd position
plt.imshow(phase_eff, cmap='gray')
plt.title('phase effects')
plt.axis('off')

plt.subplot(4, 2, 6) # 1 row, 2 columns, 2nd position
plt.imshow(diff_eff, cmap='gray')
plt.title('Diff effects')
plt.axis('off')


#plt.tight_layout()
plt.show()

'''
"""
cv2.imshow("yurt", img1)
cv2.imshow("gurt", imageadapted)
cv2.imshow("gray", gray_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""




