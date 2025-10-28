import cv2
import numpy as np 
import scipy
import matplotlib.pyplot as plt
import os
from numpy.fft import fft2, ifft2, fftfreq
from libray import *
from libeff import *
from scipy.ndimage import gaussian_filter

def local_stddev(arr):
    return np.std(arr)

import numpy as np
from scipy.optimize import nnls
from scipy.ndimage import median_filter

def unmix_xray_materials(phase, absorption, dark, wavelength, delta, mu, sigma,
                         smooth_size=3, normalize=True):
    """
    Perform physics-based material unmixing from X-ray phase, absorption, and dark-field maps.

    Parameters
    ----------
    phase : 2D array
        Retrieved phase map [radians]
    absorption : 2D array
        Log-attenuation map, e.g. -ln(I/I0)
    dark : 2D array
        Dark-field (scattering strength) map
    wavelength : float
        X-ray wavelength [m]
    delta : list or array of length N
        Refractive index decrements (δ) of each material
    mu : list or array of length N
        Linear attenuation coefficients [1/m]
    sigma : list or array of length N
        Scattering coefficients (dark-field) per unit thickness
    smooth_size : int
        Optional median filter kernel to suppress pixel noise (default = 3)
    normalize : bool
        Whether to normalize rows of S to comparable scales for numerical stability

    Returns
    -------
    thickness_maps : (H, W, N) array
        Estimated thickness map for each material
    residual : (H, W) array
        Fit residual per pixel
    S : (3, N) array
        Signature matrix actually used
    """
    H, W = phase.shape
    N = len(delta)
    # Build physics matrix S
    S = np.vstack([
        -2 * np.pi / wavelength * np.array(delta),
        np.array(mu),
        np.array(sigma)
    ])

    if normalize:
        row_norms = np.linalg.norm(S, axis=1, keepdims=True) + 1e-12
        S = S / row_norms

    # Flatten images
    m = np.stack([phase, absorption, dark], axis=-1).reshape(-1, 3)

    # Prepare output arrays
    T = np.zeros((m.shape[0], N))
    residual = np.zeros(m.shape[0])

    # Solve NNLS per pixel
    for i in range(m.shape[0]):
        T[i], rnorm = nnls(S.T, m[i])
        residual[i] = rnorm

    # Reshape to image
    T = T.reshape(H, W, N)
    residual = residual.reshape(H, W)

    # Optional median filtering to clean noise
    if smooth_size > 1:
        for k in range(N):
            T[:, :, k] = median_filter(T[:, :, k], size=smooth_size)

    return T, residual, S



def retrieve_phase_TIE_stable(imgs, dz, wavelength, pixel_size, 
                              reg=1e-8, edge_taper=True, highpass=True, min_intensity=1e-6):
    """
    Retrieve X-ray phase map using a stabilized TIE approach with artifact suppression.
    Prevents divide-by-zero warnings and large-circle artifacts.
    """

    # Load & normalize
    I0 = imgs[0][:, 1:-1]
    I1 = imgs[1][:, 1:-1]
    eps = 1e-12
    #I0 = np.maximum(I0, min_intensity)
    #I1 = np.maximum(I1, min_intensity)
    #I0 /= np.mean(I0)
    #I1 /= np.mean(I1)

    # Derivative & mean
    dI_dz = (I1 - I0) / dz
    I_mean = np.clip(0.5 * (I0 + I1), min_intensity, None)

    # Optional edge taper (reduce border discontinuities)
    if edge_taper:
        window_x = np.hanning(I_mean.shape[1])
        window_y = np.hanning(I_mean.shape[0])
        window = np.outer(window_y, window_x)
        I_mean *= window
        dI_dz *= window

    # Transport-of-Intensity RHS (safe division)
    RHS = - (2 * np.pi / wavelength) * (dI_dz / I_mean)
    RHS = np.nan_to_num(RHS, nan=0.0, posinf=0.0, neginf=0.0)

    # Fourier-domain Poisson solver with regularization
    ny, nx = RHS.shape
    fx = np.fft.fftfreq(nx, d=pixel_size)
    fy = np.fft.fftfreq(ny, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)
    f2 = FX**2 + FY**2

    RHS_hat = np.fft.fft2(RHS)
    denom = - (2 * np.pi)**2 * f2
    denom[np.abs(denom) < reg] = np.inf  # prevent divide-by-zero
    phi_hat = RHS_hat / denom
    phi_hat[0, 0] = 0.0  # remove DC term
    phi = np.fft.ifft2(phi_hat).real

    # Optional high-pass (remove slow background)
    if highpass:
        phi -= gaussian_filter(phi, sigma=3)

    # Normalize for display
    phi_norm = (phi - np.percentile(phi, 1)) / (np.percentile(phi, 99) - np.percentile(phi, 1) + eps)
    phi_norm = np.clip(phi_norm, 0, 1)

    return phi, phi_norm


dark = [cv2.imread("./res/Images/DarkField/0.3RoadTube/DF_BEFORE_00.tif"),
        cv2.imread("./res/Images/DarkField/1.0RoadTube/DF_BEFORE_00.tif"),
        cv2.imread("./res/Images/DarkField/3.0RoadTube/DF_BEFORE_00.tif")]

flat = [cv2.imread("./res/Images/Background/0.3RoadTube/BG_BEFORE_00.tif"),
        cv2.imread("./res/Images/Background/1.0RoadTube/BG_BEFORE_00.tif"),
        cv2.imread("./res/Images/Background/3.0RoadTube/BG_BEFORE_00.tif")]

raw = [cv2.imread("./res/Images/Raw/0.3RoadTube/SAMPLE_T0000.tif", cv2.IMREAD_GRAYSCALE),
       cv2.imread("./res/Images/Raw/1.0RoadTube/SAMPLE_T0000.tif", cv2.IMREAD_GRAYSCALE),
       cv2.imread("./res/Images/Raw/3.0RoadTube/SAMPLE_T0000.tif", cv2.IMREAD_GRAYSCALE)]
    
scaleFactors = [(137+1)/(137+2), (137+1)/(137+3)]

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
   
wavelength  = 1/2.016e10
#phase, phaser = retrieve_phase_TIE_stable((processed_imgs[0], processed_imgs[2]), 2.7, wavelength, 12.3e-6)
#subSimple, _, _,_ = stdev_Comparison((processed_imgs[0],processed_imgs[2]), 17)
#results = multidistance_FP(processed_imgs, (0.3,1,3),2.016e10, 12.3e-6, [0.00004, 0.00004, 0.00004])

res = dualdistance_FP([processed_imgs[0], processed_imgs[2]], (0.3,3), 1e-7, 2, 12.3e-6)

vref = res['ref']

_, div4, _,_ = stdev_Comparison((vref,processed_imgs[2]), 4)
_, div16, _,_ = stdev_Comparison((vref,processed_imgs[2]), 16)
_, div24, _,_ = stdev_Comparison((vref,processed_imgs[2]), 24)
_, div48, _,_ = stdev_Comparison((vref,processed_imgs[2]), 48)

cv2.imwrite('4.tif', div4)
cv2.imwrite('16.tif', div16)

cv2.imwrite('24.tif', div24)

cv2.imwrite('48.tif', div48)



fig, axs = plt.subplots(3, 2, figsize=(10, 10))

# Plot each array
axs[0, 0].imshow(div4, cmap='binary')
axs[0, 0].set_title('subtracted')


axs[0, 1].imshow(div16, cmap='binary')
axs[0, 1].set_title('divided')



axs[1, 0].imshow(vref, cmap='magma')
axs[1, 0].set_title('vref')

axs[1, 1].imshow(processed_imgs[0], cmap='binary')
axs[1, 1].set_title('raw1')

axs[2, 0].imshow(processed_imgs[2], cmap='binary')
axs[2, 0].set_title('raw2')

for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

"""

#results = multidistance_FP2(processed_imgs, (1,2,3), 2.016e10, 12.3e-6, [0.0001, 0.0001, 0.0001])
subSimple, _, _,_ = stdev_Comparison((processed_imgs[0],processed_imgs[2]), 17)
subSimple2, _, _,_ = stdev_Comparison((processed_imgs[0],processed_imgs[2]), 50)
subSimple3, _, _,_ = stdev_Comparison((processed_imgs[0],processed_imgs[2]), 5)

subSimple1, _, _,_ = stdev_Comparison((processed_imgs[0],processed_imgs[1]), 17)
subSimple12, _, _,_ = stdev_Comparison((processed_imgs[0],processed_imgs[1]), 50)
subSimple13, _, _,_ = stdev_Comparison((processed_imgs[0],processed_imgs[1]), 5)

#I0 = results['I0'] #white line should disappear
#phase = results['phase']
#diff_coeff = results['diff_coeff']
#sub = results['DarkSubtracted']
#div = results['DarkDivided']
#std1 = results['STD1']
#std2 = results['STD2']

subSimpleComp = subSimple - subSimple1
subSimpleComp2 = subSimple2 - subSimple12
subSimpleComp3 = subSimple3 - subSimple13

svd3comp = subSimple1 - generic_filter(processed_imgs[2], local_stddev, size=17, mode='reflect')
svd3comp2 = subSimple12 - generic_filter(processed_imgs[2], local_stddev, size=50, mode='reflect')
svd3comp3 = subSimple13 - generic_filter(processed_imgs[2], local_stddev, size=5, mode='reflect')

fig, axs = plt.subplots(4, 2, figsize=(10, 10))

# Plot each array
axs[0, 0].imshow(subSimple, cmap='binary')
axs[0, 0].set_title('stride 17')


axs[0, 1].imshow(subSimpleComp, cmap='binary')
axs[0, 1].set_title('Stride 17 Comp')


axs[1, 0].imshow(subSimple3, cmap='binary')
axs[1, 0].set_title('stride 5')

axs[1, 1].imshow(subSimpleComp3, cmap='binary')
axs[1, 1].set_title('stride 5 Comp')

axs[2, 0].imshow(subSimple2, cmap='binary')
axs[2, 0].set_title('stride 50')


axs[2, 1].imshow(subSimpleComp2, cmap='binary')
axs[2, 1].set_title('stride 50 Comp')

axs[3, 0].imshow(svd3comp, cmap='binary')
axs[3, 0].set_title('Div Comp stride 17')


axs[3, 1].imshow(svd3comp3, cmap='binary')
axs[3, 1].set_title('Div Comp stride 5')


# Optional: tidy up
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
"""
""""
Conclusion:
can just use images no need for fancy maths in respect to dark field (Using image at 0,2 distane instead of I0 term from equations)
visibility is better than a laplacian (Naive subtraction gives same result as lengthly processing) naive should be wose as it does not account for phase effects. Naive works qualtitivly but might not be appropriate for final use as it does not include all phase effects
stride/kernal size has a large impact on the final resultant image.

Qestuons 
Which approch work best
does the I0 fix also help the phase (change I0 to im0 for all I0 places) answer: NO
do different image distances give better/worse results?

"""
