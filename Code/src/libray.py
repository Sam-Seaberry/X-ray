"""Diffusion retrieval in propagation-based imaging."""

import numpy as np
from scipy.ndimage import generic_filter, gaussian_filter
import cv2


def local_stddev(arr):
    return np.std(arr)

def stdev_Comparison(arrs, stride):
    """
    Function to compair local intensity values between two arrays. 
    Sliding window with stride=stride used to compute local STDEV,
    values. Values are then subtracted from eachother:
    STDEV(reference) - STDEV(observed)

    Parameters:
    Arrs: two np.arrays of shape x,2
    Stride: stride int value

    Returns:
    Arr: The resulting array from the sliding window opperation
    """

    if len(arrs) > 2:
        return

    
    std1 = generic_filter(arrs[0], local_stddev, size=stride, mode='reflect')
    std2 = generic_filter(arrs[1], local_stddev, size=stride, mode='reflect')

    # Subtract local stddevs
    std_diff = std1 - std2
    std_div = std1 / std2

    return std_diff, std_div, std1, std2

def _fourier_freqs(arr_shape: tuple, pixel_size=12.3e-6):
    """Generate fourier frequencies for a given array

    We multiply the output of np.fft.fftfreq by 2*pi since the DFT has
    a factor of 2*pi in the exponent.

    Parameters
    ----------
    arr_shape : tuple
        Shape of the array
    pixel_size : float or tuple, optional
        Pixel size in the array. If a tuple, it must have the same length as the array
        shape, giving the pixel size in each dimension. By default 1.

    Returns
    -------
    list
        List of k-space frequencies in each dimension
    """

    if isinstance(pixel_size, (int, float)):
        pixel_size = np.full(len(arr_shape), pixel_size)
    elif len(pixel_size) != len(arr_shape):
        raise ValueError(
            "Pixel size must be a scalar or a tuple of the same length as the array "
            "shape."
        )

    ks = []
    for i in range(len(arr_shape)):
        k = 2.0 * np.pi * np.fft.fftfreq(arr_shape[i], d=pixel_size[i])
        ks.append(k)
    return ks


def _kperp2(arr_shape: tuple, pixel_size=12.3e-6):
    """Squared Fourier frequencies

    We multiply the output of np.fft.fftfreq by 2*pi since the DFT has
    a factor of 2*pi in the exponent.

    Parameters
    ----------
    arr_shape : tuple
        Shape of the array
    pixel_size : float or tuple, optional
        Pixel size in the array. If a tuple, it must have the same length as the array
        shape, giving the pixel size in each dimension. By default 1.

    Returns
    -------
    ndarray
        Squared k-space frequencies
    """
    kps = 0.0 
    ks = _fourier_freqs(arr_shape, pixel_size)
    for k in ks:
        kps = np.add.outer(kps, k**2)

    
    return kps


def dualdistance_FP(arrs, distances, delta, mu, pixel_size):
    """Phase-retrieval using the dual-distance Fokker-Planck algorithm.

    Implements the algorithm developed in [1], specifically equation (11). Note that
    this implementation does not include the final conversion to projected thickness.

    References
    ----------
    [1] Leatham, T. A. et al. IEEE Transactions on Medical Imaging 42, 1681–1695 (2023).

    Parameters
    ----------
    arrs : list[np.ndarray]
        The two images.
    distances : list[float]
        The propagation distances of the two images (m).
    delta : float
        The refractive index decrement (m^-1). try 1-10e^-7 possibly larger
    mu : float
        The attenuation coefficient (m^-1). try 0-4 = (2*k*b)
    pixel_size : float
        The physical pixel size (m).

    Returns
    -------
    dict
        A dictionary containing the reconstructed projected thickness and the
        parameters used for the reconstruction.
    """
    # Validate inputs
    if len(arrs) != 2 or len(distances) != 2:
        raise ValueError("arrs and distances must both have length 2.")
    if not isinstance(arrs, list):
        raise TypeError("arrs must be a list of two numpy arrays.")
    if not all(isinstance(arr, np.ndarray) for arr in arrs):
        raise TypeError("All elements in arrs must be numpy arrays.")

    # Ensure that the inputs are ordered correctly
    # Want lower distance first
    if distances[0] > distances[1]:
        arrs = [arrs[1], arrs[0]]
        distances = [distances[1], distances[0]]

    kperp2 = _kperp2(arrs[0].shape, pixel_size)
    d1, d2 = distances
    im1, im2 = arrs

    gamma = delta / mu
    num = np.fft.fft2((d2**2 * im1) - (d1**2 * im2))
    denom = d2**2 - d1**2 + gamma * d1 * d2 * (d2 - d1) * kperp2

    im_diffusionretrieved = np.fft.ifft2(num / denom).real

    


    T = (-1.0 / mu) * np.log(im_diffusionretrieved)

    expo = np.exp(-mu*T)

    grad_g_norm = np.gradient(expo, pixel_size, axis=(0, 1)) / im1  # eq. (4.23) try to change to just the distance at 0.3m
    dx = np.gradient(grad_g_norm[0], pixel_size, axis=0)
    dy = np.gradient(grad_g_norm[1], pixel_size, axis=1)

    ref = expo - ((delta * distances[1])/mu) * (dx + dy)

    cv2.imshow('img', ref)
    cv2.imwrite('maybe.tif', ref)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    parameters = {
        "distance": distances,
        "delta": delta,
        "mu": mu,
        "pixel_size": pixel_size,
    }

    # Store the results in a dictionary
    results = {
        "thickness": T,
        "parameters": parameters,
        'ref': ref
    }

    return results

def inverse_laplacian(arr, pixel_size, *, tik_reg=0.0001, padding=0, cutoff = 0):
    """Inverse Laplacian filter.

    Parameters
    ----------
    arr : ndarray
        Input array (1D or 2D)
    pixel_size : float
        Pixel size of the input array
    tik_reg : float, optional
        Tikhonov regularisation parameter, by default 0.0001


    Returns
    -------
    ndarray
        Inverse Laplacian filtered array
    """
    dim = len(arr.shape)

    if dim > 2:
        raise ValueError("Only 1D and 2D arrays are supported.")

    # Enforce periodicity
    



    pad_y = arr.shape[0] // 2  # vertical padding (top & bottom)
    pad_x = arr.shape[1] // 2  # horizontal padding (left & right)


    #Padding 
    match padding:
        case 0:
            #Defult Padding
            if dim == 1:
                flip = np.concatenate((arr, np.flipud(arr)))
            else:
                flip = np.concatenate((arr, np.flipud(arr)), axis=0)
                flip = np.concatenate((flip, np.fliplr(flip)), axis=1)
        case 1:
            #Reflective Padding
            flip = np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')
        case 2:
            #Symtric Padding
            flip = np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x)), mode='symmetric')
        case 3:
            #Wrapped Padding
            flip = np.pad(arr, ((arr.shape[0], arr.shape[0]), (arr.shape[1], arr.shape[1])), mode='wrap')

    
    # Prepare the k-space filte
    kperp2 = _kperp2(flip.shape, pixel_size)
    #kperp = np.sqrt(kperp2)
    kspace_filter = 1.0 / (kperp2 + tik_reg)

    #high_pass_filter = 1.0 - np.exp(-(kperp**2) / (2 * cutoff**2))

    # Combined filter
    #combined_filter = kspace_filter * high_pass_filter
    
    # Apply the filter
    arr_fft = np.fft.fft2(flip)
    
    arr_filt = np.real(-1.0 * np.fft.ifft2(kspace_filter * arr_fft))#All values in this array are vary large and quite simular

    #Create high pass image

    

    match padding:
        case 0:
            #Defult Padding
            if dim == 1:
                arr_filt = arr_filt[0 : arr.shape[0]]
            else:
                arr_filt = arr_filt[0 : arr.shape[0], 0 : arr.shape[1]]
        case 1:
            #Reflective Padding
            arr_filt = arr_filt[pad_y : arr.shape[0]+pad_y, pad_x : arr.shape[1]+pad_x]
        case 2:
            #Symtric Padding
            arr_filt = arr_filt[pad_y : arr.shape[0]+pad_y, pad_x : arr.shape[1]+pad_x]
        case 3:
            #Wrapped Padding
            arr_filt = arr_filt[arr.shape[0] : arr.shape[0]+arr.shape[0], arr.shape[1] : arr.shape[1]+arr.shape[1]]

    
    
    return arr_filt

def multidistance_FP(arrs, distances, wavenumber, pixel_size, tik_regs, padding=0, cutoff=0):
    """Phase-retrieval using the triple-distance Fokker-Planck algorithm.

    Implements the algorithm developed in chapter 4 of [1], using the second
    version of the algorithm that does not require a contact image.

    References
    ----------
    [1] Leatham, T. A. (Monash University, 2024). doi:10.26180/25649073.v1.


    Parameters
    ----------
    arrs : list[np.ndarray]
        The three images.
    distances : list[float]
        The propagation distances of the three images (m).
    wavenumber : float
        The wavenumber of the X-rays (m^-1) (2 * pi / wavelength).
    pixel_size : float
        The physical pixel size (m).
    tik_regs : list[float]
        The three Tikhonov regularisation parameters. These are used to:
        (1) Regularise the inversion of sigma to give g(x,y) [eq. (4.22)]
        (2) Regularise the inversion to give the phase signal [eq. (4.23)]
        (3) Regularise the inversion to give the dark-field signal [eq. (4.24)]

    Returns
    -------
    dict, cutoff=int(self.ui.cutoff.text())
        A dictionary containing the results and the parameters used for the
        reconstruction, with the following structure:
        - 'I0' (np.ndarray): The contact intensity.
        - 'phase' (np.ndarray): The phase signal.
        - 'diff_coeff' (np.ndarray): The diffusion coefficient.
        - 'parameters' (dict): A dictionary containing the parameters used for the
          reconstruction, including 1.0 distances, wavenumber, pixel size, and Tikhonov
          regularisation parameters.
    """
    # Convenience variables
    d1, d2, d3 = distances
    im1, im2, im3 = arrs
    k = wavenumber

    # Start by finding the contact intensity [eq. (4.17)]
    q = (
        d1**2 * (d3 * im2 - d2 * im3)
        + d1 * (d2**2 * im3 - d3**2 * im2)
        + (d2 * d3**2 - d3 * d2**2) * im1
    )
    I0_denom = d1**2* (d3 - d2) + d1 * (d2**2 - d3**2) + d2 * d3**2 - d3 * d2**2
    I0 = q / I0_denom

    # Then calculate the phase signal
    sigma_num = k*(d1**2 * im2 - d2**2 * im1 + (d2**2 - d1**2) * im1) #try to change to 0.3m distance 
    sigma_denom = d1 * d2 * (d2 - d1)
    sigma = sigma_num / sigma_denom

    g = inverse_laplacian(sigma, pixel_size, tik_reg=tik_regs[0], padding=padding, cutoff=cutoff)
    grad_g_norm = np.gradient(g, pixel_size, axis=(0, 1)) / im1  # eq. (4.23) try to change to just the distance at 0.3m
    dx = np.gradient(grad_g_norm[0], pixel_size, axis=0)
    dy = np.gradient(grad_g_norm[1], pixel_size, axis=1)
    lap_phase = dx + dy
    phase = inverse_laplacian(lap_phase, pixel_size, tik_reg=tik_regs[1], padding=padding, cutoff=cutoff)

    # Finally, calculate the dark-field signal
    diff_coeff_lap = ((im3 - im1) / d3**2) + (sigma / (k * d1)) #try to change to 0.3m distance
    #sub, div, std1, std2 = stdev_Comparison(((sigma / (k * d1)), ((im3 - im1) / d3**2)), 17)
    diff_coeff = (1.0 / im1) * inverse_laplacian(
        diff_coeff_lap, pixel_size, tik_reg=tik_regs[2], padding=padding, cutoff=cutoff
    )

    parameters = {
        "distance": distances,
        "wavenumber": wavenumber,
        "pixel_size": pixel_size,
        "tik_reg": tik_regs,
    }

    # Store the results in a dictionary
    results = {
        "I0": I0,
        "phase": phase,
        "diff_coeff": diff_coeff,
        "parameters": parameters,
    }

    return results

def multidistance_FP2(arrs, distances, wavenumber, pixel_size, tik_regs, padding=0, cutoff=0):
    """Phase-retrieval using the triple-distance Fokker-Planck algorithm.

    Implements the algorithm developed in chapter 4 of [1], using the second
    version of the algorithm that does not require a contact image.

    References
    ----------
    [1] Leatham, T. A. (Monash University, 2024). doi:10.26180/25649073.v1.


    Parameters
    ----------
    arrs : list[np.ndarray]
        The three images.
    distances : list[float]
        The propagation distances of the three images (m).
    wavenumber : float
        The wavenumber of the X-rays (m^-1) (2 * pi / wavelength).
    pixel_size : float
        The physical pixel size (m).
    tik_regs : list[float]
        The three Tikhonov regularisation parameters. These are used to:
        (1) Regularise the inversion of sigma to give g(x,y) [eq. (4.22)]
        (2) Regularise the inversion to give the phase signal [eq. (4.23)]
        (3) Regularise the inversion to give the dark-field signal [eq. (4.24)]

    Returns
    -------
    dict, cutoff=int(self.ui.cutoff.text())
        A dictionary containing the results and the parameters used for the
        reconstruction, with the following structure:
        - 'I0' (np.ndarray): The contact intensity.
        - 'phase' (np.ndarray): The phase signal.
        - 'diff_coeff' (np.ndarray): The diffusion coefficient.
        - 'parameters' (dict): A dictionary containing the parameters used for the
          reconstruction, including 1.0 distances, wavenumber, pixel size, and Tikhonov
          regularisation parameters.
    """
    # Convenience variables
    d1, d2, d3 = distances
    im1, im2, im3 = arrs
    k = wavenumber

    # Start by finding the contact intensity [eq. (4.17)]
    q = (
        d1**2 * (d3 * im2 - d2 * im3)
        + d1 * (d2**2 * im3 - d3**2 * im2)
        + (d2 * d3**2 - d3 * d2**2) * im1
    )
    I0_denom = d1**2* (d3 - d2) + d1 * (d2**2 - d3**2) + d2 * d3**2 - d3 * d2**2
    I0 = q / I0_denom

    # Then calculate the phase signal
    sigma_num = k*(d1**2 * im2 - d2**2 * im1 + (d2**2 - d1**2) * I0) #try to change to 0.3m distance 
    sigma_denom = d1 * d2 * (d2 - d1)
    sigma = sigma_num / sigma_denom

    g = inverse_laplacian(sigma, pixel_size, tik_reg=tik_regs[0], padding=padding, cutoff=cutoff)
    grad_g_norm = np.gradient(g, pixel_size, axis=(0, 1)) / I0  # eq. (4.23) try to change to just the distance at 0.3m
    dx = np.gradient(grad_g_norm[0], pixel_size, axis=0)
    dy = np.gradient(grad_g_norm[1], pixel_size, axis=1)
    lap_phase = dx + dy
    phase = inverse_laplacian(lap_phase, pixel_size, tik_reg=tik_regs[1], padding=padding, cutoff=cutoff)

    # Finally, calculate the dark-field signal
    diff_coeff_lap = ((im1 - I0) / d1**2) + (sigma / (k * d1)) #try to change to 0.3m distance
    sub, div, std1, std2 = stdev_Comparison(((sigma / (k * d1)), ((im3 - im1) / d3**2)), 17)
    diff_coeff = (1.0 / I0) * inverse_laplacian(
        diff_coeff_lap, pixel_size, tik_reg=tik_regs[2], padding=padding, cutoff=cutoff
    )

    parameters = {
        "distance": distances,
        "wavenumber": wavenumber,
        "pixel_size": pixel_size,
        "tik_reg": tik_regs,
    }

    # Store the results in a dictionary
    results = {
        "I0": I0,
        "phase": phase,
        "diff_coeff": diff_coeff,
        "parameters": parameters,
        "DarkSubtracted": sub,
        "DarkDivided": div,
        "STD1": std1,
        "STD2": std2,
    }

    return results


def getDarkField(arrs, stride):
    subtracted, _, _, _ = stdev_Comparison(arrs, stride)
    return subtracted

def getPhaseEffects(imgs, dz, wavelength, pixel_size, 
                              reg=1e-18, edge_taper=True, highpass=True, 
                              highpassVal = 3,min_intensity=1e-6, cutsize = 0):
    """
    Retrieve X-ray phase map using a stabilized TIE approach with artifact suppression.
    Prevents divide-by-zero warnings and large-circle artifacts.
    """

    # Load & normalize
    if cutsize >= 1:
        I0 = imgs[0][:, cutsize:-cutsize]
        I1 = imgs[1][:, cutsize:-cutsize]
    else:
        I0 = imgs[0]
        I1 = imgs[1]
    eps = 1e-12

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

    cv2.imshow('img', np.abs(denom))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    denom[np.abs(denom) < reg] = np.inf  # prevent divide-by-zero
    phi_hat = RHS_hat / denom 
    phi_hat[0, 0] = 0.0  # remove DC term
    phi = np.fft.ifft2(phi_hat).real

    # Optional high-pass (remove slow background)
    if highpass:
        phi -= gaussian_filter(phi, sigma=highpassVal)

    # Normalize for display
    phi_norm = (phi - np.percentile(phi, 1)) / (np.percentile(phi, 99) - np.percentile(phi, 1) + eps)
    phi_norm = np.clip(phi_norm, 0, 1)

    return phi, phi_norm

