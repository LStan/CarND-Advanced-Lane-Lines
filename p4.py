#!python3
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os

class Undistorter():
    def __init__(self, images, nx, ny):
        """
        images - images with a chessboard
        nx, ny - number of cornes in x and y directions
        """

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((nx*ny,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx,ny) ,None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        # find calibration params and return
        _, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape, None, None)


    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)


class Warper:

    def __init__(self, src, dst):
        """
        src - tranformation source points
        dst - tranformation destination points
        """
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    def warp(self, img):
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_LINEAR)  # keep same size as input image
        return warped

    def unwarp(self, img):
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, self.Minv, img_size, flags=cv2.INTER_LINEAR)  # keep same size as input image
        return warped


def get_thresholded_binary_img(img, h_thresh = (15, 100), s_thresh = (120, 255), sx_thresh = (25, 100)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    #Threshold S and H channels
    color_binary = (s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])
    color_binary &= (h_channel > h_thresh[0]) & (h_channel <= h_thresh[1])

    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = ((scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1]))

    # combine color and gradient threshold results
    binary_output = color_binary | sxbinary
    binary_output = np.uint8(255*binary_output/np.max(binary_output))
    return binary_output


def find_lane_start(mask):
    histogram = np.sum(mask[mask.shape[0]/2:,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    left_lane_start = np.argmax(histogram[:midpoint])
    right_lane_start = np.argmax(histogram[midpoint:]) + midpoint

    return left_lane_start, right_lane_start

def find_lanes(mask, left_lane_start, right_lane_start, nwindows = 9):
     # Set height of windows
    window_height = np.int(mask.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = mask.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = left_lane_start
    rightx_current = right_lane_start
    # Set the width of the windows +/- margin
    margin_left = 100
    margin_right = 100
    # Set minimum number of pixels found to recenter window
    minpix = 550
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = mask.shape[0] - (window+1)*window_height
        win_y_high = mask.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin_left
        win_xleft_high = leftx_current + margin_left
        win_xright_low = rightx_current - margin_right
        win_xright_high = rightx_current + margin_right

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position and lower the margin
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            margin_left = 50
        else:
            margin_left = 100
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            margin_right = 50
        else:
            margin_right = 100

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # If found points fit a second order polynomial to them
    if (len(leftx) > 0):
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        left_fit = None
    if (len(rightx) > 0):
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        right_fit = None

    return left_fit, right_fit

def calc_poly(poly, y):
    return poly[0]*y**2 + poly[1]*y + poly[2]

def check_lanes(left_fit, right_fit, height, slices = 5):
    """
    This functions check if the lane is good. It searchs for min and max
    distances between left and right polinomials. These distances are checked in 'slices' points
    If difference between max and min is within a range then the lane is considered good
    """
    left_lane_start = calc_poly(left_fit, height)
    right_lane_start = calc_poly(right_fit, height)
    min_dist = max_dist = right_lane_start - left_lane_start

    slice_height = np.int(height/slices)
    for slice in range(1, slices):
        y = height - slice*slice_height
        left_lane_x = calc_poly(left_fit, y)
        right_lane_x = calc_poly(right_fit, y)
        dist = right_lane_x - left_lane_x
        if dist > max_dist:
            max_dist = dist
        if dist < min_dist:
            min_dist = dist

    return (max_dist-min_dist) < 200

def draw_poly_on_img(img, left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = calc_poly(left_fit, ploty)
    right_fitx = calc_poly(right_fit, ploty)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img).astype(np.uint8)
    #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(warp_zero, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = warper.unwarp(warp_zero)
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    return result

def write_roc_and_offset_on_img(img, left_fit, right_fit):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720 # meters per pixel in y dimension
    xm_per_pix = 3.7 / 800 # meters per pixel in x dimension

    # Prepare set of points for refitting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = calc_poly(left_fit, ploty)
    right_fitx = calc_poly(right_fit, ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    curverad  = (left_curverad + right_curverad) // 2
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    cv2.putText(img, 'Radius of curvature =  {}m'.format(int(curverad)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255))


    height = img.shape[0] # height of image (index of image bottom)
    width = img.shape[1] # width of image

    # Find the bottom pixel of the lane lines
    l_px = calc_poly(left_fit, height)
    r_px = calc_poly(right_fit, height)

    # Find the midpoint
    midpoint = (l_px + r_px) / 2

    # Find the offset from the centre of the frame, and then multiply by scale
    offset = (width/2 - midpoint) * xm_per_pix

    #print(offset, 'm')
    pos = "right" if offset >= 0 else "left"
    cv2.putText(img, 'Vehicle is {}m {} of center'.format(abs(round(offset, 3)), pos), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255))
    return img

def process_frame(img):
    global prev_left_fit, prev_right_fit, last_lane_was_good
    img = undistorter.undistort(img)

    binary_mask = get_thresholded_binary_img(img)
    binary_warped = warper.warp(binary_mask)

    height = img.shape[0]
    if last_lane_was_good:
        left_lane_start = calc_poly(prev_left_fit, height)
        right_lane_start = calc_poly(prev_right_fit, height)
    else:
        left_lane_start, right_lane_start = find_lane_start(binary_warped)
    left_fit, right_fit = find_lanes(binary_warped, left_lane_start, right_lane_start)
    if left_fit is None:
        left_fit = prev_left_fit
    if right_fit is None:
        right_fit = prev_right_fit

    if left_fit is not None and right_fit is not None:
        last_lane_was_good = check_lanes(left_fit, right_fit, height)
    else:
        last_lane_was_good = False

    if not last_lane_was_good:
        left_fit, right_fit = prev_left_fit, prev_right_fit

    if prev_left_fit is not None:
        # apply exponential moving average
        alpha = 0.2
        left_fit = alpha*left_fit + (1-alpha)*prev_left_fit
        right_fit = alpha*right_fit + (1-alpha)*prev_right_fit

    prev_left_fit, prev_right_fit = left_fit, right_fit

    if prev_left_fit is not None and prev_right_fit is not None:
        result = draw_poly_on_img(img, left_fit, right_fit)
        result = write_roc_and_offset_on_img(result, left_fit, right_fit)
    else:
        result = img
    #result = np.dstack((binary_warped, binary_warped, binary_warped))
    return result


images = glob.glob('./camera_cal/calibration*.jpg')
undistorter = Undistorter(images, 9, 6)

img_size = (1280, 720)

src_trans = np.float32(
    [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 35, img_size[1]],
    [(img_size[0] / 2 + 65), img_size[1] / 2 + 100]])
dst_trans = np.float32(
    [[(img_size[0] / 5), 0],
    [(img_size[0] / 5), img_size[1]],
    [(img_size[0] * 4 / 5), img_size[1]],
    [(img_size[0] * 4 / 5), 0]])

warper = Warper(src_trans, dst_trans)


prev_left_fit, prev_right_fit = None, None
last_lane_was_good = False

from moviepy.editor import VideoFileClip

clip1 = VideoFileClip("project_video.mp4")
project_output = 'project_output.mp4'
#clip1 = VideoFileClip("challenge_video.mp4")
#project_output = 'challenge_output.mp4'
project_clip = clip1.fl_image(process_frame)
project_clip.write_videofile(project_output, audio=False)