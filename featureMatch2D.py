import cv2 as cv
import argparse
import os


lowes_ratio = 0.4
def ORB_match(gry1,gry2,out_dir):

    gry1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    gry1 = cv.medianBlur(gry1, ksize = 5)

    gry2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
    gry2 = cv.medianBlur(gry2, ksize = 5)

    # Initiate ORB detector
    orb = cv.ORB_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(gry1,None)
    kp2, des2 = orb.detectAndCompute(gry2,None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    k = 40
    matches = matches[:k]
    im3 = cv.drawMatches(im1,kp1,im2,kp2,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite(f"{out_dir}/orb.png", im3)



def SIFT_match(gry1, gry2,out_dir):

    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gry1,None)
    kp2, des2 = sift.detectAndCompute(gry2,None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < lowes_ratio*n.distance:
            good.append([m])
            
    # cv.drawMatchesKnn expects list of lists as matches.
    im3 = cv.drawMatchesKnn(im1,kp1,im2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite(f"{out_dir}/sift.png", im3)
    len(good)


def FLANN_match(gry1, gry2,out_dir):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6, # 12
                    key_size = 12,     # 20
                    multi_probe_level = 1) #2

    # Initiate SIFT with FLANN parameters detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gry1,None)
    kp2, des2 = sift.detectAndCompute(gry2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict()
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < lowes_ratio*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv.DrawMatchesFlags_DEFAULT)
    im3 = cv.drawMatchesKnn(im1,kp1,im2,kp2,matches,None,**draw_params)
    cv.imwrite(f"{out_dir}/sift_with_flann.png", im3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="2D Feature Matching ")
    parser.add_argument("--img_left", type=str,default="capture_left/0.jpg", help="image captured by left camera")
    parser.add_argument("--img_right", type=str,default="capture_right/0.jpg",  help="image captured by right camera")
    parser.add_argument("--out_dir", type=str,default="match2d",  help="output dir to save matches")

    args = parser.parse_args()

    im1  = cv.imread(args.img_right)
    im2  = cv.imread(args.img_left)

    gry1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    gry1 = cv.medianBlur(gry1, ksize = 5)

    gry2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
    gry2 = cv.medianBlur(gry2, ksize = 5)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)


    ORB_match(gry1, gry2, args.out_dir)
    SIFT_match(gry1, gry2, args.out_dir)
    FLANN_match(gry1, gry2, args.out_dir)

    