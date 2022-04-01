import cv2 as cv


def ORB_match(gry1,gry2):

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
    matches = matches[:40]
    im3 = cv.drawMatches(im1,kp1,im2,kp2,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite("ORB_RESULTS.png", im3)

    print('# matches: ', len(matches))



def SIFT_match(gry1, gry2):

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
        if m.distance < 0.75*n.distance:
            good.append([m])
            
    # cv.drawMatchesKnn expects list of lists as matches.
    im3 = cv.drawMatchesKnn(im1,kp1,im2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite("SIFT_RESULTS.png", im3)
    len(good)


def FLANN_match(gry1, gry2):
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
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv.DrawMatchesFlags_DEFAULT)
    im3 = cv.drawMatchesKnn(im1,kp1,im2,kp2,matches,None,**draw_params)
    cv.imwrite("SIFT_w_FLANN_RESULTS.png", im3)

if __name__ == "__main__":
    im1  = cv.imread('CAM1_20.jpg')
    im2  = cv.imread('CAM2_20.jpg')


    gry1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    gry1 = cv.medianBlur(gry1, ksize = 5)

    gry2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
    gry2 = cv.medianBlur(gry2, ksize = 5)

    FLANN_match(gry1, gry2)