IMAGE_PATH = '/Users/cylopez/Documents/projects/license-plate-recognition/us/us3.jpg'

# data_des = return_data_openalpr(IMAGE_PATH)


# min_xcoord, min_ycoord, max_xcoord, max_ycoord = get_coord(data_des)   
min_xcoord, min_ycoord, max_xcoord, max_ycoord = 207, 209, 258, 296
print(min_ycoord)
print(max_ycoord)
img = cv2.imread(IMAGE_PATH)
height_img = img.shape[0] # number of rows
width_img = img.shape[1] # number of cols
img_crop = img[int(min_xcoord):int(max_xcoord),int(min_ycoord):int(max_ycoord)]
height_img_crop = max_ycoord - min_ycoord
print(height_img_crop)
print(img_crop)
plt.imshow(img_crop)
plt.show()

# image processing colors
img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
img_blur = cv2.blur(img_gray, (3,3),0)
ddepth = cv2.CV_8U
ret, th = cv2.threshold(img_gray,80,255,cv2.THRESH_BINARY_INV)
plt.imshow(th)
plt.show()

# find contours in image
img_contours = copy.copy(th)
contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours_after_size_verification = []
for contour in contours:
    if verifySize(th, contour, height_img_crop):
        contours_after_size_verification.append(contour)


print(contours_after_size_verification)
for contour in contours_after_size_verification:
        x, y, w, h = cv2.boundingRect(contour)
        paddingw = int(w/5)
        paddingh = int(h/5)
        img_crop = cv2.getRectSubPix(img_contours, (w+paddingw,h+paddingh), (x+w/2,y+h/2))
        resized_image = cv2.resize(img_crop,(34,85))
        plt.imshow(resized_image)
        plt.show()

def verifySize(th, contour, img_height):
    # char sizes are 45x77
    # char sizes are 2.5"x2.5*(3 to 4)
    x, y, w, h = cv2.boundingRect(contour)
    print(w, h)
    min_char_to_plate_aspect = 40/224
    aspect = 2.5/(2.5*2.5) #based on eyeballing
    charAspect = w/h
    error = 0.4
    minHeight = 30
    minAspect = 0.2
    maxAspect = aspect+aspect*error
    print(min_char_to_plate_aspect)
    print(h/img_height)

    if charAspect > minAspect and charAspect < maxAspect and (h/img_height)>=min_char_to_plate_aspect:
        cv2.rectangle(th,(x,y),(x+w,y+h),(0,255,0),2)
        plt.imshow(th), plt.title("Bounding Box Contours")
        plt.show()        
        return True
    else:
        return False
