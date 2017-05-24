
import tensorflow as tf
import cv2
import numpy as np
import random
import glob



X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

# 32개의 필터
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))



L1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME'))

L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, 0.8)

# 64개의 필터
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.relu(tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME'))
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


L2 = tf.reshape(L2, [-1, 7 * 7 * 64])
L2 = tf.nn.dropout(L2, 0.8)

# 256개의 필터
W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
L3 = tf.nn.relu(tf.matmul(L2, W3))
L3 = tf.nn.dropout(L3, 0.5)

#출력
W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L3, W4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= model,labels= Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 저장된 뉴런 weight 값 받아오기
saver = tf.train.Saver()
saver.restore(sess, "./CNN_test.ckpt")


total_cost=0

# 틀렸을 경우 다시 교육
def test2(roi,num,total_cost=total_cost):
    if (num == 0):
        y = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    if (num == 1):
        y = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    if (num == 2):
        y = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
    if (num == 3):
        y = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
    if (num == 4):
        y = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
    if (num == 5):
        y = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
    if (num == 6):
        y = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
    if (num == 7):
        y = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
    if (num == 8):
        y = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
    if (num == 9):
        y = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    _, cost_val = sess.run([optimizer, cost], feed_dict={X: roi, Y: y})
    print("Prediction: ", sess.run(
        tf.argmax(model, 1), feed_dict={X: roi})),
    #sess.run(optimizer, feed_dict={X: roi, Y: y})






# 이미지를 받아오면 뉴런에 트레이닝
def test(img,num,total_cost=total_cost):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi=gray
    roi = cv2.resize(gray, (28, 28))

    roi =roi / 255.0
    roi = roi.reshape(-1, 28, 28, 1)

    if (num == 0):
        y = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    if (num == 1):
        y = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    if (num == 2):
        y = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
    if (num == 3):
        y = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
    if (num == 4):
        y = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
    if (num == 5):
        y = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
    if (num == 6):
        y = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
    if (num == 7):
        y = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
    if (num == 8):
        y = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
    if (num == 9):
        y = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    _, cost_val = sess.run([optimizer, cost], feed_dict={X: roi, Y: y})
    print("Prediction: ", sess.run(
        tf.argmax(model, 1), feed_dict={X: roi})),
    #sess.run(optimizer, feed_dict={X: roi, Y: y})



#트레이닝
# count=0
# while (count < 20):
#     count += 1
#     print(count)
#     for fname in glob.glob('images/0/*.bmp'):
#         test(fname,0)
#     for fname in glob.glob('images/1/*.bmp'):
#         test(fname,1)
#     for fname in glob.glob('images/2/*.bmp'):
#         test(fname,2)
#     for fname in glob.glob('images/3/*.bmp'):
#         test(fname,3)
#     for fname in glob.glob('images/4/*.bmp'):
#         test(fname,4)
#     for fname in glob.glob('images/5/*.bmp'):
#         test(fname,5)
#     for fname in glob.glob('images/6/*.bmp'):
#         test(fname,6)
#     for fname in glob.glob('images/7/*.bmp'):
#         test(fname,7)
#     for fname in glob.glob('images/8/*.bmp'):
#         test(fname,8)
#     for fname in glob.glob('images/9/*.bmp'):
#         test(fname,9)
#     #if count% 20 == 0:
#     saver.save(sess, './CNN.ckpt', global_step=count)



#트레이닝 후 테스트 이미지

nn=0
# 계속 반복하기 용도
for nn in range(500000):
    image = cv2.imread("images/test.bmp")
    # cv2.imshow("test",image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    digitCnts = []

    # loop over the digit area candidates
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # if the contour is sufficiently large, it must be a digit
        if w >= 3 and (h >= 10 and h <= 300):
            digitCnts.append(c)
    n = 0
    for c in digitCnts:
        # extract the digit ROI
        (x, y, w, h) = cv2.boundingRect(c)
        marginw=(int)(w*0.3)
        marginh=(int)(h*0.3)
        x=x-marginw
        y=y-marginh
        w=w+(2*marginw)
        h=h+(2*marginh)
        roi = gray[y:y + h, x:x + w]

        # compute the width and height of each of the 7 segments
        # we are going to examine
        (roiH, roiW) = roi.shape
        (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
        dHC = int(roiH * 0.05)

        # define the set of 7 segments
        segments = [
            ((0, 0), (w, dH)),  # top
            ((0, 0), (dW, h // 2)),  # top-left
            ((w - dW, 0), (w, h // 2)),  # top-right
            ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
            ((0, h // 2), (dW, h)),  # bottom-left
            ((w - dW, h // 2), (w, h)),  # bottom-right
            ((0, h - dH), (w, h))  # bottom
        ]
        on = [0] * len(segments)
        cv2.imshow('ts',roi)

        roi = cv2.resize(roi, (28, 28))

        r = random.randint(0, 1000000)
        k=str(r)
        print(r)
        cv2.imwrite('images/'+k + '.bmp', roi)

        roi = roi / 255.0
        cv2.imshow('t2', roi)

        roi=roi.reshape(-1, 28, 28, 1)

        #print(roi)
        print("Prediction: ", sess.run(
            tf.argmax(model, 1), feed_dict={X: roi })),


        k=cv2.waitKey()
        print('key', k)

        if (k > 47) & (k < 58):
            for t in range(100):
                test2(roi, k-48)


        saver.save(sess, './CNN_test_train.ckpt',)


    print('=========================')

