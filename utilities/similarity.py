# import the necessary packages
import numpy as np
import imutils
import cv2


def similar_region(image, template):
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    return maxVal, maxLoc, result


def similarity(image, template_image, minimum_zoom, maximum_zoom,
               steps_zoom_counts, maximum_not_update, debug=False):
    found = None
    not_update_count = 0
    template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (tH, tW) = template_gray.shape[:2]

    result = None
    # loop over the scales of the image
    for scale in np.linspace(minimum_zoom, maximum_zoom, steps_zoom_counts):
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = float(resized.shape[1]) / gray.shape[1]

        if not_update_count > maximum_not_update:
            break

        maxVal, maxLoc, result_similar = similar_region(resized, template_gray)

        # check to see if the iteration should be visualized
        if debug:
            print(maxVal, r)
            # draw a bounding box around the detected region
            res = imutils.resize(image, width=int(gray.shape[1] * scale))
            cv2.rectangle(res, (maxLoc[0], maxLoc[1]),
                          (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 20)
            cv2.namedWindow('Debug', cv2.WINDOW_NORMAL)
            cv2.imshow("Debug", res)
            cv2.waitKey(10)

        not_update_count += 1
        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            not_update_count = 0
            found = (maxVal, maxLoc, r)
            result = result_similar
            if debug:
                print("found:{}".format(found))

    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] / r), int(maxLoc[1] / r))
    (endX, endY) = (int((maxLoc[0] + tW) / r), int((maxLoc[1] + tH) / r))

    if debug:
        (minVal, maxVal, _, _) = cv2.minMaxLoc(result)
        result[maxLoc[1]:maxLoc[1] + tH, maxLoc[0]:maxLoc[0] + tW] = maxVal
        result = imutils.resize(result, width=int(result.shape[1] / r))
        heatmapshow = None
        heatmapshow = cv2.normalize(result, heatmapshow, alpha=0, beta=255,
                                    norm_type=cv2.NORM_MINMAX,
                                    dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        cv2.namedWindow('heatmapshow', cv2.WINDOW_NORMAL)
        cv2.imshow("heatmapshow", heatmapshow)

    return (startX, startY), (endX, endY)

