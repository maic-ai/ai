# import the necessary packages
import argparse
import cv2
import similarity


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--template", required=True, help="Path to template image")
    ap.add_argument("-i", "--image", required=True,
                    help="Path to images where template will be matched")
    ap.add_argument("-v", "--debug",
                    help="Flag indicating whether or not to debug")
    args = vars(ap.parse_args())

    template_image = cv2.imread(args["template"])
    image = cv2.imread(args["image"])

    minimum_zoom = 1
    maximum_zoom = 8
    steps_zoom_counts = 15
    maximum_not_update = 2

    debug = args.get("debug", False)

    cv2.namedWindow('Template image', cv2.WINDOW_NORMAL)
    cv2.imshow("Template image", template_image)

    (startX, startY), (endX, endY) = similarity.similarity(image, template_image,
                                                           minimum_zoom, maximum_zoom,
                                                           steps_zoom_counts,
                                                           maximum_not_update, debug)

    # draw a bounding box around the detected result and display the image
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.imshow("Result", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
