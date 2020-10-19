# import the necessary packages
import unittest
import cv2
import similarity


TEST_MAXVAL = 16406257.0
TEST_MAXLOC = (834, 524)
TEST_P1 = (657, 490)
TEST_P2 = (681, 515)


class TestSimilarityMethods(unittest.TestCase):

    def test_similar_region(self):
        test_image = cv2.imread('test_image.jpg')
        query_test_image = cv2.imread('query_test_image.png')

        maxVal, maxLoc = similarity.similar_region(test_image, query_test_image)

        self.assertEqual(maxVal, TEST_MAXVAL)
        self.assertEqual(maxLoc, TEST_MAXLOC)

    def test_similarity(self):
        test_image = cv2.imread('test_image.jpg')
        query_test_image = cv2.imread('query_test_image2.png')

        minimum_zoom = 6
        maximum_zoom = 6
        steps_zoom_counts = 1
        maximum_not_update = 2

        (startX, startY), (endX, endY) = similarity.similarity(test_image, query_test_image, minimum_zoom,
                                                               maximum_zoom,
                                                               steps_zoom_counts,
                                                               maximum_not_update)

        self.assertEqual((startX, startY), TEST_P1)
        self.assertEqual((endX, endY), TEST_P2)


if __name__ == '__main__':
    unittest.main()
