import os
import sys
import cv2
import numpy as np

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        '../../../../'
    )
)

from sdks.novavision.src.base.component import Component
from sdks.novavision.src.base.model import (
    KeyPoints,
    Detection,
    Connection
)

from sdks.novavision.src.helper.executor import Executor

from components.SiftComparisonTest.src.utils.response import (
    build_response_sift_comparison_test
)

from components.SiftComparisonTest.src.models.PackageModel import (
    PackageModel
)


class SiftComparisonTest(Component):

    def __init__(self, request, bootstrap):

        super().__init__(
            request,
            bootstrap
        )

        self.request.model = PackageModel(
            **self.request.data
        )

        self.good_matches_threshold = self.request.get_param(
            "GoodMatchesThreshold"
        )

        self.ratio_threshold = self.request.get_param(
            "RatioThreshold"
        )

        self.matcher = self.request.get_param(
            "Matcher"
        )

        self.visualize = self.request.get_param(
            "Visualize"
        )

        self.sift_output_1 = self.request.get_param(
            "InputSIFTOutput1"
        )

        self.sift_output_2 = self.request.get_param(
            "InputSIFTOutput2"
        )


    # ========================================================
    # BOOTSTRAP
    # ========================================================

    @staticmethod
    def bootstrap(config: dict) -> dict:

        return {}


    # ========================================================
    # IMAGE CONVERSION
    # ========================================================

    def _get_image(self, image):

        if image is None:
            raise ValueError(
                "Image input is None."
            )

        if isinstance(image, list):

            if len(image) == 0:
                raise ValueError(
                    "Image input list is empty."
                )

            image = image[0]

        if isinstance(image, np.ndarray):
            return image

        if hasattr(image, "value"):

            image = image.value

            if isinstance(image, np.ndarray):
                return image

        if isinstance(image, bytes):

            image_array = np.frombuffer(
                image,
                dtype=np.uint8
            )

            decoded = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )

            if decoded is None:
                raise ValueError(
                    "Image could not be decoded."
                )

            return decoded

        raise TypeError(
            f"Unsupported image type: {type(image)}"
        )


    # ========================================================
    # SIFT
    # ========================================================

    def _calculate_sift(self, image):

        sift = cv2.SIFT_create()

        keypoints, descriptors = (
            sift.detectAndCompute(
                image,
                None
            )
        )

        if descriptors is None:

            descriptors = np.empty(
                (0, 128),
                dtype=np.float32
            )

        else:

            descriptors = np.asarray(
                descriptors,
                dtype=np.float32
            )

        return keypoints, descriptors


    # ========================================================
    # MATCHER
    # ========================================================

    def _create_matcher(self):

        if self.matcher == "FlannBasedMatcher":

            index_params = {
                "algorithm": 1,
                "trees": 5
            }

            search_params = {
                "checks": 50
            }

            return cv2.FlannBasedMatcher(
                index_params,
                search_params
            )

        elif self.matcher == "BFMatcher":

            return cv2.BFMatcher(
                cv2.NORM_L2,
                crossCheck=False
            )

        else:

            raise ValueError(
                f"Unsupported matcher: {self.matcher}"
            )


    # ========================================================
    # VISUALIZATION
    # ========================================================

    def _build_visualization(
        self,
        image_1,
        image_2,
        keypoints_1,
        keypoints_2,
        good_matches
    ):

        canvas = cv2.drawMatches(
            image_1,
            keypoints_1,
            image_2,
            keypoints_2,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        return canvas


    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        try:

            # ------------------------------------------------
            # 1. IMAGE INPUT
            # ------------------------------------------------

            image_1 = self._get_image(
                self.sift_output_1
            )

            image_2 = self._get_image(
                self.sift_output_2
            )


            # ------------------------------------------------
            # 2. GRAYSCALE
            # ------------------------------------------------

            if len(image_1.shape) == 3:

                gray_1 = cv2.cvtColor(
                    image_1,
                    cv2.COLOR_BGR2GRAY
                )

            else:

                gray_1 = image_1


            if len(image_2.shape) == 3:

                gray_2 = cv2.cvtColor(
                    image_2,
                    cv2.COLOR_BGR2GRAY
                )

            else:

                gray_2 = image_2


            # ------------------------------------------------
            # 3. SIFT
            # ------------------------------------------------

            keypoints1, descriptors1 = (
                self._calculate_sift(
                    gray_1
                )
            )

            keypoints2, descriptors2 = (
                self._calculate_sift(
                    gray_2
                )
            )


            print(
                "Keypoints 1:",
                len(keypoints1)
            )

            print(
                "Keypoints 2:",
                len(keypoints2)
            )


            # ------------------------------------------------
            # 4. MATCHING
            # ------------------------------------------------

            good_matches = []

            if (
                len(descriptors1) >= 2
                and len(descriptors2) >= 2
            ):

                matcher = self._create_matcher()

                matches = matcher.knnMatch(
                    descriptors1,
                    descriptors2,
                    k=2
                )


                # ------------------------------------------------
                # 5. LOWE RATIO TEST
                # ------------------------------------------------

                for match_pair in matches:

                    if len(match_pair) < 2:
                        continue

                    m, n = match_pair

                    if (
                        m.distance
                        < self.ratio_threshold * n.distance
                    ):

                        good_matches.append(m)


            # ------------------------------------------------
            # 6. GOOD MATCH COUNT
            # ------------------------------------------------

            good_matches_count = len(
                good_matches
            )

            print(
                "Good matches:",
                good_matches_count
            )


            # ------------------------------------------------
            # 7. MATCH / NO MATCH
            # ------------------------------------------------

            images_match = (
                good_matches_count
                >= self.good_matches_threshold
            )


            # ------------------------------------------------
            # 8. KEYPOINTS
            # ------------------------------------------------

            all_keypoints = (
                keypoints1 + keypoints2
            )

            offset = len(
                keypoints1
            )

            keypoints = [

                KeyPoints(
                    cx=float(kp.pt[0]),
                    cy=float(kp.pt[1]),
                    confidence=1.0
                )

                for kp in all_keypoints

            ]


            # ------------------------------------------------
            # 9. CONNECTIONS
            # ------------------------------------------------

            connections = [

                Connection(
                    p1=match.queryIdx,
                    p2=match.trainIdx + offset
                )

                for match in good_matches

            ]


            # ------------------------------------------------
            # 10. OUTPUT DETECTIONS
            # ------------------------------------------------

            self.output_detections = [

                Detection(

                    boundingBox=None,

                    keyPoints=keypoints,

                    connections=connections,

                    confidence=float(
                        good_matches_count
                    ),

                    classId=(
                        1
                        if images_match
                        else 0
                    ),

                    classLabel=(
                        "Match"
                        if images_match
                        else "NoMatch"
                    ),

                    imgUID=self.uID

                )

            ]


            # ------------------------------------------------
            # 11. VISUALIZATION
            # ------------------------------------------------

            self.output_visualization = None

            if self.visualize:

                self.output_visualization = (
                    self._build_visualization(
                        image_1,
                        image_2,
                        keypoints1,
                        keypoints2,
                        good_matches
                    )
                )


            # ------------------------------------------------
            # 12. RESPONSE
            # ------------------------------------------------

            return build_response_sift_comparison_test(
                context=self
            )


        except Exception as e:

            print(
                "SIFT Comparison Error:",
                repr(e)
            )

            raise


# ============================================================
# EXECUTOR
# ============================================================

if __name__ == "__main__":

    Executor(
        sys.argv[1]
    ).run()