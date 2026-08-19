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

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.base.model import (
    KeyPoints,
    Detection,
    Connection
)
from sdks.novavision.src.helper.executor import Executor

from components.SiftComparisonTest.src.models.PackageModel import (
    PackageModel
)

from components.SiftComparisonTest.src.utils.response import (
    build_response_sift_comparison_test
)


class SiftComparisonTest(Component):

    def __init__(self, request, bootstrap):

        super().__init__(
            request,
            bootstrap
        )

        self.request.model = PackageModel(
            **(self.request.data)
        )

        # ====================================================
        # INPUTS
        # ====================================================

        self.image_1 = self.request.get_param(
            "InputSIFTOutput1"
        )

        self.image_2 = self.request.get_param(
            "InputSIFTOutput2"
        )

        # ====================================================
        # CONFIGS
        # ====================================================

        self.good_matches_threshold = (
            self.request.get_param(
                "GoodMatchesThreshold"
            )
        )

        self.ratio_threshold = (
            self.request.get_param(
                "RatioThreshold"
            )
        )

        self.matcher = (
            self.request.get_param(
                "Matcher"
            )
        )

    # ========================================================
    # BOOTSTRAP
    # ========================================================

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    # ========================================================
    # SIFT
    # ========================================================

    def calculate_sift(self, image):

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

            descriptors = descriptors.astype(
                np.float32
            )

        return (
            keypoints,
            descriptors
        )

    # ========================================================
    # MATCHER
    # ========================================================

    def create_matcher(self):

        if self.matcher == "BFMatcher":

            return cv2.BFMatcher(
                cv2.NORM_L2,
                crossCheck=False
            )

        # Default: FLANN

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

    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        try:

            print("")
            print("========================================")
            print(" SIFT COMPARISON EXECUTOR STARTED")
            print("========================================")

            # =================================================
            # IMAGE 1
            # =================================================

            img1 = Image.get_frame(
                img=self.image_1,
                redis_db=self.redis_db
            )

            # =================================================
            # IMAGE 2
            # =================================================

            img2 = Image.get_frame(
                img=self.image_2,
                redis_db=self.redis_db
            )

            print(
                "Image 1 shape:",
                img1.value.shape
            )

            print(
                "Image 2 shape:",
                img2.value.shape
            )

            # =================================================
            # GRAYSCALE
            # =================================================

            if len(img1.value.shape) == 3:

                gray1 = cv2.cvtColor(
                    img1.value,
                    cv2.COLOR_BGR2GRAY
                )

            else:

                gray1 = img1.value

            if len(img2.value.shape) == 3:

                gray2 = cv2.cvtColor(
                    img2.value,
                    cv2.COLOR_BGR2GRAY
                )

            else:

                gray2 = img2.value

            # =================================================
            # SIFT
            # =================================================

            keypoints1, descriptors1 = (
                self.calculate_sift(
                    gray1
                )
            )

            keypoints2, descriptors2 = (
                self.calculate_sift(
                    gray2
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

            print(
                "Descriptors 1:",
                len(descriptors1)
            )

            print(
                "Descriptors 2:",
                len(descriptors2)
            )

            # =================================================
            # MATCHING
            # =================================================

            good_matches = []

            if (
                len(descriptors1) > 0
                and
                len(descriptors2) > 0
            ):

                matcher = self.create_matcher()

                matches = matcher.knnMatch(
                    descriptors1,
                    descriptors2,
                    k=2
                )

                for pair in matches:

                    if len(pair) < 2:
                        continue

                    m, n = pair

                    if (
                        m.distance
                        <
                        self.ratio_threshold * n.distance
                    ):

                        good_matches.append(m)

            # =================================================
            # COMPARISON RESULT
            # =================================================

            good_matches_count = len(
                good_matches
            )

            images_match = (
                good_matches_count
                >=
                self.good_matches_threshold
            )

            print(
                "Good matches:",
                good_matches_count
            )

            print(
                "Images match:",
                images_match
            )

            # =================================================
            # KEYPOINTS
            # =================================================

            keypoints = []

            for kp in keypoints1:

                keypoints.append(
                    KeyPoints(
                        cx=float(kp.pt[0]),
                        cy=float(kp.pt[1]),
                        confidence=1.0
                    )
                )

            offset = len(keypoints)

            for kp in keypoints2:

                keypoints.append(
                    KeyPoints(
                        cx=float(kp.pt[0]),
                        cy=float(kp.pt[1]),
                        confidence=1.0
                    )
                )

            # =================================================
            # CONNECTIONS
            # =================================================

            connections = []

            for match in good_matches:

                connections.append(
                    Connection(
                        p1=match.queryIdx,
                        p2=match.trainIdx + offset
                    )
                )

            # =================================================
            # OUTPUT DETECTIONS
            # =================================================

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

            print("")
            print("========================================")
            print(" SIFT COMPARISON OUTPUT CREATED")
            print("========================================")

            # =================================================
            # RESPONSE
            # =================================================

            packageModel = (
                build_response_sift_comparison_test(
                    context=self
                )
            )

            return packageModel

        except Exception as e:

            print("")
            print("========================================")
            print(" SIFT COMPARISON ERROR")
            print("========================================")

            print(
                repr(e)
            )

            print("========================================")
            print("")

            raise


# ============================================================
# EXECUTOR
# ============================================================

if "__main__" == __name__:

    Executor(
        sys.argv[1]
    ).run()