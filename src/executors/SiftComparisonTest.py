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
            **self.request.data
        )

        self.image_1 = self.request.get_param(
            "InputSIFTOutput1"
        )

        self.image_2 = self.request.get_param(
            "InputSIFTOutput2"
        )

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

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def create_matcher(self):

        if self.matcher == "BFMatcher":

            return cv2.BFMatcher(
                cv2.NORM_L2,
                crossCheck=False
            )

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

    def run(self):

        print("========================================")
        print("SIFT COMPARISON EXECUTOR STARTED")
        print("========================================")

        # ----------------------------------------------------
        # IMAGE 1
        # ----------------------------------------------------

        img1 = Image.get_frame(
            img=self.image_1,
            redis_db=self.redis_db
        )

        # ----------------------------------------------------
        # IMAGE 2
        # ----------------------------------------------------

        img2 = Image.get_frame(
            img=self.image_2,
            redis_db=self.redis_db
        )

        print("Image 1:", img1.value.shape)
        print("Image 2:", img2.value.shape)

        # ----------------------------------------------------
        # GRAYSCALE
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # SIFT
        # ----------------------------------------------------

        sift = cv2.SIFT_create()

        keypoints_1, descriptors_1 = (
            sift.detectAndCompute(
                gray1,
                None
            )
        )

        keypoints_2, descriptors_2 = (
            sift.detectAndCompute(
                gray2,
                None
            )
        )

        print(
            "Keypoints 1:",
            len(keypoints_1)
        )

        print(
            "Keypoints 2:",
            len(keypoints_2)
        )

        # ----------------------------------------------------
        # DESCRIPTORS
        # ----------------------------------------------------

        if descriptors_1 is None:

            descriptors_1 = np.empty(
                (0, 128),
                dtype=np.float32
            )

        if descriptors_2 is None:

            descriptors_2 = np.empty(
                (0, 128),
                dtype=np.float32
            )

        # ----------------------------------------------------
        # MATCHING
        # ----------------------------------------------------

        good_matches = []

        if (
            len(descriptors_1) > 0
            and
            len(descriptors_2) > 0
        ):

            matcher = self.create_matcher()

            matches = matcher.knnMatch(
                descriptors_1,
                descriptors_2,
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

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        good_matches_count = len(
            good_matches
        )

        images_match = (
            good_matches_count
            >= self.good_matches_threshold
        )

        print(
            "Good matches:",
            good_matches_count
        )

        print(
            "Images match:",
            images_match
        )

        # ----------------------------------------------------
        # KEYPOINT OUTPUT
        # ----------------------------------------------------

        def keypoints_to_list(keypoints):

            result = []

            for kp in keypoints:

                result.append({
                    "cx": float(kp.pt[0]),
                    "cy": float(kp.pt[1]),
                    "size": float(kp.size),
                    "angle": float(kp.angle),
                    "response": float(kp.response),
                    "octave": int(kp.octave)
                })

            return result

        # ----------------------------------------------------
        # DESCRIPTOR OUTPUT
        # ----------------------------------------------------

        def descriptors_to_list(descriptors):

            if descriptors is None:
                return []

            return descriptors.astype(
                np.float32
            ).tolist()

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        output = {

            "keypoints_1":
                keypoints_to_list(
                    keypoints_1
                ),

            "descriptors_1":
                descriptors_to_list(
                    descriptors_1
                ),

            "keypoints_2":
                keypoints_to_list(
                    keypoints_2
                ),

            "descriptors_2":
                descriptors_to_list(
                    descriptors_2
                ),

            "good_matches_count":
                good_matches_count,

            "images_match":
                images_match
        }

        self.output_detections = [
            output
        ]

        print(
            "SIFT Comparison output created successfully."
        )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return build_response_sift_comparison_test(
            context=self
        )


if "__main__" == __name__:

    Executor(
        sys.argv[1]
    ).run()