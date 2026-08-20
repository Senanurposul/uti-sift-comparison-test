import os
import sys
import json

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
from sdks.novavision.src.media.image import Image
from sdks.novavision.src.helper.executor import Executor

from components.SiftComparisonTest.src.models.PackageModel import (
    PackageModel
)

from components.SiftComparisonTest.src.utils.response import (
    build_response_sift_comparison_test
)


class SiftComparisonTest(Component):

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)

        self.request.model = PackageModel(
            **self.request.data
        )

        # --------------------------------------------------------
        # CONFIGS
        # --------------------------------------------------------

        self.good_matches_threshold = self.request.get_param(
            "GoodMatchesThreshold"
        )

        self.ratio_threshold = self.request.get_param(
            "RatioThreshold"
        )

        self.matcher = self.request.get_param(
            "Matcher"
        )

        # --------------------------------------------------------
        # SIFT OUTPUTS
        # --------------------------------------------------------

        self.sift_output_1 = self.request.get_param(
            "InputSIFTOutput1"
        )

        self.sift_output_2 = self.request.get_param(
            "InputSIFTOutput2"
        )

        # --------------------------------------------------------
        # ORIGINAL IMAGES
        # --------------------------------------------------------

        self.image_input_1 = self.request.get_param(
            "InputImage1"
        )

        self.image_input_2 = self.request.get_param(
            "InputImage2"
        )

        self.output_detections = []
        self.output_visualization = None

    # ============================================================
    # BOOTSTRAP
    # ============================================================

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    # ============================================================
    # SIFT OUTPUT PARSE
    # ============================================================

    def _extract_keypoints_and_descriptors(
        self,
        sift_output
    ):

        keypoints_dicts = []
        descriptors = []

        if isinstance(sift_output, str):
            sift_output = json.loads(sift_output)

        if not isinstance(sift_output, list):
            raise ValueError(
                "SIFT output must be a list."
            )

        for detection in sift_output:

            for kp in detection.get(
                "keyPoints",
                []
            ):

                if "descriptor" not in kp:
                    continue

                keypoints_dicts.append(
                    {
                        "pt": (
                            float(kp["cx"]),
                            float(kp["cy"])
                        )
                    }
                )

                descriptors.append(
                    kp["descriptor"]
                )

        if not descriptors:

            return (
                keypoints_dicts,
                np.empty(
                    (0, 128),
                    dtype=np.float32
                )
            )

        descriptors = np.asarray(
            descriptors,
            dtype=np.float32
        )

        return (
            keypoints_dicts,
            descriptors
        )

    # ============================================================
    # VISUALIZATION
    # ============================================================

    def _create_visualization(
        self,
        image_1,
        image_2,
        keypoints1,
        keypoints2,
        good_matches
    ):

        if image_1 is None or image_2 is None:
            return None

        # OpenCV images
        img1 = image_1.copy()
        img2 = image_2.copy()

        # --------------------------------------------------------
        # KeyPoint objelerine dönüştür
        # --------------------------------------------------------

        kp1 = [
            cv2.KeyPoint(
                float(kp["pt"][0]),
                float(kp["pt"][1]),
                1
            )
            for kp in keypoints1
        ]

        kp2 = [
            cv2.KeyPoint(
                float(kp["pt"][0]),
                float(kp["pt"][1]),
                1
            )
            for kp in keypoints2
        ]

        # --------------------------------------------------------
        # Match çizimi
        # --------------------------------------------------------

        visualization = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        return visualization

    # ============================================================
    # RUN
    # ============================================================

    def run(self):

        try:

            # ====================================================
            # 1. SIFT OUTPUT
            # ====================================================

            (
                keypoints1_dicts,
                descriptors1
            ) = self._extract_keypoints_and_descriptors(
                self.sift_output_1
            )

            (
                keypoints2_dicts,
                descriptors2
            ) = self._extract_keypoints_and_descriptors(
                self.sift_output_2
            )

            print(
                "SIFT Comparison:",
                "descriptor1 =",
                len(descriptors1),
                "descriptor2 =",
                len(descriptors2)
            )

            # ====================================================
            # 2. MATCHER
            # ====================================================

            good_matches = []

            if (
                len(descriptors1) >= 2
                and len(descriptors2) >= 2
            ):

                if self.matcher == "FlannBasedMatcher":

                    index_params = {
                        "algorithm": 1,
                        "trees": 5
                    }

                    search_params = {
                        "checks": 50
                    }

                    matcher = cv2.FlannBasedMatcher(
                        index_params,
                        search_params
                    )

                elif self.matcher == "BFMatcher":

                    matcher = cv2.BFMatcher(
                        cv2.NORM_L2,
                        crossCheck=False
                    )

                else:

                    raise ValueError(
                        f"Unsupported matcher: {self.matcher}"
                    )

                matches = matcher.knnMatch(
                    descriptors1,
                    descriptors2,
                    k=2
                )

                # =================================================
                # 3. LOWE RATIO
                # =================================================

                for match_pair in matches:

                    if len(match_pair) < 2:
                        continue

                    m, n = match_pair

                    if (
                        m.distance
                        <
                        self.ratio_threshold * n.distance
                    ):
                        good_matches.append(m)

            # ====================================================
            # 4. RESULT
            # ====================================================

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
                "Result:",
                "Match" if images_match else "NoMatch"
            )

            # ====================================================
            # 5. KEYPOINTS
            # ====================================================

            all_keypoints_dicts = (
                keypoints1_dicts
                + keypoints2_dicts
            )

            offset = len(
                keypoints1_dicts
            )

            keypoints = [
                KeyPoints(
                    cx=float(kp["pt"][0]),
                    cy=float(kp["pt"][1]),
                    confidence=1.0
                )
                for kp in all_keypoints_dicts
            ]

            # ====================================================
            # 6. CONNECTIONS
            # ====================================================

            connections = [
                Connection(
                    p1=m.queryIdx,
                    p2=m.trainIdx + offset
                )
                for m in good_matches
            ]

            # ====================================================
            # 7. OUTPUT DETECTION
            # ====================================================

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

            # ====================================================
            # 8. ORIGINAL IMAGE'LERİ AL
            # ====================================================

            img1 = Image.get_frame(
                img=self.image_input_1,
                redis_db=self.redis_db
            )

            img2 = Image.get_frame(
                img=self.image_input_2,
                redis_db=self.redis_db
            )

            # ====================================================
            # 9. VISUALIZATION
            # ====================================================

            visualization = self._create_visualization(
                img1.value,
                img2.value,
                keypoints1_dicts,
                keypoints2_dicts,
                good_matches
            )

            # ====================================================
            # 10. VISUALIZATION IMAGE
            # ====================================================

            if visualization is not None:

                img1.value = visualization

                self.output_visualization = Image.set_frame(
                    img=img1,
                    package_uID=self.uID,
                    redis_db=self.redis_db
                )

            else:

                self.output_visualization = None

            # ====================================================
            # 11. RESPONSE
            # ====================================================

            return build_response_sift_comparison_test(
                context=self
            )

        except Exception as e:

            print(
                "SIFT Comparison Error:",
                repr(e)
            )

            raise


if __name__ == "__main__":

    Executor(
        sys.argv[1]
    ).run()