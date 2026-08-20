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
from sdks.novavision.src.helper.executor import Executor

from components.SiftComparisonTest.src.utils.response import (
    build_response_sift_comparison_test
)

from components.SiftComparisonTest.src.models.PackageModel import (
    PackageModel
)


class SiftComparisonTest(Component):

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)

        # --------------------------------------------------------
        # Request model
        # --------------------------------------------------------

        self.request.model = PackageModel(
            **self.request.data
        )

        # --------------------------------------------------------
        # Configs
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
        # SIFT outputs
        # --------------------------------------------------------

        self.sift_output_1 = self.request.get_param(
            "InputSIFTOutput1"
        )

        self.sift_output_2 = self.request.get_param(
            "InputSIFTOutput2"
        )

        self.output_detections = []

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

        # JSON string geldiyse parse et
        if isinstance(sift_output, str):
            sift_output = json.loads(sift_output)

        if not isinstance(sift_output, list):
            raise ValueError(
                "SIFT output must be a list."
            )

        # Detection'ların içindeki keypoint'leri gez
        for detection in sift_output:

            for kp in detection.get(
                "keyPoints",
                []
            ):

                # Descriptor yoksa kullanma
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

        # Descriptor yoksa boş matris
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
    # NO MATCH
    # ============================================================

    def _no_match_result(self):

        return [
            Detection(
                boundingBox=None,
                keyPoints=[],
                connections=[],
                confidence=0.0,
                classId=0,
                classLabel="NoMatch",
                imgUID=self.uID
            )
        ]

    # ============================================================
    # RUN
    # ============================================================

    def run(self):

        try:

            # ----------------------------------------------------
            # 1. SIFT outputlarını ayır
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # 2. KNN için en az 2 descriptor gerekli
            # ----------------------------------------------------

            if (
                len(descriptors1) < 2
                or len(descriptors2) < 2
            ):

                self.output_detections = (
                    self._no_match_result()
                )

                return build_response_sift_comparison_test(
                    context=self
                )

            # ----------------------------------------------------
            # 3. Matcher
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # 4. KNN matching
            # ----------------------------------------------------

            matches = matcher.knnMatch(
                descriptors1,
                descriptors2,
                k=2
            )

            # ----------------------------------------------------
            # 5. Lowe Ratio Test
            # ----------------------------------------------------

            good_matches = []

            for match_pair in matches:

                if len(match_pair) < 2:
                    continue

                m, n = match_pair

                if (
                    m.distance
                    < self.ratio_threshold * n.distance
                ):
                    good_matches.append(m)

            # ----------------------------------------------------
            # 6. Good match sayısı
            # ----------------------------------------------------

            good_matches_count = len(
                good_matches
            )

            print(
                "SIFT Comparison:",
                "good_matches =",
                good_matches_count
            )

            # ----------------------------------------------------
            # 7. Match / NoMatch
            # ----------------------------------------------------

            images_match = (
                good_matches_count
                >= self.good_matches_threshold
            )

            # ----------------------------------------------------
            # 8. Keypoint'leri birleştir
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # 9. Connections
            # ----------------------------------------------------

            connections = [
                Connection(
                    p1=m.queryIdx,
                    p2=m.trainIdx + offset
                )
                for m in good_matches
            ]

            # ----------------------------------------------------
            # 10. Detection
            # ----------------------------------------------------

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

            print(
                "SIFT Comparison:",
                "result =",
                "Match" if images_match
                else "NoMatch"
            )

            # ----------------------------------------------------
            # 11. Response
            # ----------------------------------------------------

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