import os
import sys
import json
import cv2
import numpy as np


# ============================================================
# ROOT PATH
# ============================================================

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        '../../../../'
    )
)


# ============================================================
# NOVAVISION IMPORTS
# ============================================================

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


# ============================================================
# SIFT COMPARISON COMPONENT
# ============================================================

class SiftComparisonTest(Component):

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)

        # Request'i PackageModel ile doğruluyoruz.
        self.request.model = PackageModel(
            **self.request.data
        )

        # Minimum good match sayısı.
        self.good_matches_threshold = self.request.get_param(
            "GoodMatchesThreshold"
        )

        # Lowe Ratio Test threshold.
        self.ratio_threshold = self.request.get_param(
            "RatioThreshold"
        )

        # Matcher seçimi.
        self.matcher = self.request.get_param(
            "Matcher"
        )

        # Birinci SIFT output'u.
        self.sift_output_1 = self.request.get_param(
            "InputSIFTOutput1"
        )

        # İkinci SIFT output'u.
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
    # SIFT OUTPUT PARSING
    # ========================================================

    def _extract_keypoints_and_descriptors(
        self,
        sift_output
    ):
        """
        SIFT output'undan:

        - keypoint koordinatlarını
        - descriptor'ları

        ayırır.
        """

        keypoints_dicts = []
        descriptors = []

        # JSON string geldiyse Python objesine çeviriyoruz.
        if isinstance(sift_output, str):
            sift_output = json.loads(sift_output)

        # SIFT output'unun liste olması gerekiyor.
        if not isinstance(sift_output, list):
            raise ValueError(
                "SIFT output must be a list."
            )

        # Detection'ların içerisindeki keypoint'leri
        # topluyoruz.
        for detection in sift_output:

            for kp in detection.get("keyPoints", []):

                # Descriptor yoksa bu keypoint'i
                # matching işlemine almıyoruz.
                if "descriptor" not in kp:
                    continue

                # Keypoint koordinatlarını saklıyoruz.
                keypoints_dicts.append(
                    {
                        "pt": (
                            float(kp["cx"]),
                            float(kp["cy"])
                        )
                    }
                )

                # Descriptor'ı saklıyoruz.
                descriptors.append(
                    kp["descriptor"]
                )

        # Descriptor bulunamadıysa boş matris.
        if not descriptors:
            return (
                keypoints_dicts,
                np.empty(
                    (0, 128),
                    dtype=np.float32
                )
            )

        # OpenCV için float32 kullanıyoruz.
        descriptors = np.asarray(
            descriptors,
            dtype=np.float32
        )

        return (
            keypoints_dicts,
            descriptors
        )


    # ========================================================
    # NO MATCH RESULT
    # ========================================================

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


    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        try:

            # ------------------------------------------------
            # 1. SIFT output'larını alıyoruz.
            # ------------------------------------------------

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


            # ------------------------------------------------
            # 2. DEBUG
            # ------------------------------------------------

            print("")
            print("========================================")
            print("       SIFT COMPARISON DEBUG")
            print("========================================")
            print(
                "GoodMatchesThreshold:",
                self.good_matches_threshold
            )
            print(
                "RatioThreshold:",
                self.ratio_threshold
            )
            print(
                "Matcher:",
                self.matcher
            )
            print(
                "Descriptor1 count:",
                len(descriptors1)
            )
            print(
                "Descriptor2 count:",
                len(descriptors2)
            )
            print("========================================")
            print("")


            # ------------------------------------------------
            # 3. En az 2 descriptor gerekiyor.
            # ------------------------------------------------

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


            # ------------------------------------------------
            # 4. MATCHER
            # ------------------------------------------------

            if self.matcher == "FlannBasedMatcher":

                # SIFT descriptor'ları float olduğu için
                # FLANN + KD-Tree kullanıyoruz.
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

                # SIFT descriptor'ları için
                # Euclidean distance / L2 norm.
                matcher = cv2.BFMatcher(
                    cv2.NORM_L2,
                    crossCheck=False
                )

            else:

                raise ValueError(
                    f"Unsupported matcher: {self.matcher}"
                )


            # ------------------------------------------------
            # 5. KNN MATCHING
            # ------------------------------------------------

            matches = matcher.knnMatch(
                descriptors1,
                descriptors2,
                k=2
            )


            # ------------------------------------------------
            # 6. LOWE RATIO TEST
            # ------------------------------------------------

            good_matches = []

            for match_pair in matches:

                # İki sonuç yoksa ratio testi yapılamaz.
                if len(match_pair) < 2:
                    continue

                m, n = match_pair

                # Lowe Ratio Test.
                if (
                    m.distance
                    < self.ratio_threshold * n.distance
                ):
                    good_matches.append(m)


            # ------------------------------------------------
            # 7. GOOD MATCH COUNT
            # ------------------------------------------------

            good_matches_count = len(
                good_matches
            )

            print(
                "Good matches count:",
                good_matches_count
            )


            # ------------------------------------------------
            # 8. MATCH / NOMATCH
            # ------------------------------------------------

            images_match = (
                good_matches_count
                >= self.good_matches_threshold
            )


            # ------------------------------------------------
            # 9. KEYPOINTS
            # ------------------------------------------------

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


            # ------------------------------------------------
            # 10. CONNECTIONS
            # ------------------------------------------------

            connections = [
                Connection(
                    p1=m.queryIdx,
                    p2=m.trainIdx + offset
                )
                for m in good_matches
            ]


            # ------------------------------------------------
            # 11. FINAL DETECTION
            # ------------------------------------------------

            self.output_detections = [
                Detection(
                    boundingBox=None,

                    keyPoints=keypoints,

                    connections=connections,

                    # Good match sayısı.
                    confidence=float(
                        good_matches_count
                    ),

                    # Match = 1
                    # NoMatch = 0
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


        except Exception as e:

            # Gerçek hatayı gizlemiyoruz.
            print(
                "SIFT Comparison Error:",
                repr(e)
            )

            raise


        # ----------------------------------------------------
        # 12. RESPONSE
        # ----------------------------------------------------

        return build_response_sift_comparison_test(
            context=self
        )


# ============================================================
# EXECUTOR
# ============================================================

if __name__ == "__main__":

    Executor(
        sys.argv[1]
    ).run()