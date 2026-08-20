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

from components.SiftComparisonTest.src.utils.response import (
    build_response_sift_comparison_test
)

from components.SiftComparisonTest.src.models.PackageModel import (
    PackageModel
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

        self.image_1 = self.request.get_param(
            "InputImage1"
        )

        self.image_2 = self.request.get_param(
            "InputImage2"
        )

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    # ============================================================
    # SIFT OUTPUT PARSING
    # ============================================================

    def _extract_keypoints_and_descriptors(self, sift_output):

        keypoints_dicts = []
        descriptors = []

        if isinstance(sift_output, str):
            sift_output = json.loads(sift_output)

        if not isinstance(sift_output, list):
            raise ValueError(
                "SIFT output must be a list."
            )

        for detection in sift_output:

            for kp in detection.get("keyPoints", []):

                if "descriptor" not in kp:
                    continue

                keypoints_dicts.append(
                    {
                        "pt": (
                            float(kp["cx"]),
                            float(kp["cy"])
                        ),
                        "size": float(
                            kp.get("size", 1.0)
                        ),
                        "angle": float(
                            kp.get("angle", -1.0)
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
    # VISUALIZATION
    # ============================================================

    def _create_visualization(
        self,
        image1,
        image2,
        keypoints1_dicts,
        keypoints2_dicts,
        good_matches
    ):

        if image1 is None or image2 is None:
            return None

        # OpenCV KeyPoint listesi
        cv_keypoints1 = []

        for kp in keypoints1_dicts:

            cv_keypoints1.append(
                cv2.KeyPoint(
                    float(kp["pt"][0]),
                    float(kp["pt"][1]),
                    float(kp["size"])
                )
            )

        cv_keypoints2 = []

        for kp in keypoints2_dicts:

            cv_keypoints2.append(
                cv2.KeyPoint(
                    float(kp["pt"][0]),
                    float(kp["pt"][1]),
                    float(kp["size"])
                )
            )

        # Good matches'i görselleştir
        visualization = cv2.drawMatches(
            image1,
            cv_keypoints1,
            image2,
            cv_keypoints2,
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

            # ----------------------------------------------------
            # 2. Görüntüleri al
            # ----------------------------------------------------

            image1_frame = None
            image2_frame = None

            if self.image_1 is not None:

                image1_frame = Image.get_frame(
                    img=self.image_1,
                    redis_db=self.redis_db
                )

            if self.image_2 is not None:

                image2_frame = Image.get_frame(
                    img=self.image_2,
                    redis_db=self.redis_db
                )

            # ----------------------------------------------------
            # 3. Yeterli descriptor yoksa
            # ----------------------------------------------------

            if (
                len(descriptors1) < 2
                or len(descriptors2) < 2
            ):

                self.output_detections = (
                    self._no_match_result()
                )

                self.output_visualization = None

                return build_response_sift_comparison_test(
                    context=self
                )

            # ----------------------------------------------------
            # 4. Matcher
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
            # 5. KNN MATCH
            # ----------------------------------------------------

            matches = matcher.knnMatch(
                descriptors1,
                descriptors2,
                k=2
            )

            # ----------------------------------------------------
            # 6. Lowe Ratio Test
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
            # 7. Match count
            # ----------------------------------------------------

            good_matches_count = len(
                good_matches
            )

            images_match = (
                good_matches_count
                >= self.good_matches_threshold
            )

            # ----------------------------------------------------
            # 8. Keypoints
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

            # ----------------------------------------------------
            # 11. VISUALIZATION
            # ----------------------------------------------------

            self.output_visualization = None

            if (
                image1_frame is not None
                and image2_frame is not None
            ):

                visualization = self._create_visualization(
                    image1_frame.value,
                    image2_frame.value,
                    keypoints1_dicts,
                    keypoints2_dicts,
                    good_matches
                )

                # Image objesini kullan
                image1_frame.value = visualization

                self.output_visualization = Image.set_frame(
                    img=image1_frame,
                    package_uID=self.uID,
                    redis_db=self.redis_db
                )

            # ----------------------------------------------------
            # 12. RESPONSE
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
# MAIN
# ============================================================

if __name__ == "__main__":

    Executor(
        sys.argv[1]
    ).run()