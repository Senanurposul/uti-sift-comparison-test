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

from sdks.novavision.src.media.image import Image
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

        self.request.model = PackageModel(
            **self.request.data
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

        self.visualize = (
            self.request.get_param(
                "Visualize"
            )
        )

        # ====================================================
        # SIFT OUTPUTS
        # ====================================================

        self.sift_output_1 = (
            self.request.get_param(
                "InputSIFTOutput1"
            )
        )

        self.sift_output_2 = (
            self.request.get_param(
                "InputSIFTOutput2"
            )
        )

        # ====================================================
        # ORIGINAL IMAGES
        # ====================================================

        self.image_1 = (
            self.request.get_param(
                "InputImage1"
            )
        )

        self.image_2 = (
            self.request.get_param(
                "InputImage2"
            )
        )

        # ====================================================
        # OUTPUT VARIABLES
        # ====================================================

        self.output_detections = []
        self.output_visualization = None

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

        keypoints_dicts = []
        descriptors = []

        if isinstance(
            sift_output,
            str
        ):
            sift_output = json.loads(
                sift_output
            )

        if isinstance(
            sift_output,
            dict
        ):
            sift_output = sift_output.get(
                "value",
                sift_output
            )

        if not isinstance(
            sift_output,
            list
        ):
            raise ValueError(
                "SIFT output must be a list."
            )

        for detection in sift_output:

            if not isinstance(
                detection,
                dict
            ):
                continue

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
                        ),

                        "size": float(
                            kp.get(
                                "size",
                                10.0
                            )
                        ),

                        "angle": float(
                            kp.get(
                                "angle",
                                -1.0
                            )
                        ),

                        "response": float(
                            kp.get(
                                "response",
                                0.0
                            )
                        ),

                        "octave": int(
                            kp.get(
                                "octave",
                                0
                            )
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

    # ========================================================
    # CREATE VISUALIZATION
    # ========================================================

    def _build_visualization(
        self,
        img1,
        img2,
        keypoints1_dicts,
        keypoints2_dicts,
        good_matches
    ):

        cv_kp1 = [

            cv2.KeyPoint(
                float(kp["pt"][0]),
                float(kp["pt"][1]),
                float(kp["size"]),
                float(kp["angle"]),
                float(kp["response"]),
                int(kp["octave"]),
                0
            )

            for kp in keypoints1_dicts
        ]

        cv_kp2 = [

            cv2.KeyPoint(
                float(kp["pt"][0]),
                float(kp["pt"][1]),
                float(kp["size"]),
                float(kp["angle"]),
                float(kp["response"]),
                int(kp["octave"]),
                0
            )

            for kp in keypoints2_dicts
        ]

        result_img = cv2.drawMatches(
            img1,
            cv_kp1,
            img2,
            cv_kp2,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        return result_img

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

        self.output_visualization = None

        try:

            # =================================================
            # 1. GET SIFT OUTPUTS
            # =================================================

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

            # =================================================
            # 2. DEBUG
            # =================================================

            print(
                "========================================"
            )

            print(
                "SIFT COMPARISON DEBUG"
            )

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
                "Visualize:",
                self.visualize
            )

            print(
                "Descriptor1 count:",
                len(descriptors1)
            )

            print(
                "Descriptor2 count:",
                len(descriptors2)
            )

            print(
                "========================================"
            )

            # =================================================
            # 3. DESCRIPTOR CHECK
            # =================================================

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

            # =================================================
            # 4. MATCHER
            # =================================================

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

            # =================================================
            # 5. KNN MATCHING
            # =================================================

            matches = matcher.knnMatch(
                descriptors1,
                descriptors2,
                k=2
            )

            # =================================================
            # 6. LOWE RATIO TEST
            # =================================================

            good_matches = []

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

            # =================================================
            # 7. GOOD MATCH COUNT
            # =================================================

            good_matches_count = len(
                good_matches
            )

            print(
                "Good matches count:",
                good_matches_count
            )

            # =================================================
            # 8. MATCH / NOMATCH
            # =================================================

            images_match = (
                good_matches_count
                >= self.good_matches_threshold
            )

            # =================================================
            # 9. KEYPOINTS
            # =================================================

            all_keypoints_dicts = (
                keypoints1_dicts
                +
                keypoints2_dicts
            )

            offset = len(
                keypoints1_dicts
            )

            keypoints = [

                KeyPoints(
                    cx=float(
                        kp["pt"][0]
                    ),
                    cy=float(
                        kp["pt"][1]
                    ),
                    confidence=1.0
                )

                for kp in all_keypoints_dicts
            ]

            # =================================================
            # 10. CONNECTIONS
            # =================================================

            connections = [

                Connection(
                    p1=m.queryIdx,
                    p2=m.trainIdx + offset
                )

                for m in good_matches
            ]

            # =================================================
            # 11. DETECTION
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

            # =================================================
            # 12. VISUALIZATION
            # =================================================

            if self.visualize:

                print(
                    "Visualization requested."
                )

                # ---------------------------------------------
                # Get original images from Redis
                # ---------------------------------------------

                image1 = Image.get_frame(
                    img=self.image_1,
                    redis_db=self.redis_db
                )

                image2 = Image.get_frame(
                    img=self.image_2,
                    redis_db=self.redis_db
                )

                # ---------------------------------------------
                # Create visualization
                # ---------------------------------------------

                visualization = (
                    self._build_visualization(
                        image1.value,
                        image2.value,
                        keypoints1_dicts,
                        keypoints2_dicts,
                        good_matches
                    )
                )

                # ---------------------------------------------
                # Store visualization as NovaVision Image
                # ---------------------------------------------

                image1.value = visualization

                self.output_visualization = (
                    Image.set_frame(
                        img=image1,
                        package_uID=self.uID,
                        redis_db=self.redis_db
                    )
                )

                print(
                    "Visualization created successfully."
                )

            # =================================================
            # 13. RESPONSE
            # =================================================

            return build_response_sift_comparison_test(
                context=self
            )

        except Exception as e:

            print(
                "========================================"
            )

            print(
                "SIFT COMPARISON ERROR:"
            )

            print(
                type(e).__name__,
                str(e)
            )

            print(
                "========================================"
            )

            raise


# ============================================================
# EXECUTOR
# ============================================================

if __name__ == "__main__":

    Executor(
        sys.argv[1]
    ).run()