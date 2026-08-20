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
        "../../../../"
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
from sdks.novavision.src.media.image import Image
from sdks.novavision.src.helper.executor import Executor


# ============================================================
# PACKAGE IMPORTS
# ============================================================

from components.SiftComparisonTest.src.utils.response import (
    build_response_sift_comparison_test
)

from components.SiftComparisonTest.src.models.PackageModel import (
    PackageModel
)


# ============================================================
# SIFT COMPARISON TEST
# ============================================================

class SiftComparisonTest(Component):

    def __init__(self, request, bootstrap):

        super().__init__(
            request,
            bootstrap
        )

        # --------------------------------------------------------
        # Request model
        # --------------------------------------------------------

        self.request.model = PackageModel(
            **self.request.data
        )

        # --------------------------------------------------------
        # MATCH CONFIGS
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # VISUALIZATION MATCH LIMIT
        #
        # Enabled:
        #     VisualizationMatchesValue -> sayı
        #
        # Disabled:
        #     VisualizationMatchesValue -> None
        #
        # None / 0 olduğunda bütün match'ler çizilir.
        # --------------------------------------------------------

        self.visualization_matches = (
            self.request.get_param(
                "VisualizationMatchesValue"
            )
        )

        if self.visualization_matches is None:
            self.visualization_matches = 0

        try:
            self.visualization_matches = int(
                self.visualization_matches
            )
        except (
            TypeError,
            ValueError
        ):
            self.visualization_matches = 0

        # --------------------------------------------------------
        # SIFT OUTPUT 1
        # --------------------------------------------------------

        self.sift_output_1 = (
            self.request.get_param(
                "InputSIFTOutput1"
            )
        )

        # --------------------------------------------------------
        # SIFT OUTPUT 2
        # --------------------------------------------------------

        self.sift_output_2 = (
            self.request.get_param(
                "InputSIFTOutput2"
            )
        )

        # --------------------------------------------------------
        # ORIGINAL IMAGE 1
        # --------------------------------------------------------

        self.visualization_input_1 = (
            self.request.get_param(
                "InputVisualization1"
            )
        )

        # --------------------------------------------------------
        # ORIGINAL IMAGE 2
        # --------------------------------------------------------

        self.visualization_input_2 = (
            self.request.get_param(
                "InputVisualization2"
            )
        )

        # --------------------------------------------------------
        # OUTPUTS
        # --------------------------------------------------------

        self.output_detections = []

        self.output_visualization = None


    # ============================================================
    # BOOTSTRAP
    # ============================================================

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}


    # ============================================================
    # SIFT OUTPUT PARSING
    # ============================================================

    def _extract_keypoints_and_descriptors(
        self,
        sift_output
    ):

        keypoints_dicts = []

        descriptors = []

        # --------------------------------------------------------
        # JSON string ise parse et
        # --------------------------------------------------------

        if isinstance(
            sift_output,
            str
        ):

            sift_output = json.loads(
                sift_output
            )

        # --------------------------------------------------------
        # Output list olmalı
        # --------------------------------------------------------

        if not isinstance(
            sift_output,
            list
        ):

            raise ValueError(
                "SIFT output must be a list."
            )

        # --------------------------------------------------------
        # Detection -> KeyPoints
        # --------------------------------------------------------

        for detection in sift_output:

            for kp in detection.get(
                "keyPoints",
                []
            ):

                # Descriptor yoksa matching'e alma
                if "descriptor" not in kp:
                    continue

                # ------------------------------------------------
                # Keypoint
                # ------------------------------------------------

                keypoints_dicts.append(
                    {
                        "pt": (
                            float(
                                kp["cx"]
                            ),
                            float(
                                kp["cy"]
                            )
                        ),

                        "size": float(
                            kp.get(
                                "size",
                                1.0
                            )
                        ),

                        "angle": float(
                            kp.get(
                                "angle",
                                -1.0
                            )
                        )
                    }
                )

                # ------------------------------------------------
                # Descriptor
                # ------------------------------------------------

                descriptors.append(
                    kp["descriptor"]
                )

        # --------------------------------------------------------
        # Descriptor yoksa
        # --------------------------------------------------------

        if not descriptors:

            return (
                keypoints_dicts,

                np.empty(
                    (
                        0,
                        128
                    ),
                    dtype=np.float32
                )
            )

        # --------------------------------------------------------
        # OpenCV için float32
        # --------------------------------------------------------

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
        keypoints1_dicts,
        keypoints2_dicts,
        good_matches
    ):

        # --------------------------------------------------------
        # IMAGE 1
        # --------------------------------------------------------

        image1_frame = Image.get_frame(
            img=self.visualization_input_1,
            redis_db=self.redis_db
        )

        # --------------------------------------------------------
        # IMAGE 2
        # --------------------------------------------------------

        image2_frame = Image.get_frame(
            img=self.visualization_input_2,
            redis_db=self.redis_db
        )

        # --------------------------------------------------------
        # Image bulunamadı
        # --------------------------------------------------------

        if (
            image1_frame is None
            or image2_frame is None
        ):

            return None

        image1 = image1_frame.value

        image2 = image2_frame.value

        # --------------------------------------------------------
        # Image boş
        # --------------------------------------------------------

        if (
            image1 is None
            or image2 is None
        ):

            return None

        image1 = np.asarray(
            image1
        )

        image2 = np.asarray(
            image2
        )

        if (
            image1.size == 0
            or image2.size == 0
        ):

            return None

        # --------------------------------------------------------
        # OpenCV KeyPoints - IMAGE 1
        # --------------------------------------------------------

        cv_keypoints1 = []

        for kp in keypoints1_dicts:

            cv_keypoints1.append(
                cv2.KeyPoint(
                    x=float(
                        kp["pt"][0]
                    ),

                    y=float(
                        kp["pt"][1]
                    ),

                    size=max(
                        float(
                            kp.get(
                                "size",
                                1.0
                            )
                        ),
                        1.0
                    )
                )
            )

        # --------------------------------------------------------
        # OpenCV KeyPoints - IMAGE 2
        # --------------------------------------------------------

        cv_keypoints2 = []

        for kp in keypoints2_dicts:

            cv_keypoints2.append(
                cv2.KeyPoint(
                    x=float(
                        kp["pt"][0]
                    ),

                    y=float(
                        kp["pt"][1]
                    ),

                    size=max(
                        float(
                            kp.get(
                                "size",
                                1.0
                            )
                        ),
                        1.0
                    )
                )
            )

        # ========================================================
        # VISUALIZATION MATCH LIMIT
        # ========================================================

        # --------------------------------------------------------
        # Disabled
        # --------------------------------------------------------
        #
        # 0 -> bütün match'ler
        # --------------------------------------------------------

        if self.visualization_matches <= 0:

            matches_to_draw = list(
                good_matches
            )

        # --------------------------------------------------------
        # Enabled
        # --------------------------------------------------------
        #
        # Kullanıcının verdiği sayı kadar
        # en iyi match'i çiz.
        # --------------------------------------------------------

        else:

            sorted_matches = sorted(
                good_matches,
                key=lambda match: match[0].distance
            )

            matches_to_draw = (
                sorted_matches[
                    :self.visualization_matches
                ]
            )

        # --------------------------------------------------------
        # Good matches -> DMatch
        # --------------------------------------------------------

        cv_matches = [
            match[0]
            for match in matches_to_draw
        ]

        # ========================================================
        # DRAW MATCHES
        # ========================================================

        visualization = cv2.drawMatches(
            image1,
            cv_keypoints1,

            image2,
            cv_keypoints2,

            cv_matches,

            None,

            flags=(
                cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
        )

        return visualization


    # ============================================================
    # RUN
    # ============================================================

    def run(self):

        try:

            # ====================================================
            # 1. SIFT OUTPUT 1
            # ====================================================

            (
                keypoints1_dicts,
                descriptors1
            ) = (
                self._extract_keypoints_and_descriptors(
                    self.sift_output_1
                )
            )

            # ====================================================
            # 2. SIFT OUTPUT 2
            # ====================================================

            (
                keypoints2_dicts,
                descriptors2
            ) = (
                self._extract_keypoints_and_descriptors(
                    self.sift_output_2
                )
            )

            # ====================================================
            # 3. DESCRIPTOR CHECK
            # ====================================================

            if (
                len(descriptors1) < 2
                or len(descriptors2) < 2
            ):

                self.output_detections = [
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

                self.output_visualization = None

                return (
                    build_response_sift_comparison_test(
                        context=self
                    )
                )

            # ====================================================
            # 4. MATCHER
            # ====================================================

            if self.matcher == "FlannBasedMatcher":

                # ------------------------------------------------
                # FLANN + KD Tree
                # ------------------------------------------------

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

                # ------------------------------------------------
                # Brute Force + L2
                # ------------------------------------------------

                matcher = cv2.BFMatcher(
                    cv2.NORM_L2,
                    crossCheck=False
                )

            else:

                raise ValueError(
                    f"Unsupported matcher: {self.matcher}"
                )

            # ====================================================
            # 5. KNN MATCHING
            # ====================================================

            matches = matcher.knnMatch(
                descriptors1,
                descriptors2,
                k=2
            )

            # ====================================================
            # 6. LOWE RATIO TEST
            # ====================================================

            good_matches = []

            for match_pair in matches:

                # Ratio test için iki sonuç gerekiyor
                if len(match_pair) < 2:
                    continue

                m, n = match_pair

                if (
                    m.distance
                    <
                    self.ratio_threshold * n.distance
                ):

                    good_matches.append(
                        [m]
                    )

            # ====================================================
            # 7. GOOD MATCH COUNT
            # ====================================================

            good_matches_count = len(
                good_matches
            )

            # ====================================================
            # 8. MATCH / NOMATCH
            # ====================================================

            images_match = (
                good_matches_count
                >=
                self.good_matches_threshold
            )

            # ====================================================
            # 9. ALL KEYPOINTS
            # ====================================================

            all_keypoints_dicts = (
                keypoints1_dicts
                +
                keypoints2_dicts
            )

            # ----------------------------------------------------
            # Image 2 keypoint index offset
            # ----------------------------------------------------

            offset = len(
                keypoints1_dicts
            )

            # ====================================================
            # 10. NOVAVISION KEYPOINTS
            # ====================================================

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

            # ====================================================
            # 11. CONNECTIONS
            # ====================================================

            connections = [

                Connection(
                    p1=match[0].queryIdx,

                    p2=(
                        match[0].trainIdx
                        +
                        offset
                    )
                )

                for match in good_matches
            ]

            # ====================================================
            # 12. OUTPUT DETECTIONS
            # ====================================================
            #
            # Burada BÜTÜN good_matches korunuyor.
            #
            # Visualization limiti bu output'u etkilemiyor.
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
            # 13. VISUALIZATION
            # ====================================================

            self.output_visualization = None

            try:

                visualization = (
                    self._create_visualization(
                        keypoints1_dicts,
                        keypoints2_dicts,
                        good_matches
                    )
                )

                if visualization is not None:

                    # ------------------------------------------------
                    # Original image frame
                    # ------------------------------------------------

                    visualization_frame = (
                        Image.get_frame(
                            img=self.visualization_input_1,
                            redis_db=self.redis_db
                        )
                    )

                    if visualization_frame is not None:

                        # ------------------------------------------------
                        # Visualization'ı image value'ya koy
                        # ------------------------------------------------

                        visualization_frame.value = (
                            visualization
                        )

                        # ------------------------------------------------
                        # NovaVision Image oluştur
                        # ------------------------------------------------

                        self.output_visualization = (
                            Image.set_frame(
                                img=visualization_frame,
                                package_uID=self.uID,
                                redis_db=self.redis_db
                            )
                        )

            except Exception as visualization_error:

                print(
                    "SIFT COMPARISON TEST - "
                    "Visualization Error:",
                    repr(
                        visualization_error
                    )
                )

                self.output_visualization = None

        # ========================================================
        # MAIN ERROR
        # ========================================================

        except Exception as error:

            print(
                "SIFT COMPARISON TEST - ERROR:",
                repr(error)
            )

            self.output_detections = [

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

            self.output_visualization = None

        # ========================================================
        # RESPONSE
        # ========================================================

        return (
            build_response_sift_comparison_test(
                context=self
            )
        )


# ============================================================
# EXECUTOR
# ============================================================

if __name__ == "__main__":

    Executor(
        sys.argv[1]
    ).run()