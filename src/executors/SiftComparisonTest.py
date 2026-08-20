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

        super().__init__(
            request,
            bootstrap
        )

        self.request.model = PackageModel(
            **self.request.data
        )

        # ====================================================
        # MATCH CONFIGS
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

        # ====================================================
        # VISUALIZATION CONFIG
        # ====================================================

        self.visualization_matches = (
            self.request.get_param(
                "ConfigVisualizationMatches"
            )
        )

        # ----------------------------------------------------
        # Varsayılan
        # ----------------------------------------------------

        if self.visualization_matches is None:

            self.visualization_matches = (
                "VisualizationMatchesEnabled"
            )

        # ====================================================
        # SIFT INPUTS
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
        # IMAGE INPUTS
        # ====================================================

        self.visualization_input_1 = (
            self.request.get_param(
                "InputVisualization1"
            )
        )

        self.visualization_input_2 = (
            self.request.get_param(
                "InputVisualization2"
            )
        )

        # ====================================================
        # OUTPUTS
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
    # EXTRACT KEYPOINTS + DESCRIPTORS
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

        for detection in sift_output:

            for kp in detection.get(
                "keyPoints",
                []
            ):

                if kp.get(
                    "descriptor"
                ) is None:

                    continue

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

                descriptors.append(
                    kp["descriptor"]
                )

        return (
            keypoints_dicts,
            np.array(
                descriptors,
                dtype=np.float32
            )
        )


    # ========================================================
    # VISUALIZATION
    # ========================================================

    def _create_visualization(
        self,
        frame1,
        frame2,
        keypoints1_dicts,
        keypoints2_dicts,
        good_matches
    ):

        # ----------------------------------------------------
        # OpenCV keypoints
        # ----------------------------------------------------

        cv_keypoints1 = [

            cv2.KeyPoint(
                float(
                    kp["pt"][0]
                ),
                float(
                    kp["pt"][1]
                ),
                max(
                    float(
                        kp["size"]
                    ),
                    1.0
                )
            )

            for kp in keypoints1_dicts
        ]

        cv_keypoints2 = [

            cv2.KeyPoint(
                float(
                    kp["pt"][0]
                ),
                float(
                    kp["pt"][1]
                ),
                max(
                    float(
                        kp["size"]
                    ),
                    1.0
                )
            )

            for kp in keypoints2_dicts
        ]

        # ====================================================
        # VISUALIZATION MATCH SELECTION
        # ====================================================

        if (
            self.visualization_matches
            ==
            "VisualizationMatchesDisabled"
        ):

            # ------------------------------------------------
            # DISABLED
            # Bütün eşleşmeleri çiz
            # ------------------------------------------------

            matches_to_draw = list(
                good_matches
            )

        else:

            # ------------------------------------------------
            # ENABLED
            # Kullanıcının girdiği sayı kadar çiz
            # ------------------------------------------------

            max_matches = 20

            # ------------------------------------------------
            # Enabled objesinden değeri almaya çalış
            # ------------------------------------------------

            try:

                config_value = (
                    self.visualization_matches
                )

                if hasattr(
                    config_value,
                    "value"
                ):

                    enabled_value = (
                        config_value.value
                    )

                    if hasattr(
                        enabled_value,
                        "VisualizationMatchesValue"
                    ):

                        max_matches = int(
                            enabled_value
                            .VisualizationMatchesValue
                            .value
                        )

                elif isinstance(
                    config_value,
                    dict
                ):

                    enabled_value = (
                        config_value.get(
                            "value",
                            config_value
                        )
                    )

                    if isinstance(
                        enabled_value,
                        dict
                    ):

                        value_config = (
                            enabled_value.get(
                                "VisualizationMatchesValue"
                            )
                        )

                        if isinstance(
                            value_config,
                            dict
                        ):

                            max_matches = int(
                                value_config.get(
                                    "value",
                                    20
                                )
                            )

            except Exception:

                max_matches = 20

            # ------------------------------------------------
            # En iyi eşleşmeleri seç
            # ------------------------------------------------

            sorted_matches = sorted(
                good_matches,
                key=lambda match: match.distance
            )

            matches_to_draw = sorted_matches[
                :max_matches
            ]

        # ====================================================
        # DRAW
        # ====================================================

        visualization = cv2.drawMatches(
            frame1,
            cv_keypoints1,

            frame2,
            cv_keypoints2,

            matches_to_draw,

            None,

            flags=(
                cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
        )

        return visualization


    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        try:

            # =================================================
            # SIFT OUTPUT 1
            # =================================================

            (
                keypoints1_dicts,
                descriptors1
            ) = (
                self._extract_keypoints_and_descriptors(
                    self.sift_output_1
                )
            )

            # =================================================
            # SIFT OUTPUT 2
            # =================================================

            (
                keypoints2_dicts,
                descriptors2
            ) = (
                self._extract_keypoints_and_descriptors(
                    self.sift_output_2
                )
            )

            # =================================================
            # DESCRIPTOR CHECK
            # =================================================

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

            # =================================================
            # MATCHER
            # =================================================

            if self.matcher == "BFMatcher":

                matcher = cv2.BFMatcher(
                    cv2.NORM_L2
                )

            else:

                matcher = cv2.FlannBasedMatcher(

                    dict(
                        algorithm=1,
                        trees=5
                    ),

                    dict(
                        checks=50
                    )
                )

            # =================================================
            # KNN MATCH
            # =================================================

            matches = matcher.knnMatch(
                descriptors1,
                descriptors2,
                k=2
            )

            # =================================================
            # LOWE RATIO TEST
            # =================================================

            good_matches = []

            for pair in matches:

                if len(pair) < 2:

                    continue

                m, n = pair

                if (
                    m.distance
                    <
                    self.ratio_threshold * n.distance
                ):

                    good_matches.append(
                        m
                    )

            # =================================================
            # MATCH COUNT
            # =================================================

            good_matches_count = len(
                good_matches
            )

            images_match = (
                good_matches_count
                >= self.good_matches_threshold
            )

            # =================================================
            # ALL KEYPOINTS
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
            # CONNECTIONS
            # =================================================

            connections = [

                Connection(
                    p1=int(
                        match.queryIdx
                    ),
                    p2=int(
                        match.trainIdx
                    ) + offset
                )

                for match in good_matches
            ]

            # =================================================
            # OUTPUT DETECTIONS
            #
            # BÜTÜN MATCH'LER BURADA KORUNUYOR.
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
            # IMAGE 1
            # =================================================

            image1_frame = Image.get_frame(
                img=self.visualization_input_1,
                redis_db=self.redis_db
            )

            # =================================================
            # IMAGE 2
            # =================================================

            image2_frame = Image.get_frame(
                img=self.visualization_input_2,
                redis_db=self.redis_db
            )

            # =================================================
            # IMAGE CHECK
            # =================================================

            if (
                image1_frame is not None
                and image2_frame is not None
                and image1_frame.value is not None
                and image2_frame.value is not None
            ):

                frame1 = np.asarray(
                    image1_frame.value
                )

                frame2 = np.asarray(
                    image2_frame.value
                )

                # =================================================
                # CREATE VISUALIZATION
                # =================================================

                visualization = (
                    self._create_visualization(
                        frame1,
                        frame2,
                        keypoints1_dicts,
                        keypoints2_dicts,
                        good_matches
                    )
                )

                # =================================================
                # SAVE VISUALIZATION
                # =================================================

                image1_frame.value = visualization

                self.output_visualization = (
                    Image.set_frame(
                        img=image1_frame,
                        package_uID=self.uID,
                        redis_db=self.redis_db
                    )
                )

        except Exception as e:

            print(
                "SiftComparisonTest Error:",
                repr(e)
            )

            if not self.output_detections:

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

        # =====================================================
        # RESPONSE
        # =====================================================

        return (
            build_response_sift_comparison_test(
                context=self
            )
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    Executor(
        sys.argv[1]
    ).run()