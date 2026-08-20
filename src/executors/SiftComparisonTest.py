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

        super().__init__(
            request,
            bootstrap
        )

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

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    # ========================================================
    # EXTRACT SIFT KEYPOINTS + DESCRIPTORS
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

                keypoints_dicts.append(
                    {
                        "pt": (
                            float(kp["cx"]),
                            float(kp["cy"])
                        ),
                        "size": float(
                            kp["size"]
                        ),
                        "angle": float(
                            kp["angle"]
                        ),
                        "response": float(
                            kp["response"]
                        ),
                        "octave": int(
                            kp["octave"]
                        )
                    }
                )

                descriptors.append(
                    kp["descriptor"]
                )

        if len(descriptors) == 0:

            return (
                keypoints_dicts,
                np.empty(
                    (0, 128),
                    dtype=np.float32
                )
            )

        return (
            keypoints_dicts,
            np.asarray(
                descriptors,
                dtype=np.float32
            )
        )

    # ========================================================
    # CREATE VISUALIZATION
    # ========================================================

    def _create_visualization(
        self,
        keypoints1_dicts,
        keypoints2_dicts,
        good_matches
    ):

        # ----------------------------------------------------
        # GET IMAGE 1
        # ----------------------------------------------------

        image1_frame = Image.get_frame(
            img=self.visualization_input_1,
            redis_db=self.redis_db
        )

        # ----------------------------------------------------
        # GET IMAGE 2
        # ----------------------------------------------------

        image2_frame = Image.get_frame(
            img=self.visualization_input_2,
            redis_db=self.redis_db
        )

        if (
            image1_frame is None
            or image2_frame is None
        ):
            return None

        image1 = image1_frame.value
        image2 = image2_frame.value

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

        # ----------------------------------------------------
        # CREATE OPENCV KEYPOINTS
        # ----------------------------------------------------

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
                            kp["size"]
                        ),
                        1.0
                    )
                )
            )

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
                            kp["size"]
                        ),
                        1.0
                    )
                )
            )

        # ----------------------------------------------------
        # GOOD MATCHES
        # ----------------------------------------------------

        cv_matches = [
            match[0]
            for match in good_matches
        ]

        # ----------------------------------------------------
        # DRAW MATCHES
        # ----------------------------------------------------

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

    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        try:

            # =================================================
            # EXTRACT SIFT OUTPUT 1
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
            # EXTRACT SIFT OUTPUT 2
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
            # NOT ENOUGH DESCRIPTORS
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

                return build_response_sift_comparison_test(
                    context=self
                )

            # =================================================
            # CREATE MATCHER
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
                        [m]
                    )

            # =================================================
            # MATCH RESULT
            # =================================================

            good_matches_count = len(
                good_matches
            )

            images_match = (
                good_matches_count
                >= self.good_matches_threshold
            )

            # =================================================
            # DETECTION KEYPOINTS
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
                    p1=m[0].queryIdx,
                    p2=(
                        m[0].trainIdx
                        +
                        offset
                    )
                )

                for m in good_matches
            ]

            # =================================================
            # EXISTING DETECTION OUTPUT
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
            # VISUALIZATION
            # =================================================

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

                    # -----------------------------------------
                    # Get original image frame
                    # -----------------------------------------

                    visualization_frame = (
                        Image.get_frame(
                            img=self.visualization_input_1,
                            redis_db=self.redis_db
                        )
                    )

                    if visualization_frame is not None:

                        # -------------------------------------
                        # Replace image content
                        # -------------------------------------

                        visualization_frame.value = (
                            visualization
                        )

                        # -------------------------------------
                        # Store as NovaVision Image
                        # -------------------------------------

                        self.output_visualization = (
                            Image.set_frame(
                                img=visualization_frame,
                                package_uID=self.uID,
                                redis_db=self.redis_db
                            )
                        )

            except Exception as visualization_error:

                print(
                    "Visualization error:",
                    repr(
                        visualization_error
                    )
                )

                self.output_visualization = None

        except Exception as e:

            print(
                "SIFT Comparison Error:",
                repr(e)
            )

            # =================================================
            # FALLBACK DETECTION OUTPUT
            # =================================================

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

        # ====================================================
        # RESPONSE
        # ====================================================

        return build_response_sift_comparison_test(
            context=self
        )


if "__main__" == __name__:
    Executor(
        sys.argv[1]
    ).run()