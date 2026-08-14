import os
import sys
import json

import cv2
import numpy as np

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../../../"
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

from components.SiftComparison.src.utils.response import (
    build_response_sift_comparison
)

from components.SiftComparison.src.models.PackageModel import (
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
        # OUTPUT VARIABLES
        # ====================================================

        self.output_detections = []
        self.visualization_image = None

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    # ========================================================
    # EXTRACT KEYPOINTS AND DESCRIPTORS
    # ========================================================

    def _extract_keypoints_and_descriptors(
        self,
        sift_output
    ):

        keypoints_dicts = []
        descriptors = []

        if isinstance(sift_output, str):

            sift_output = json.loads(
                sift_output
            )

        if isinstance(sift_output, dict):

            sift_output = sift_output.get(
                "value",
                sift_output
            )

        if not isinstance(
            sift_output,
            list
        ):

            return (
                keypoints_dicts,
                np.empty(
                    (0, 128),
                    dtype=np.float32
                )
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

                keypoints_dicts.append({

                    "pt": (
                        float(kp["cx"]),
                        float(kp["cy"])
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
                })

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
        image1,
        image2,
        keypoints1_dicts,
        keypoints2_dicts,
        good_matches
    ):

        cv_keypoints1 = []

        for kp in keypoints1_dicts:

            cv_keypoints1.append(
                cv2.KeyPoint(
                    float(kp["pt"][0]),
                    float(kp["pt"][1]),
                    float(kp["size"]),
                    float(kp["angle"]),
                    float(kp["response"]),
                    int(kp["octave"]),
                    0
                )
            )

        cv_keypoints2 = []

        for kp in keypoints2_dicts:

            cv_keypoints2.append(
                cv2.KeyPoint(
                    float(kp["pt"][0]),
                    float(kp["pt"][1]),
                    float(kp["size"]),
                    float(kp["angle"]),
                    float(kp["response"]),
                    int(kp["octave"]),
                    0
                )
            )

        matches_for_draw = [
            match[0]
            for match in good_matches
        ]

        visualization = cv2.drawMatches(
            image1,
            cv_keypoints1,
            image2,
            cv_keypoints2,
            matches_for_draw,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        return visualization

    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        try:

            print("=== SIFT COMPARISON START ===")

            # =================================================
            # GET ORIGINAL IMAGES
            # =================================================

            print("Getting visualization image 1...")

            image1 = Image.get_frame(
                img=self.visualization_input_1,
                redis_db=self.redis_db
            )

            print("Getting visualization image 2...")

            image2 = Image.get_frame(
                img=self.visualization_input_2,
                redis_db=self.redis_db
            )

            print("Images received.")

            # =================================================
            # GET SIFT OUTPUTS
            # =================================================

            print("Extracting SIFT output 1...")

            (
                keypoints1_dicts,
                descriptors1
            ) = self._extract_keypoints_and_descriptors(
                self.sift_output_1
            )

            print(
                "SIFT 1 descriptors:",
                len(descriptors1)
            )

            print("Extracting SIFT output 2...")

            (
                keypoints2_dicts,
                descriptors2
            ) = self._extract_keypoints_and_descriptors(
                self.sift_output_2
            )

            print(
                "SIFT 2 descriptors:",
                len(descriptors2)
            )

            # =================================================
            # CHECK DESCRIPTORS
            # =================================================

            if (
                len(descriptors1) < 2
                or len(descriptors2) < 2
            ):

                print(
                    "Not enough descriptors."
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

                image1.value = image1.value

                self.visualization_image = Image.set_frame(
                    img=image1,
                    package_uID=self.uID,
                    redis_db=self.redis_db
                )

                return build_response_sift_comparison(
                    context=self
                )

            # =================================================
            # SELECT MATCHER
            # =================================================

            print(
                "Matcher:",
                self.matcher
            )

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
            # KNN MATCHING
            # =================================================

            print("Running KNN matching...")

            matches = matcher.knnMatch(
                descriptors1,
                descriptors2,
                k=2
            )

            print(
                "Raw matches:",
                len(matches)
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

            print(
                "Good matches:",
                len(good_matches)
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

            print(
                "Images match:",
                images_match
            )

            # =================================================
            # NOVAVISION KEYPOINTS
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
                    p1=match[0].queryIdx,
                    p2=(
                        match[0].trainIdx
                        +
                        offset
                    )
                )

                for match in good_matches
            ]

            # =================================================
            # DETECTION OUTPUT
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

            print(
                "Detection output created."
            )

            # =================================================
            # VISUALIZATION
            # =================================================

            print(
                "Creating visualization..."
            )

            visualization = self._create_visualization(

                image1.value,
                image2.value,

                keypoints1_dicts,
                keypoints2_dicts,

                good_matches
            )

            print(
                "Visualization created."
            )

            # =================================================
            # SAVE VISUALIZATION
            # =================================================

            image1.value = visualization

            print(
                "Saving visualization frame..."
            )

            self.visualization_image = Image.set_frame(
                img=image1,
                package_uID=self.uID,
                redis_db=self.redis_db
            )

            print(
                "Visualization frame saved."
            )

            # =================================================
            # RESPONSE
            # =================================================

            print(
                "Building response..."
            )

            response = build_response_sift_comparison(
                context=self
            )

            print(
                "=== SIFT COMPARISON SUCCESS ==="
            )

            return response

        except Exception as e:

            print(
                "========================================"
            )
            print(
                "SIFT COMPARISON ERROR"
            )
            print(
                "ERROR TYPE:",
                type(e).__name__
            )
            print(
                "ERROR:",
                str(e)
            )
            print(
                "========================================"
            )

            raise


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    Executor(
        sys.argv[1]
    ).run()