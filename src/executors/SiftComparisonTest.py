import os
import sys
import json
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.base.model import KeyPoints, Detection, Connection
from sdks.novavision.src.helper.executor import Executor

from components.SiftComparisonTest.src.utils.response import (
    build_response_sift_comparison_test
)

from components.SiftComparisonTest.src.models.PackageModel import PackageModel


class SiftComparisonTest(Component):

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)

        self.request.model = PackageModel(**(self.request.data))

        # =====================================================
        # CONFIGS
        # =====================================================

        self.good_matches_threshold = self.request.get_param(
            "GoodMatchesThreshold"
        )

        self.ratio_threshold = self.request.get_param(
            "RatioThreshold"
        )

        self.matcher = self.request.get_param(
            "Matcher"
        )

        # =====================================================
        # SIFT INPUTS
        # =====================================================

        self.sift_output_1 = self.request.get_param(
            "InputSIFTOutput1"
        )

        self.sift_output_2 = self.request.get_param(
            "InputSIFTOutput2"
        )

        # =====================================================
        # VISUALIZATION INPUTS
        # =====================================================

        self.visualization_input_1 = self.request.get_param(
            "InputVisualization1"
        )

        self.visualization_input_2 = self.request.get_param(
            "InputVisualization2"
        )

        # =====================================================
        # OUTPUTS
        # =====================================================

        self.output_detections = []
        self.output_visualization = None

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    # =========================================================
    # SIFT OUTPUT PARSE
    # =========================================================

    def _extract_keypoints_and_descriptors(self, sift_output):

        keypoints_dicts = []
        descriptors = []

        if isinstance(sift_output, str):
            sift_output = json.loads(sift_output)

        for detection in sift_output:

            for kp in detection.get("keyPoints", []):

                keypoints_dicts.append({
                    "pt": (
                        kp["cx"],
                        kp["cy"]
                    ),
                    "size": kp["size"],
                    "angle": kp["angle"],
                    "response": kp["response"],
                    "octave": kp["octave"],
                })

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

    # =========================================================
    # RUN
    # =========================================================

    def run(self):

        try:

            print(
                "SIFTCOMPARISONTEST - RUN START",
                flush=True
            )

            # =================================================
            # GET SIFT OUTPUTS
            # =================================================

            keypoints1_dicts, descriptors1 = (
                self._extract_keypoints_and_descriptors(
                    self.sift_output_1
                )
            )

            keypoints2_dicts, descriptors2 = (
                self._extract_keypoints_and_descriptors(
                    self.sift_output_2
                )
            )

            print(
                "SIFTCOMPARISONTEST - DESCRIPTORS:",
                len(descriptors1),
                len(descriptors2),
                flush=True
            )

            # =================================================
            # NOT ENOUGH DESCRIPTORS
            # =================================================

            if len(descriptors1) < 2 or len(descriptors2) < 2:

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

                # Visualization test
                self._create_test_visualization()

                return build_response_sift_comparison_test(
                    context=self
                )

            # =================================================
            # MATCHER
            # =================================================

            if self.matcher == "BFMatcher":

                print(
                    "SIFTCOMPARISONTEST - USING BF MATCHER",
                    flush=True
                )

                matcher = cv2.BFMatcher(
                    cv2.NORM_L2
                )

            else:

                print(
                    "SIFTCOMPARISONTEST - USING FLANN MATCHER",
                    flush=True
                )

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
                    good_matches.append([m])

            # =================================================
            # MATCH RESULT
            # =================================================

            good_matches_count = len(
                good_matches
            )

            images_match = (
                good_matches_count
                >=
                self.good_matches_threshold
            )

            print(
                "SIFTCOMPARISONTEST - GOOD MATCHES:",
                good_matches_count,
                flush=True
            )

            print(
                "SIFTCOMPARISONTEST - MATCH:",
                images_match,
                flush=True
            )

            # =================================================
            # KEYPOINTS
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
                    p2=m[0].trainIdx + offset
                )

                for m in good_matches
            ]

            # =================================================
            # OUTPUT DETECTIONS
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
                "SIFTCOMPARISONTEST - DETECTIONS CREATED",
                flush=True
            )

            # =================================================
            # VISUALIZATION TEST
            # =================================================

            self._create_test_visualization()

            # =================================================
            # RESPONSE
            # =================================================

            print(
                "SIFTCOMPARISONTEST - BUILD RESPONSE",
                flush=True
            )

            return build_response_sift_comparison_test(
                context=self
            )

        except Exception as e:

            print(
                "SIFTCOMPARISONTEST - ERROR:",
                repr(e),
                flush=True
            )

            # =================================================
            # FALLBACK DETECTION
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

            return build_response_sift_comparison_test(
                context=self
            )

    # =========================================================
    # VISUALIZATION TEST
    # =========================================================

    def _create_test_visualization(self):

        print(
            "SIFTCOMPARISONTEST - VISUALIZATION TEST START",
            flush=True
        )

        try:

            # -------------------------------------------------
            # CHECK INPUT
            # -------------------------------------------------

            print(
                "SIFTCOMPARISONTEST - visualization_input_1:",
                self.visualization_input_1,
                flush=True
            )

            if self.visualization_input_1 is None:

                print(
                    "SIFTCOMPARISONTEST - VISUALIZATION INPUT 1 IS NONE",
                    flush=True
                )

                self.output_visualization = None
                return

            # -------------------------------------------------
            # GET IMAGE FROM REDIS
            # -------------------------------------------------

            image1 = Image.get_frame(
                img=self.visualization_input_1,
                redis_db=self.redis_db
            )

            print(
                "SIFTCOMPARISONTEST - image1:",
                image1,
                flush=True
            )

            # -------------------------------------------------
            # IMAGE CHECK
            # -------------------------------------------------

            if image1 is None:

                print(
                    "SIFTCOMPARISONTEST - IMAGE1 IS NONE",
                    flush=True
                )

                self.output_visualization = None
                return

            if image1.value is None:

                print(
                    "SIFTCOMPARISONTEST - IMAGE1.VALUE IS NONE",
                    flush=True
                )

                self.output_visualization = None
                return

            print(
                "SIFTCOMPARISONTEST - IMAGE1 VALUE OK",
                flush=True
            )

            print(
                "SIFTCOMPARISONTEST - IMAGE SHAPE:",
                np.asarray(image1.value).shape,
                flush=True
            )

            # -------------------------------------------------
            # COPY IMAGE
            # -------------------------------------------------

            image1.value = np.asarray(
                image1.value
            ).copy()

            # -------------------------------------------------
            # SAVE AS OUTPUT IMAGE
            # -------------------------------------------------

            self.output_visualization = Image.set_frame(
                img=image1,
                package_uID=self.uID,
                redis_db=self.redis_db
            )

            print(
                "SIFTCOMPARISONTEST - OUTPUT VISUALIZATION:",
                self.output_visualization,
                flush=True
            )

            print(
                "SIFTCOMPARISONTEST - VISUALIZATION TEST SUCCESS",
                flush=True
            )

        except Exception as e:

            print(
                "SIFTCOMPARISONTEST - VISUALIZATION ERROR:",
                repr(e),
                flush=True
            )

            self.output_visualization = None


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    Executor(
        sys.argv[1]
    ).run()