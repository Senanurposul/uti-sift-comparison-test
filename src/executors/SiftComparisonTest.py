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

        self.good_matches_threshold = self.request.get_param(
            "GoodMatchesThreshold"
        )

        self.ratio_threshold = self.request.get_param(
            "RatioThreshold"
        )

        self.matcher = self.request.get_param(
            "Matcher"
        )

        # ====================================================
        # SIFT OUTPUT INPUTS
        # ====================================================

        self.sift_output_1 = self.request.get_param(
            "InputSIFTOutput1"
        )

        self.sift_output_2 = self.request.get_param(
            "InputSIFTOutput2"
        )

        # ====================================================
        # VISUALIZATION INPUTS
        # ====================================================

        self.visualization_input_1 = self.request.get_param(
            "InputVisualization1"
        )

        self.visualization_input_2 = self.request.get_param(
            "InputVisualization2"
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
    # EXTRACT SIFT KEYPOINTS + DESCRIPTORS
    # ========================================================

    def _extract_keypoints_and_descriptors(
        self,
        sift_output
    ):

        keypoints_dicts = []
        descriptors = []

        if isinstance(sift_output, str):
            sift_output = json.loads(sift_output)

        if isinstance(sift_output, dict):

            if "value" in sift_output:
                sift_output = sift_output["value"]

            elif "detections" in sift_output:
                sift_output = sift_output["detections"]

            elif "outputDetections" in sift_output:
                sift_output = sift_output["outputDetections"]

        if sift_output is None:
            return (
                [],
                np.empty(
                    (0, 128),
                    dtype=np.float32
                )
            )

        for detection in sift_output:

            if not isinstance(detection, dict):
                continue

            if "value" in detection:
                detection = detection["value"]

            keypoints = detection.get(
                "keyPoints",
                []
            )

            for kp in keypoints:

                descriptor = kp.get(
                    "descriptor"
                )

                if descriptor is None:
                    continue

                keypoints_dicts.append(
                    {
                        "pt": (
                            float(kp["cx"]),
                            float(kp["cy"])
                        ),
                        "size": max(
                            float(kp.get("size", 1.0)),
                            1.0
                        ),
                        "angle": float(
                            kp.get("angle", -1.0)
                        ),
                        "response": float(
                            kp.get("response", 0.0)
                        ),
                        "octave": int(
                            kp.get("octave", 0)
                        )
                    }
                )

                descriptors.append(
                    descriptor
                )

        if len(descriptors) == 0:

            descriptors_array = np.empty(
                (0, 128),
                dtype=np.float32
            )

        else:

            descriptors_array = np.asarray(
                descriptors,
                dtype=np.float32
            )

        return (
            keypoints_dicts,
            descriptors_array
        )

    # ========================================================
    # CREATE VISUALIZATION
    # ========================================================

    def _create_visualization(
        self,
        frame1,
        frame2,
        keypoints1_dicts,
        keypoints2_dicts,
        good_match_objects
    ):

        if frame1 is None:
            raise ValueError(
                "Visualization frame 1 is None"
            )

        if frame2 is None:
            raise ValueError(
                "Visualization frame 2 is None"
            )

        frame1 = np.asarray(frame1)
        frame2 = np.asarray(frame2)

        if frame1.size == 0:
            raise ValueError(
                "Visualization frame 1 is empty"
            )

        if frame2.size == 0:
            raise ValueError(
                "Visualization frame 2 is empty"
            )

        # ====================================================
        # UINT8
        # ====================================================

        if frame1.dtype != np.uint8:
            frame1 = frame1.astype(np.uint8)

        if frame2.dtype != np.uint8:
            frame2 = frame2.astype(np.uint8)

        # ====================================================
        # GRAYSCALE -> BGR
        # ====================================================

        if len(frame1.shape) == 2:

            frame1 = cv2.cvtColor(
                frame1,
                cv2.COLOR_GRAY2BGR
            )

        if len(frame2.shape) == 2:

            frame2 = cv2.cvtColor(
                frame2,
                cv2.COLOR_GRAY2BGR
            )

        # ====================================================
        # OPENCV KEYPOINTS
        # ====================================================

        cv_keypoints1 = []

        for kp in keypoints1_dicts:

            cv_keypoints1.append(
                cv2.KeyPoint(
                    float(kp["pt"][0]),
                    float(kp["pt"][1]),
                    max(
                        float(kp["size"]),
                        1.0
                    )
                )
            )

        cv_keypoints2 = []

        for kp in keypoints2_dicts:

            cv_keypoints2.append(
                cv2.KeyPoint(
                    float(kp["pt"][0]),
                    float(kp["pt"][1]),
                    max(
                        float(kp["size"]),
                        1.0
                    )
                )
            )

        # ====================================================
        # SADECE EN İYİ 50 MATCH
        # ====================================================

        visualization_matches = sorted(
            good_match_objects,
            key=lambda match: match.distance
        )[:50]

        print(
            "SIFTCOMPARISONTEST - TOTAL GOOD MATCHES:",
            len(good_match_objects),
            flush=True
        )

        print(
            "SIFTCOMPARISONTEST - VISUALIZATION MATCHES:",
            len(visualization_matches),
            flush=True
        )

        # ====================================================
        # DRAW MATCHES
        # ====================================================

        visualization = cv2.drawMatches(
            frame1,
            cv_keypoints1,
            frame2,
            cv_keypoints2,
            visualization_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        print(
            "SIFTCOMPARISONTEST - VISUALIZATION CREATED:",
            visualization.shape,
            flush=True
        )

        return visualization

    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        try:

            print(
                "SIFTCOMPARISONTEST - RUN START",
                flush=True
            )

            # =================================================
            # SIFT OUTPUT 1
            # =================================================

            (
                keypoints1_dicts,
                descriptors1
            ) = self._extract_keypoints_and_descriptors(
                self.sift_output_1
            )

            # =================================================
            # SIFT OUTPUT 2
            # =================================================

            (
                keypoints2_dicts,
                descriptors2
            ) = self._extract_keypoints_and_descriptors(
                self.sift_output_2
            )

            print(
                "SIFTCOMPARISONTEST - KEYPOINTS:",
                len(keypoints1_dicts),
                len(keypoints2_dicts),
                flush=True
            )

            # =================================================
            # YETERLİ DESCRIPTOR YOKSA
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

                return build_response_sift_comparison_test(
                    context=self
                )

            # =================================================
            # MATCHER
            # =================================================

            if self.matcher == "BFMatcher":

                matcher = cv2.BFMatcher(
                    cv2.NORM_L2
                )

                print(
                    "SIFTCOMPARISONTEST - BF MATCHER",
                    flush=True
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

                print(
                    "SIFTCOMPARISONTEST - FLANN MATCHER",
                    flush=True
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

            good_match_objects = []

            for pair in matches:

                if len(pair) < 2:
                    continue

                m, n = pair

                if (
                    m.distance
                    <
                    float(self.ratio_threshold)
                    * n.distance
                ):

                    good_match_objects.append(
                        m
                    )

            # =================================================
            # MATCH COUNT
            # =================================================

            good_matches_count = len(
                good_match_objects
            )

            print(
                "SIFTCOMPARISONTEST - GOOD MATCHES:",
                good_matches_count,
                flush=True
            )

            images_match = (
                good_matches_count
                >= int(
                    self.good_matches_threshold
                )
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

                for kp
                in all_keypoints_dicts
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

                for match
                in good_match_objects
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

            if image1_frame is None:
                raise ValueError(
                    "Image.get_frame returned None for image 1"
                )

            if image2_frame is None:
                raise ValueError(
                    "Image.get_frame returned None for image 2"
                )

            if image1_frame.value is None:
                raise ValueError(
                    "Image 1 value is None"
                )

            if image2_frame.value is None:
                raise ValueError(
                    "Image 2 value is None"
                )

            frame1 = np.asarray(
                image1_frame.value
            )

            frame2 = np.asarray(
                image2_frame.value
            )

            print(
                "SIFTCOMPARISONTEST - FRAME1 SHAPE:",
                frame1.shape,
                flush=True
            )

            print(
                "SIFTCOMPARISONTEST - FRAME2 SHAPE:",
                frame2.shape,
                flush=True
            )

            # =================================================
            # CREATE VISUALIZATION
            # =================================================

            visualization = self._create_visualization(
                frame1,
                frame2,
                keypoints1_dicts,
                keypoints2_dicts,
                good_match_objects
            )

            # =================================================
            # SAVE VISUALIZATION
            # =================================================

            image1_frame.value = visualization

            self.output_visualization = Image.set_frame(
                img=image1_frame,
                package_uID=self.uID,
                redis_db=self.redis_db
            )

            print(
                "SIFTCOMPARISONTEST - VISUALIZATION SAVED",
                flush=True
            )

        except Exception as e:

            print(
                "SIFTCOMPARISONTEST - ERROR:",
                repr(e),
                flush=True
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

        return build_response_sift_comparison_test(
            context=self
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    Executor(
        sys.argv[1]
    ).run()