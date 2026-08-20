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

        # =====================================================
        # CONFIGS
        # =====================================================

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

        # =====================================================
        # SIFT INPUTS
        # =====================================================

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

        # =====================================================
        # VISUALIZATION INPUTS
        # =====================================================

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

        # =====================================================
        # OUTPUTS
        # =====================================================

        self.output_detections = []

        self.output_visualization = None

    # =========================================================
    # BOOTSTRAP
    # =========================================================

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    # =========================================================
    # EXTRACT SIFT KEYPOINTS + DESCRIPTORS
    # =========================================================

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

        if sift_output is None:
            return (
                [],
                np.empty(
                    (0, 128),
                    dtype=np.float32
                )
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
                        ),
                    }
                )

                descriptor = kp.get(
                    "descriptor"
                )

                if descriptor is not None:
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

    # =========================================================
    # CREATE VISUALIZATION
    # =========================================================

    def _create_visualization(
        self,
        keypoints1_dicts,
        keypoints2_dicts,
        good_matches
    ):

        print(
            "SIFTCOMPARISONTEST - DRAW MATCHES START",
            flush=True
        )

        try:

            # =================================================
            # GET IMAGE 1
            # =================================================

            image1 = Image.get_frame(
                img=self.visualization_input_1,
                redis_db=self.redis_db
            )

            # =================================================
            # GET IMAGE 2
            # =================================================

            image2 = Image.get_frame(
                img=self.visualization_input_2,
                redis_db=self.redis_db
            )

            print(
                "SIFTCOMPARISONTEST - IMAGE1:",
                image1,
                flush=True
            )

            print(
                "SIFTCOMPARISONTEST - IMAGE2:",
                image2,
                flush=True
            )

            # =================================================
            # CHECK IMAGES
            # =================================================

            if image1 is None:
                print(
                    "SIFTCOMPARISONTEST - IMAGE1 IS NONE",
                    flush=True
                )

                self.output_visualization = None
                return

            if image2 is None:
                print(
                    "SIFTCOMPARISONTEST - IMAGE2 IS NONE",
                    flush=True
                )

                self.output_visualization = None
                return

            if image1.value is None:
                print(
                    "SIFTCOMPARISONTEST - IMAGE1 VALUE IS NONE",
                    flush=True
                )

                self.output_visualization = None
                return

            if image2.value is None:
                print(
                    "SIFTCOMPARISONTEST - IMAGE2 VALUE IS NONE",
                    flush=True
                )

                self.output_visualization = None
                return

            # =================================================
            # CONVERT TO NUMPY
            # =================================================

            frame1 = np.asarray(
                image1.value
            ).copy()

            frame2 = np.asarray(
                image2.value
            ).copy()

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
            # ENSURE 3 CHANNEL IMAGE
            # =================================================

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

            # =================================================
            # CREATE OPENCV KEYPOINTS
            # =================================================

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
                        size=float(
                            kp["size"]
                        ),
                        angle=float(
                            kp["angle"]
                        ),
                        response=float(
                            kp["response"]
                        ),
                        octave=int(
                            kp["octave"]
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
                        size=float(
                            kp["size"]
                        ),
                        angle=float(
                            kp["angle"]
                        ),
                        response=float(
                            kp["response"]
                        ),
                        octave=int(
                            kp["octave"]
                        )
                    )
                )

            print(
                "SIFTCOMPARISONTEST - KEYPOINTS:",
                len(cv_keypoints1),
                len(cv_keypoints2),
                flush=True
            )

            # =================================================
            # CONVERT MATCHES
            # =================================================

            cv_matches = []

            for match in good_matches:

                if isinstance(
                    match,
                    list
                ):

                    if len(match) > 0:
                        cv_matches.append(
                            match[0]
                        )

                else:

                    cv_matches.append(
                        match
                    )

            print(
                "SIFTCOMPARISONTEST - GOOD MATCHES:",
                len(cv_matches),
                flush=True
            )

            # =================================================
            # DRAW MATCHES
            # =================================================

            visualization = cv2.drawMatches(
                frame1,
                cv_keypoints1,
                frame2,
                cv_keypoints2,
                cv_matches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            print(
                "SIFTCOMPARISONTEST - DRAW MATCHES SUCCESS",
                flush=True
            )

            print(
                "SIFTCOMPARISONTEST - VISUALIZATION SHAPE:",
                visualization.shape,
                flush=True
            )

            # =================================================
            # PUT VISUALIZATION INTO NOVAVISION IMAGE
            # =================================================

            image1.value = visualization

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

        except Exception as e:

            print(
                "SIFTCOMPARISONTEST - VISUALIZATION ERROR:",
                repr(e),
                flush=True
            )

            self.output_visualization = None

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
            # EXTRACT SIFT 1
            # =================================================

            (
                keypoints1_dicts,
                descriptors1
            ) = self._extract_keypoints_and_descriptors(
                self.sift_output_1
            )

            # =================================================
            # EXTRACT SIFT 2
            # =================================================

            (
                keypoints2_dicts,
                descriptors2
            ) = self._extract_keypoints_and_descriptors(
                self.sift_output_2
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

            if (
                len(descriptors1) < 2
                or
                len(descriptors2) < 2
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

                self._create_visualization(
                    keypoints1_dicts,
                    keypoints2_dicts,
                    []
                )

                return build_response_sift_comparison_test(
                    context=self
                )

            # =================================================
            # SELECT MATCHER
            # =================================================

            if self.matcher == "BFMatcher":

                print(
                    "SIFTCOMPARISONTEST - BF MATCHER",
                    flush=True
                )

                matcher = cv2.BFMatcher(
                    cv2.NORM_L2
                )

            else:

                print(
                    "SIFTCOMPARISONTEST - FLANN MATCHER",
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

                    good_matches.append(
                        [m]
                    )

            # =================================================
            # MATCH COUNT
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
                "SIFTCOMPARISONTEST - GOOD MATCHES COUNT:",
                good_matches_count,
                flush=True
            )

            print(
                "SIFTCOMPARISONTEST - IMAGES MATCH:",
                images_match,
                flush=True
            )

            # =================================================
            # COMBINE KEYPOINTS
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
            # CREATE VISUALIZATION
            # =================================================

            self._create_visualization(
                keypoints1_dicts,
                keypoints2_dicts,
                good_matches
            )

            # =================================================
            # BUILD RESPONSE
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


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    Executor(
        sys.argv[1]
    ).run()