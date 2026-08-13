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

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _extract_keypoints_and_descriptors(
        self,
        sift_output
    ):

        keypoints_dicts = []
        descriptors = []

        # ---------------------------------------------
        # Convert JSON string if necessary
        # ---------------------------------------------

        if isinstance(sift_output, str):
            sift_output = json.loads(
                sift_output
            )

        if not isinstance(
            sift_output,
            list
        ):
            raise ValueError(
                "SIFT output must be a list."
            )

        # ---------------------------------------------
        # Extract keypoints and descriptors
        # ---------------------------------------------

        for detection in sift_output:

            keypoints = detection.get(
                "keyPoints",
                []
            )

            for kp in keypoints:

                if "descriptor" not in kp:
                    raise ValueError(
                        "SIFT keypoint does not contain "
                        "'descriptor'."
                    )

                keypoints_dicts.append(
                    {
                        "pt": (
                            float(kp["cx"]),
                            float(kp["cy"])
                        )
                    }
                )

                descriptors.append(
                    kp["descriptor"]
                )

        # ---------------------------------------------
        # Empty descriptor check
        # ---------------------------------------------

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

    def run(self):

        try:

            # =============================================
            # STEP 1 - Extract SIFT outputs
            # =============================================

            print(
                "SIFT Comparison: extracting input 1"
            )

            (
                keypoints1_dicts,
                descriptors1
            ) = self._extract_keypoints_and_descriptors(
                self.sift_output_1
            )

            print(
                "SIFT Comparison: extracting input 2"
            )

            (
                keypoints2_dicts,
                descriptors2
            ) = self._extract_keypoints_and_descriptors(
                self.sift_output_2
            )

            # =============================================
            # STEP 2 - Print descriptor information
            # =============================================

            print(
                "Descriptor 1 shape:",
                descriptors1.shape
            )

            print(
                "Descriptor 2 shape:",
                descriptors2.shape
            )

            print(
                "Keypoints 1:",
                len(keypoints1_dicts)
            )

            print(
                "Keypoints 2:",
                len(keypoints2_dicts)
            )

            print(
                "Matcher:",
                self.matcher
            )

            print(
                "Ratio threshold:",
                self.ratio_threshold
            )

            print(
                "Good matches threshold:",
                self.good_matches_threshold
            )

            # =============================================
            # STEP 3 - Check descriptor count
            # =============================================

            if (
                len(descriptors1) < 2
                or len(descriptors2) < 2
            ):

                print(
                    "Not enough descriptors."
                )

                self.output_detections = (
                    self._no_match_result()
                )

                return (
                    build_response_sift_comparison_test(
                        context=self
                    )
                )

            # =============================================
            # STEP 4 - Create matcher
            # =============================================

            print(
                "Creating matcher..."
            )

            if self.matcher == "BFMatcher":

                matcher = cv2.BFMatcher(
                    cv2.NORM_L2,
                    crossCheck=False
                )

            else:

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

            print(
                "Matcher created."
            )

            # =============================================
            # STEP 5 - KNN matching
            # =============================================

            print(
                "MATCHING STARTED"
            )

            matches = matcher.knnMatch(
                descriptors1,
                descriptors2,
                k=2
            )

            print(
                "MATCHING FINISHED"
            )

            print(
                "Total match groups:",
                len(matches)
            )

            # =============================================
            # STEP 6 - Lowe Ratio Test
            # =============================================

            print(
                "Ratio test started."
            )

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

            print(
                "Ratio test finished."
            )

            # =============================================
            # STEP 7 - Calculate result
            # =============================================

            good_matches_count = len(
                good_matches
            )

            images_match = (
                good_matches_count
                >= self.good_matches_threshold
            )

            print(
                "Good matches:",
                good_matches_count
            )

            print(
                "Images match:",
                images_match
            )

            # =============================================
            # STEP 8 - Combine keypoints
            # =============================================

            all_keypoints_dicts = (
                keypoints1_dicts
                + keypoints2_dicts
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

            # =============================================
            # STEP 9 - Create connections
            # =============================================

            connections = [
                Connection(
                    p1=m.queryIdx,
                    p2=m.trainIdx + offset
                )
                for m in good_matches
            ]

            print(
                "Connections:",
                len(connections)
            )

            # =============================================
            # STEP 10 - Create output
            # =============================================

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
                "SIFT Comparison completed."
            )

        except Exception as e:

            print(
                "================================="
            )

            print(
                "SIFT COMPARISON ERROR:"
            )

            print(
                repr(e)
            )

            print(
                "================================="
            )

            raise

        return (
            build_response_sift_comparison_test(
                context=self
            )
        )


if __name__ == "__main__":
    Executor(
        sys.argv[1]
    ).run()