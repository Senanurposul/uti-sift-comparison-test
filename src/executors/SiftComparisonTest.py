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

from components.SiftComparison.src.models.PackageModel import PackageModel

from components.SiftComparison.src.utils.response import (
    build_response_sift_comparison
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
        # SIFT OUTPUTS
        # ====================================================

        self.sift_output_1 = self.request.get_param(
            "InputSIFTOutput1"
        )

        self.sift_output_2 = self.request.get_param(
            "InputSIFTOutput2"
        )

        # ====================================================
        # IMAGE INPUTS
        # ====================================================

        self.image_one = self.request.get_param(
            "InputImageOne"
        )

        self.image_two = self.request.get_param(
            "InputImageTwo"
        )

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

        if isinstance(sift_output, str):
            sift_output = json.loads(sift_output)

        if isinstance(sift_output, dict):
            sift_output = sift_output.get(
                "value",
                sift_output
            )

        if not isinstance(sift_output, list):
            return (
                keypoints_dicts,
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

                if "descriptor" not in kp:
                    continue

                keypoints_dicts.append({
                    "pt": (
                        float(kp["cx"]),
                        float(kp["cy"])
                    ),
                    "size": float(
                        kp.get("size", 1.0)
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
    # RUN
    # ========================================================

    def run(self):

        # ====================================================
        # IMAGE INPUTLARINI AL
        # ====================================================

        image_one = Image.get_frame(
            img=self.image_one,
            redis_db=self.redis_db
        )

        image_two = Image.get_frame(
            img=self.image_two,
            redis_db=self.redis_db
        )

        # ====================================================
        # SIFT OUTPUTLARINI AL
        # ====================================================

        (
            keypoints_one,
            descriptors_one
        ) = self._extract_keypoints_and_descriptors(
            self.sift_output_1
        )

        (
            keypoints_two,
            descriptors_two
        ) = self._extract_keypoints_and_descriptors(
            self.sift_output_2
        )

        # ====================================================
        # YETERSİZ DESCRIPTOR
        # ====================================================

        if (
            len(descriptors_one) < 2
            or len(descriptors_two) < 2
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

            return build_response_sift_comparison(
                context=self
            )

        # ====================================================
        # MATCHER
        # ====================================================

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

        # ====================================================
        # KNN MATCHING
        # ====================================================

        matches = matcher.knnMatch(
            descriptors_one,
            descriptors_two,
            k=2
        )

        # ====================================================
        # LOWE RATIO TEST
        # ====================================================

        good_matches = []

        for pair in matches:

            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < (
                self.ratio_threshold * n.distance
            ):
                good_matches.append([m])

        # ====================================================
        # MATCH COUNT
        # ====================================================

        good_matches_count = len(
            good_matches
        )

        images_match = (
            good_matches_count
            >= self.good_matches_threshold
        )

        # ====================================================
        # KEYPOINTS
        # ====================================================

        all_keypoints_dicts = (
            keypoints_one +
            keypoints_two
        )

        offset = len(
            keypoints_one
        )

        keypoints = [
            KeyPoints(
                cx=float(kp["pt"][0]),
                cy=float(kp["pt"][1]),
                confidence=1.0
            )
            for kp in all_keypoints_dicts
        ]

        # ====================================================
        # CONNECTIONS
        # ====================================================

        connections = [
            Connection(
                p1=m[0].queryIdx,
                p2=m[0].trainIdx + offset
            )
            for m in good_matches
        ]

        # ====================================================
        # OUTPUT
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
        # RESPONSE
        # ====================================================

        return build_response_sift_comparison(
            context=self
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    Executor(
        sys.argv[1]
    ).run()