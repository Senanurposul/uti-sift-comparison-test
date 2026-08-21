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

        self.good_matches_threshold = self.request.get_param(
            "GoodMatchesThreshold"
        )

        self.ratio_threshold = self.request.get_param(
            "RatioThreshold"
        )

        self.matcher = self.request.get_param(
            "Matcher"
        )

        # Only a True / False switch.
        self.visualize = self.request.get_param(
            "Visualize"
        )

        self.visualization_input_1 = self.request.get_param(
            "InputVisualization1"
        )

        self.visualization_input_2 = self.request.get_param(
            "InputVisualization2"
        )

        self.sift_output_1 = self.request.get_param(
            "InputSIFTOutput1"
        )

        self.sift_output_2 = self.request.get_param(
            "InputSIFTOutput2"
        )

        self.output_visualization_1 = None
        self.output_visualization_2 = None
        self.output_visualization_matches = None

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _visualization_enabled(self):
        value = self.visualize

        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            return value.strip().lower() in (
                "true",
                "1",
                "yes",
                "enabled"
            )

        nested_value = getattr(value, "value", None)

        if isinstance(nested_value, bool):
            return nested_value

        if isinstance(nested_value, str):
            return nested_value.strip().lower() in (
                "true",
                "1",
                "yes",
                "enabled"
            )

        if isinstance(value, dict):
            nested_value = value.get("value")

            if isinstance(nested_value, bool):
                return nested_value

            if isinstance(nested_value, str):
                return nested_value.strip().lower() in (
                    "true",
                    "1",
                    "yes",
                    "enabled"
                )

        return False

    def _extract_keypoints_and_descriptors(self, sift_output):

        keypoints_dicts = []
        descriptors = []

        if isinstance(sift_output, str):
            sift_output = json.loads(sift_output)

        if not isinstance(sift_output, list):
            raise ValueError(
                "SIFT output must be a list."
            )

        for detection in sift_output:

            for kp in detection.get("keyPoints", []):

                if "descriptor" not in kp:
                    continue

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

        if not descriptors:
            return keypoints_dicts, np.empty(
                (0, 128),
                dtype=np.float32
            )

        descriptors = np.asarray(
            descriptors,
            dtype=np.float32
        )

        return keypoints_dicts, descriptors

    def _create_keypoint_visualization(
        self,
        frame,
        keypoints
    ):
        frame = np.asarray(frame)

        if frame.size == 0:
            raise ValueError(
                "Visualization image is empty."
            )

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        cv_keypoints = [
            cv2.KeyPoint(
                float(point["pt"][0]),
                float(point["pt"][1]),
                1.0
            )
            for point in keypoints
        ]

        return cv2.drawKeypoints(
            frame,
            cv_keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

    def _create_matches_visualization(
        self,
        frame1,
        frame2,
        keypoints1,
        keypoints2,
        good_matches
    ):
        frame1 = np.asarray(frame1)
        frame2 = np.asarray(frame2)

        if frame1.size == 0 or frame2.size == 0:
            raise ValueError(
                "Visualization image is empty."
            )

        if frame1.dtype != np.uint8:
            frame1 = frame1.astype(np.uint8)

        if frame2.dtype != np.uint8:
            frame2 = frame2.astype(np.uint8)

        cv_keypoints1 = [
            cv2.KeyPoint(
                float(point["pt"][0]),
                float(point["pt"][1]),
                1.0
            )
            for point in keypoints1
        ]

        cv_keypoints2 = [
            cv2.KeyPoint(
                float(point["pt"][0]),
                float(point["pt"][1]),
                1.0
            )
            for point in keypoints2
        ]

        # No custom match-count limit:
        # every good match is drawn.
        return cv2.drawMatches(
            frame1,
            cv_keypoints1,
            frame2,
            cv_keypoints2,
            list(good_matches),
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

    def _save_image(self, frame, image):
        frame.value = image

        return Image.set_frame(
            img=frame,
            package_uID=self.uID,
            redis_db=self.redis_db
        )

    def run(self):

        try:

            # ------------------------------------------------
            # Existing SIFT extraction / matching
            # ------------------------------------------------

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

            if (
                len(descriptors1) < 2
                or len(descriptors2) < 2
            ):
                self.output_detections = []
                return build_response_sift_comparison_test(
                    context=self
                )

            if self.matcher == "BFMatcher":

                matcher = cv2.BFMatcher(
                    cv2.NORM_L2,
                    crossCheck=False
                )

            elif self.matcher == "FlannBasedMatcher":

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

            else:
                raise ValueError(
                    f"Unsupported matcher: {self.matcher}"
                )

            matches = matcher.knnMatch(
                descriptors1,
                descriptors2,
                k=2
            )

            good_matches = []

            for pair in matches:

                if len(pair) < 2:
                    continue

                m, n = pair

                if (
                    m.distance
                    < self.ratio_threshold * n.distance
                ):
                    good_matches.append(m)

            # ------------------------------------------------
            # Existing detection output
            # ------------------------------------------------

            good_match_count = len(good_matches)

            is_match = (
                good_match_count
                >= self.good_matches_threshold
            )

            self.output_detections = [
                {
                    "boundingBox": None,
                    "confidence": 1.0,
                    "classLabel": (
                        "Match"
                        if is_match
                        else "NoMatch"
                    ),
                    "classId": (
                        1
                        if is_match
                        else 0
                    ),
                    "keyPoints": [],
                    "connections": [],
                }
            ]

            # ------------------------------------------------
            # Visualization
            # ------------------------------------------------
            # True:
            #   1) image 1 + its keypoints
            #   2) image 2 + its keypoints
            #   3) two images side by side + all good matches
            #
            # False:
            #   no visualization output
            # ------------------------------------------------

            if (
                self._visualization_enabled()
                and self.visualization_input_1 is not None
                and self.visualization_input_2 is not None
            ):

                frame1 = Image.get_frame(
                    img=self.visualization_input_1,
                    redis_db=self.redis_db
                )

                frame2 = Image.get_frame(
                    img=self.visualization_input_2,
                    redis_db=self.redis_db
                )

                if frame1 is None or frame2 is None:
                    raise ValueError(
                        "Visualization images could not be loaded."
                    )

                visualization1 = (
                    self._create_keypoint_visualization(
                        frame1.value,
                        keypoints1_dicts
                    )
                )

                visualization2 = (
                    self._create_keypoint_visualization(
                        frame2.value,
                        keypoints2_dicts
                    )
                )

                visualization_matches = (
                    self._create_matches_visualization(
                        frame1.value,
                        frame2.value,
                        keypoints1_dicts,
                        keypoints2_dicts,
                        good_matches
                    )
                )

                self.output_visualization_1 = (
                    self._save_image(
                        frame1,
                        visualization1
                    )
                )

                self.output_visualization_2 = (
                    self._save_image(
                        frame2,
                        visualization2
                    )
                )

                self.output_visualization_matches = (
                    self._save_image(
                        frame1,
                        visualization_matches
                    )
                )

        except Exception as e:
            print(
                "SIFT Comparison Error:",
                repr(e)
            )
            raise

        return build_response_sift_comparison_test(
            context=self
        )


if __name__ == "__main__":
    Executor(
        sys.argv[1]
    ).run()