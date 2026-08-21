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

        # ------------------------------------------------
        # Existing configs
        # ------------------------------------------------

        self.good_matches_threshold = self.request.get_param(
            "GoodMatchesThreshold"
        )

        self.ratio_threshold = self.request.get_param(
            "RatioThreshold"
        )

        self.matcher = self.request.get_param(
            "Matcher"
        )

        # Roboflow v2:
        # visualize = False by default
        self.visualize = self.request.get_param(
            "Visualize"
        )

        # ------------------------------------------------
        # Existing inputs
        # ------------------------------------------------

        self.sift_output_1 = self.request.get_param(
            "InputSIFTOutput1"
        )

        self.sift_output_2 = self.request.get_param(
            "InputSIFTOutput2"
        )

        self.visualization_input_1 = self.request.get_param(
            "InputVisualization1"
        )

        self.visualization_input_2 = self.request.get_param(
            "InputVisualization2"
        )

        # ------------------------------------------------
        # Outputs
        # ------------------------------------------------

        self.output_detections = []

        self.output_visualization_1 = None
        self.output_visualization_2 = None
        self.output_visualization_matches = None

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    # ====================================================
    # VISUALIZE
    # ====================================================

    def _visualization_enabled(self):

        value = self.visualize

        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            return value.strip().lower() in (
                "true",
                "1",
                "yes"
            )

        nested_value = getattr(
            value,
            "value",
            None
        )

        if isinstance(nested_value, bool):
            return nested_value

        if isinstance(nested_value, str):
            return nested_value.strip().lower() in (
                "true",
                "1",
                "yes"
            )

        return False

    # ====================================================
    # SIFT OUTPUT PARSING
    # ====================================================

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

            if "value" in sift_output:
                sift_output = sift_output["value"]

            elif "detections" in sift_output:
                sift_output = sift_output["detections"]

        if not isinstance(sift_output, list):
            raise ValueError(
                "SIFT output must be a list."
            )

        for detection in sift_output:

            for kp in detection.get(
                "keyPoints",
                []
            ):

                if "descriptor" not in kp:
                    continue

                keypoints_dicts.append(
                    {
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
                                -1
                            )
                        ),
                        "response": float(
                            kp.get(
                                "response",
                                0
                            )
                        ),
                        "octave": int(
                            kp.get(
                                "octave",
                                0
                            )
                        ),
                        "class_id": int(
                            kp.get(
                                "classId",
                                -1
                            )
                        )
                    }
                )

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

    # ====================================================
    # KEYPOINT VISUALIZATION
    # ====================================================

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
            frame = frame.astype(
                np.uint8
            )

        cv_keypoints = []

        for point in keypoints:

            cv_keypoints.append(
                cv2.KeyPoint(
                    float(point["pt"][0]),
                    float(point["pt"][1]),
                    float(
                        point.get(
                            "size",
                            1.0
                        )
                    ),
                    float(
                        point.get(
                            "angle",
                            -1
                        )
                    ),
                    float(
                        point.get(
                            "response",
                            0
                        )
                    ),
                    int(
                        point.get(
                            "octave",
                            0
                        )
                    ),
                    int(
                        point.get(
                            "class_id",
                            -1
                        )
                    )
                )
            )

        # Same basic OpenCV operation used by Roboflow v2.
        return cv2.drawKeypoints(
            frame,
            cv_keypoints,
            None
        )

    # ====================================================
    # MATCH VISUALIZATION
    # ====================================================

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

        cv_keypoints1 = [
            cv2.KeyPoint(
                float(point["pt"][0]),
                float(point["pt"][1]),
                float(
                    point.get(
                        "size",
                        1.0
                    )
                ),
                float(
                    point.get(
                        "angle",
                        -1
                    )
                ),
                float(
                    point.get(
                        "response",
                        0
                    )
                ),
                int(
                    point.get(
                        "octave",
                        0
                    )
                ),
                int(
                    point.get(
                        "class_id",
                        -1
                    )
                )
            )
            for point in keypoints1
        ]

        cv_keypoints2 = [
            cv2.KeyPoint(
                float(point["pt"][0]),
                float(point["pt"][1]),
                float(
                    point.get(
                        "size",
                        1.0
                    )
                ),
                float(
                    point.get(
                        "angle",
                        -1
                    )
                ),
                float(
                    point.get(
                        "response",
                        0
                    )
                ),
                int(
                    point.get(
                        "octave",
                        0
                    )
                ),
                int(
                    point.get(
                        "class_id",
                        -1
                    )
                )
            )
            for point in keypoints2
        ]

        # Roboflow does NOT apply a custom visualization
        # match-count limit.
        #
        # Every good match from Lowe's ratio test is drawn.
        good_matches_for_draw = [
            [match]
            for match in good_matches
        ]

        if self.matcher == "BFMatcher":

            draw_params = {
                "flags":
                    cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            }

        else:

            draw_params = {
                "matchColor": (0, 255, 0),
                "singlePointColor": (0, 0, 255),
                "flags":
                    cv2.DrawMatchesFlags_DEFAULT
            }

        return cv2.drawMatchesKnn(
            frame1,
            cv_keypoints1,
            frame2,
            cv_keypoints2,
            good_matches_for_draw,
            None,
            **draw_params
        )

    # ====================================================
    # SAVE IMAGE
    # ====================================================

    def _save_image(
        self,
        frame,
        image
    ):

        frame.value = image

        return Image.set_frame(
            img=frame,
            package_uID=self.uID,
            redis_db=self.redis_db
        )

    # ====================================================
    # RUN
    # ====================================================

    def run(self):

        # ------------------------------------------------
        # 1. Parse SIFT outputs
        # ------------------------------------------------

        (
            keypoints1_dicts,
            descriptors1
        ) = self._extract_keypoints_and_descriptors(
            self.sift_output_1
        )

        (
            keypoints2_dicts,
            descriptors2
        ) = self._extract_keypoints_and_descriptors(
            self.sift_output_2
        )

        # ------------------------------------------------
        # 2. Not enough descriptors
        # ------------------------------------------------

        if (
            len(descriptors1) < 2
            or len(descriptors2) < 2
        ):

            self.output_detections = []

            return build_response_sift_comparison_test(
                self
            )

        # ------------------------------------------------
        # 3. Matcher
        # ------------------------------------------------

        if self.matcher == "BFMatcher":

            matcher = cv2.BFMatcher(
                cv2.NORM_L2,
                crossCheck=False
            )

        elif self.matcher == "FlannBasedMatcher":

            matcher = cv2.FlannBasedMatcher(
                {
                    "algorithm": 1,
                    "trees": 5
                },
                {
                    "checks": 50
                }
            )

        else:

            raise ValueError(
                f"Unsupported matcher: {self.matcher}"
            )

        # ------------------------------------------------
        # 4. KNN matching
        # ------------------------------------------------

        matches = matcher.knnMatch(
            descriptors1,
            descriptors2,
            k=2
        )

        # ------------------------------------------------
        # 5. Lowe Ratio Test
        # ------------------------------------------------

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
        # 6. Result
        # ------------------------------------------------

        good_match_count = len(
            good_matches
        )

        is_match = (
            good_match_count
            >= self.good_matches_threshold
        )

        self.output_detections = [
            {
                "boundingBox": None,
                "confidence": float(
                    good_match_count
                ),
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
                "connections": []
            }
        ]

        # ------------------------------------------------
        # 7. VISUALIZATION
        # ------------------------------------------------
        #
        # Roboflow v2:
        #
        # visualize=False
        #   -> no visualization
        #
        # visualize=True + image inputs
        #   -> visualization_1
        #   -> visualization_2
        #   -> visualization_matches
        #
        # No visualization match-count parameter.
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

            # Separate frames so outputs do not overwrite
            # each other.

            frame1_kp = Image.get_frame(
                img=self.visualization_input_1,
                redis_db=self.redis_db
            )

            frame2_kp = Image.get_frame(
                img=self.visualization_input_2,
                redis_db=self.redis_db
            )

            frame_matches = Image.get_frame(
                img=self.visualization_input_1,
                redis_db=self.redis_db
            )

            # 7.1 First image keypoints

            visualization1 = (
                self._create_keypoint_visualization(
                    frame1.value,
                    keypoints1_dicts
                )
            )

            # 7.2 Second image keypoints

            visualization2 = (
                self._create_keypoint_visualization(
                    frame2.value,
                    keypoints2_dicts
                )
            )

            # 7.3 Match visualization

            visualization_matches = (
                self._create_matches_visualization(
                    frame1.value,
                    frame2.value,
                    keypoints1_dicts,
                    keypoints2_dicts,
                    good_matches
                )
            )

            # Save outputs independently.

            self.output_visualization_1 = (
                self._save_image(
                    frame1_kp,
                    visualization1
                )
            )

            self.output_visualization_2 = (
                self._save_image(
                    frame2_kp,
                    visualization2
                )
            )

            self.output_visualization_matches = (
                self._save_image(
                    frame_matches,
                    visualization_matches
                )
            )

        return build_response_sift_comparison_test(
            self
        )


if __name__ == "__main__":
    Executor(
        sys.argv[1]
    ).run()