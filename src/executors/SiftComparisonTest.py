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

from components.SiftComparisonTest.src.models.PackageModel import PackageModel


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

        self.visualize = self.request.get_param(
            "Visualize"
        )

        self.input_1 = self.request.get_param("Input1")
        self.input_2 = self.request.get_param("Input2")

        self.images_match = False
        self.good_matches_count = 0

        self.keypoints_1 = None
        self.descriptors_1 = None
        self.keypoints_2 = None
        self.descriptors_2 = None

        self.visualization_1 = None
        self.visualization_2 = None
        self.visualization_matches = None

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _is_image(self, value):
        return isinstance(value, Image)

    def _get_frame(self, value):
        frame = Image.get_frame(
            img=value,
            redis_db=self.redis_db
        )
        if frame is None:
            raise ValueError("Could not load input image.")
        return frame

    def _extract_from_image(self, image_input):
        frame = self._get_frame(image_input)
        image = np.asarray(frame.value)

        if image.size == 0:
            raise ValueError("Input image is empty.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp, descriptors = sift.detectAndCompute(gray, None)

        if descriptors is None:
            descriptors = np.empty(
                (0, 128),
                dtype=np.float32
            )

        keypoints = [
            {
                "cx": float(point.pt[0]),
                "cy": float(point.pt[1]),
                "size": float(point.size),
                "angle": float(point.angle),
                "response": float(point.response),
                "octave": int(point.octave),
                "classId": int(point.class_id),
            }
            for point in kp
        ]

        visualization = None

        if self._visualize_enabled():
            visualization = cv2.drawKeypoints(
                gray,
                kp,
                None
            )

        return (
            image,
            kp,
            keypoints,
            descriptors.astype(np.float32),
            visualization
        )

    def _extract_from_descriptors(self, value):
        # Roboflow v2 accepts pre-computed descriptor arrays.
        # For compatibility with the existing Novavision SIFT output,
        # nested JSON/list forms are also accepted.
        if isinstance(value, str):
            value = json.loads(value)

        if isinstance(value, np.ndarray):
            descriptors = value.astype(np.float32)
        else:
            descriptors = None

            if isinstance(value, list):
                # Existing SIFT package output:
                # [{"keyPoints": [{"descriptor": [...]}, ...]}]
                if value and isinstance(value[0], dict):
                    descriptor_list = []

                    for detection in value:
                        for kp in detection.get("keyPoints", []):
                            if "descriptor" in kp:
                                descriptor_list.append(
                                    kp["descriptor"]
                                )

                    if descriptor_list:
                        descriptors = np.asarray(
                            descriptor_list,
                            dtype=np.float32
                        )

                if descriptors is None:
                    descriptors = np.asarray(
                        value,
                        dtype=np.float32
                    )

            if descriptors is None:
                raise ValueError(
                    "Input must be an image or a SIFT descriptor array."
                )

        if descriptors.ndim != 2 or descriptors.shape[1] != 128:
            raise ValueError(
                "SIFT descriptors must have shape (N, 128)."
            )

        return descriptors

    def _visualize_enabled(self):
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

        if isinstance(value, dict):
            value = value.get(
                "value",
                value.get("name")
            )

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

        return False

    def _create_match_visualization(
        self,
        image_1,
        image_2,
        kp_1,
        kp_2,
        good_matches
    ):
        if self.matcher == "BFMatcher":
            draw_params = {
                "flags": cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            }
        else:
            draw_params = {
                "matchColor": (0, 255, 0),
                "singlePointColor": (0, 0, 255),
                "flags": cv2.DrawMatchesFlags_DEFAULT
            }

        # All good matches are drawn. There is no custom match-count limit.
        return cv2.drawMatches(
            image_1,
            kp_1,
            image_2,
            kp_2,
            good_matches,
            None,
            **draw_params
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
            image_1 = None
            image_2 = None
            kp_1 = None
            kp_2 = None

            # ------------------------------------------------
            # 1. Input 1
            # ------------------------------------------------
            if self._is_image(self.input_1):
                (
                    image_1,
                    kp_1,
                    self.keypoints_1,
                    self.descriptors_1,
                    self.visualization_1
                ) = self._extract_from_image(self.input_1)
            else:
                self.descriptors_1 = self._extract_from_descriptors(
                    self.input_1
                )

            # ------------------------------------------------
            # 2. Input 2
            # ------------------------------------------------
            if self._is_image(self.input_2):
                (
                    image_2,
                    kp_2,
                    self.keypoints_2,
                    self.descriptors_2,
                    self.visualization_2
                ) = self._extract_from_image(self.input_2)
            else:
                self.descriptors_2 = self._extract_from_descriptors(
                    self.input_2
                )

            # ------------------------------------------------
            # 3. Descriptor validation
            # ------------------------------------------------
            if (
                len(self.descriptors_1) < 2
                or len(self.descriptors_2) < 2
            ):
                return build_response_sift_comparison_test(
                    context=self
                )

            # ------------------------------------------------
            # 4. Matcher
            # ------------------------------------------------
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

            # ------------------------------------------------
            # 5. KNN + Lowe Ratio Test
            # ------------------------------------------------
            matches = matcher.knnMatch(
                self.descriptors_1,
                self.descriptors_2,
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
            # 6. Results
            # ------------------------------------------------
            self.good_matches_count = len(good_matches)

            self.images_match = (
                self.good_matches_count
                >= self.good_matches_threshold
            )

            # ------------------------------------------------
            # 7. Visualizations
            # Only generated when visualize=True AND both inputs
            # are actual images, matching Roboflow v2 behavior.
            # ------------------------------------------------
            if (
                self._visualize_enabled()
                and image_1 is not None
                and image_2 is not None
            ):
                frame_1 = self._get_frame(self.input_1)
                frame_2 = self._get_frame(self.input_2)

                if self.visualization_1 is not None:
                    self.visualization_1 = self._save_image(
                        frame_1,
                        self.visualization_1
                    )

                if self.visualization_2 is not None:
                    self.visualization_2 = self._save_image(
                        frame_2,
                        self.visualization_2
                    )

                matches_image = self._create_match_visualization(
                    image_1,
                    image_2,
                    kp_1,
                    kp_2,
                    good_matches
                )

                self.visualization_matches = self._save_image(
                    frame_1,
                    matches_image
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