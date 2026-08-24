import os
import sys
import json

import cv2
import numpy as np


sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../../../",
    )
)


from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.base.model import (
    KeyPoints,
    Detection,
    Connection,
)
from sdks.novavision.src.helper.executor import Executor

from components.SiftComparisonTest.src.models.PackageModel import (
    PackageModel,
)

from components.SiftComparisonTest.src.utils.response import (
    build_response_sift_comparison_test,
)


class SiftComparisonTest(Component):

    def __init__(self, request, bootstrap):

        super().__init__(
            request,
            bootstrap,
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
        # VISUALIZE
        # ====================================================

        visualize = (
            self.request.get_param(
                "Visualize"
            )
        )

        if isinstance(
            visualize,
            bool,
        ):

            self.visualize = visualize

        elif hasattr(
            visualize,
            "value",
        ):

            self.visualize = bool(
                visualize.value
            )

        elif isinstance(
            visualize,
            str,
        ):

            self.visualize = (
                visualize.strip().lower()
                in (
                    "true",
                    "1",
                    "yes",
                    "on",
                )
            )

        else:

            self.visualize = False

        # ====================================================
        # INPUTS
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
        # VISUALIZATION OUTPUTS
        # ====================================================

        self.output_visualization_1 = None
        self.output_visualization_2 = None
        self.output_visualization_matches = None

    # ========================================================
    # BOOTSTRAP
    # ========================================================

    @staticmethod
    def bootstrap(
        config: dict,
    ) -> dict:

        return {}

    # ========================================================
    # SIFT OUTPUT PARSING
    # ========================================================

    def _extract_keypoints_and_descriptors(
        self,
        sift_output,
    ):

        keypoints_dicts = []
        descriptors = []

        if isinstance(
            sift_output,
            str,
        ):

            sift_output = json.loads(
                sift_output
            )

        if not isinstance(
            sift_output,
            list,
        ):

            raise ValueError(
                "SIFT output must be a list."
            )

        for detection in sift_output:

            for kp in detection.get(
                "keyPoints",
                [],
            ):

                if "descriptor" not in kp:
                    continue

                keypoints_dicts.append(
                    {
                        "pt": (
                            float(
                                kp["cx"]
                            ),
                            float(
                                kp["cy"]
                            ),
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
                    (
                        0,
                        128,
                    ),
                    dtype=np.float32,
                ),
            )

        descriptors = np.asarray(
            descriptors,
            dtype=np.float32,
        )

        return (
            keypoints_dicts,
            descriptors,
        )

    # ========================================================
    # NO MATCH RESULT
    # ========================================================

    def _no_match_result(self):

        return [
            Detection(
                boundingBox=None,
                keyPoints=[],
                connections=[],
                confidence=0.0,
                classId=0,
                classLabel="NoMatch",
                imgUID=self.uID,
            )
        ]

    # ========================================================
    # FRAME PREPARATION
    # ========================================================

    @staticmethod
    def _prepare_frame(
        frame,
    ):

        frame = np.asarray(
            frame
        )

        if frame.size == 0:

            raise ValueError(
                "Visualization frame is empty"
            )

        if frame.dtype != np.uint8:

            frame = frame.astype(
                np.uint8
            )

        if len(
            frame.shape
        ) == 2:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_GRAY2BGR,
            )

        return frame

    # ========================================================
    # CONVERT KEYPOINTS
    # ========================================================

    @staticmethod
    def _to_cv_keypoints(
        keypoints_dicts,
    ):

        return [
            cv2.KeyPoint(
                float(
                    kp["pt"][0]
                ),
                float(
                    kp["pt"][1]
                ),
                1.0,
            )
            for kp in keypoints_dicts
        ]

    # ========================================================
    # KEYPOINT VISUALIZATION
    # ========================================================

    def _create_keypoint_visualization(
        self,
        frame,
        keypoints_dicts,
    ):

        frame = self._prepare_frame(
            frame
        )

        cv_keypoints = (
            self._to_cv_keypoints(
                keypoints_dicts
            )
        )

        return cv2.drawKeypoints(
            frame,
            cv_keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

    # ========================================================
    # MATCH VISUALIZATION
    # ========================================================

    def _create_visualization_matches(
        self,
        frame1,
        frame2,
        keypoints1,
        keypoints2,
        good_matches,
    ):

        frame1 = self._prepare_frame(
            frame1
        )

        frame2 = self._prepare_frame(
            frame2
        )

        cv_keypoints1 = (
            self._to_cv_keypoints(
                keypoints1
            )
        )

        cv_keypoints2 = (
            self._to_cv_keypoints(
                keypoints2
            )
        )

        return cv2.drawMatches(
            frame1,
            cv_keypoints1,
            frame2,
            cv_keypoints2,
            list(good_matches),
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    # ========================================================
    # IMAGE OUTPUT
    # ========================================================

    def _make_image_output(
        self,
        frame,
    ):

        frame.value = (
            frame.value.astype(
                np.uint8
            )
        )

        return Image.set_frame(
            img=frame,
            package_uID=self.uID,
            redis_db=self.redis_db,
        )

    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        try:

            # ==================================================
            # SIFT 1
            # ==================================================

            (
                keypoints1_dicts,
                descriptors1,
            ) = (
                self._extract_keypoints_and_descriptors(
                    self.sift_output_1
                )
            )

            # ==================================================
            # SIFT 2
            # ==================================================

            (
                keypoints2_dicts,
                descriptors2,
            ) = (
                self._extract_keypoints_and_descriptors(
                    self.sift_output_2
                )
            )

            # ==================================================
            # NOT ENOUGH DESCRIPTORS
            # ==================================================

            if (
                len(descriptors1) < 2
                or len(descriptors2) < 2
            ):

                self.output_detections = (
                    self._no_match_result()
                )

                return (
                    build_response_sift_comparison_test(
                        context=self
                    )
                )

            # ==================================================
            # FLANN
            # ==================================================

            if (
                self.matcher
                == "FlannBasedMatcher"
            ):

                index_params = {
                    "algorithm": 1,
                    "trees": 5,
                }

                search_params = {
                    "checks": 50,
                }

                matcher = (
                    cv2.FlannBasedMatcher(
                        index_params,
                        search_params,
                    )
                )

            # ==================================================
            # BF MATCHER
            # ==================================================

            elif (
                self.matcher
                == "BFMatcher"
            ):

                matcher = cv2.BFMatcher(
                    cv2.NORM_L2,
                    crossCheck=False,
                )

            else:

                raise ValueError(
                    f"Unsupported matcher: "
                    f"{self.matcher}"
                )

            # ==================================================
            # KNN MATCH
            # ==================================================

            matches = matcher.knnMatch(
                descriptors1,
                descriptors2,
                k=2,
            )

            # ==================================================
            # LOWE RATIO TEST
            # ==================================================

            good_matches = []

            for match_pair in matches:

                if len(
                    match_pair
                ) < 2:

                    continue

                m, n = match_pair

                if (
                    m.distance
                    < (
                        self.ratio_threshold
                        * n.distance
                    )
                ):

                    good_matches.append(
                        m
                    )

            # ==================================================
            # MATCH COUNT
            # ==================================================

            good_matches_count = len(
                good_matches
            )

            images_match = (
                good_matches_count
                >= self.good_matches_threshold
            )

            # ==================================================
            # KEYPOINT OUTPUT
            # ==================================================

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
                    confidence=1.0,
                )
                for kp in all_keypoints_dicts
            ]

            # ==================================================
            # CONNECTIONS
            # ==================================================

            connections = [
                Connection(
                    p1=m.queryIdx,
                    p2=(
                        m.trainIdx
                        + offset
                    ),
                )
                for m in good_matches
            ]

            # ==================================================
            # DETECTIONS
            # ==================================================

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
                    imgUID=self.uID,
                )
            ]

            # ==================================================
            # VISUALIZATION
            # ==================================================

            if (
                self.visualize
                and self.visualization_input_1
                is not None
                and self.visualization_input_2
                is not None
            ):

                image1_frame = (
                    Image.get_frame(
                        img=(
                            self.visualization_input_1
                        ),
                        redis_db=self.redis_db,
                    )
                )

                image2_frame = (
                    Image.get_frame(
                        img=(
                            self.visualization_input_2
                        ),
                        redis_db=self.redis_db,
                    )
                )

                if (
                    image1_frame
                    is not None
                    and image2_frame
                    is not None
                    and image1_frame.value
                    is not None
                    and image2_frame.value
                    is not None
                ):

                    # ------------------------------------------
                    # ORIGINAL FRAMES
                    # ------------------------------------------

                    raw_frame_1 = (
                        image1_frame.value
                    )

                    raw_frame_2 = (
                        image2_frame.value
                    )

                    # ------------------------------------------
                    # VISUALIZATION 1
                    # ------------------------------------------

                    vis1 = (
                        self._create_keypoint_visualization(
                            raw_frame_1,
                            keypoints1_dicts,
                        )
                    )

                    # ------------------------------------------
                    # VISUALIZATION 2
                    # ------------------------------------------

                    vis2 = (
                        self._create_keypoint_visualization(
                            raw_frame_2,
                            keypoints2_dicts,
                        )
                    )

                    # ------------------------------------------
                    # MATCH VISUALIZATION
                    # ------------------------------------------

                    vis_matches = (
                        self._create_visualization_matches(
                            raw_frame_1,
                            raw_frame_2,
                            keypoints1_dicts,
                            keypoints2_dicts,
                            good_matches,
                        )
                    )

                    # ------------------------------------------
                    # OUTPUT 1
                    # ------------------------------------------

                    image1_frame.value = vis1

                    self.output_visualization_1 = (
                        self._make_image_output(
                            image1_frame
                        )
                    )

                    # ------------------------------------------
                    # OUTPUT 2
                    # ------------------------------------------

                    image2_frame.value = vis2

                    self.output_visualization_2 = (
                        self._make_image_output(
                            image2_frame
                        )
                    )

                    # ------------------------------------------
                    # MATCH OUTPUT
                    # ------------------------------------------

                    matches_frame = (
                        Image.get_frame(
                            img=(
                                self.visualization_input_1
                            ),
                            redis_db=self.redis_db,
                        )
                    )

                    matches_frame.value = (
                        vis_matches
                    )

                    self.output_visualization_matches = (
                        self._make_image_output(
                            matches_frame
                        )
                    )

        except Exception as e:

            print(
                "SIFT Comparison Error:",
                repr(e),
                flush=True,
            )

            raise

        return (
            build_response_sift_comparison_test(
                context=self
            )
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    Executor(
        sys.argv[1]
    ).run()