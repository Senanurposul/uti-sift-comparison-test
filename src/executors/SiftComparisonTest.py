import os
import sys
import json

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.base.component import Component
from sdks.novavision.src.base.model import KeyPoints, Detection, Connection
from sdks.novavision.src.media.image import Image
from sdks.novavision.src.helper.executor import Executor

from components.SiftComparisonTest.src.models.PackageModel import PackageModel
from components.SiftComparisonTest.src.utils.response import (
    build_response_sift_comparison_test
)


class SiftComparisonTest(Component):

    def __init__(self, request, bootstrap):

        super().__init__(request, bootstrap)

        self.request.model = PackageModel(**(self.request.data))

        # ----------------------------------------------------
        # CONFIGS
        # ----------------------------------------------------

        self.good_matches_threshold = self.request.get_param(
            "GoodMatchesThreshold"
        )

        self.ratio_threshold = self.request.get_param(
            "RatioThreshold"
        )

        self.matcher = self.request.get_param(
            "Matcher"
        )

        # ----------------------------------------------------
        # SIFT OUTPUTS
        # ----------------------------------------------------

        self.sift_output_1 = self.request.get_param(
            "InputSIFTOutput1"
        )

        self.sift_output_2 = self.request.get_param(
            "InputSIFTOutput2"
        )

        # ----------------------------------------------------
        # ORIGINAL IMAGES
        # ----------------------------------------------------

        self.image_input_1 = self.request.get_param(
            "InputImage1"
        )

        self.image_input_2 = self.request.get_param(
            "InputImage2"
        )

        # Output variables
        self.output_detections = []
        self.output_visualization = None

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    # ========================================================
    # SIFT OUTPUT PARSING
    # ========================================================

    def _extract_keypoints_and_descriptors(self, sift_output):

        keypoints_dicts = []
        descriptors = []

        if isinstance(sift_output, str):
            sift_output = json.loads(sift_output)

        # Bazı durumlarda output doğrudan liste,
        # bazı durumlarda dictionary olabilir.
        if isinstance(sift_output, dict):

            if "value" in sift_output:
                sift_output = sift_output["value"]

            elif "detections" in sift_output:
                sift_output = sift_output["detections"]

        if not isinstance(sift_output, list):
            return keypoints_dicts, np.empty(
                (0, 128),
                dtype=np.float32
            )

        for detection in sift_output:

            for kp in detection.get("keyPoints", []):

                if "descriptor" not in kp:
                    continue

                keypoints_dicts.append({
                    "pt": (
                        kp["cx"],
                        kp["cy"]
                    ),
                    "size": kp.get("size", 1.0),
                    "angle": kp.get("angle", -1.0),
                    "response": kp.get("response", 0.0),
                    "octave": kp.get("octave", 0)
                })

                descriptors.append(
                    kp["descriptor"]
                )

        if not descriptors:
            return keypoints_dicts, np.empty(
                (0, 128),
                dtype=np.float32
            )

        return (
            keypoints_dicts,
            np.array(
                descriptors,
                dtype=np.float32
            )
        )

    # ========================================================
    # IMAGE VISUALIZATION
    # ========================================================

    def _create_visualization(
        self,
        image1,
        image2,
        keypoints1_dicts,
        keypoints2_dicts,
        good_matches
    ):

        # OpenCV KeyPoint nesnelerini oluşturuyoruz.
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

        # good_matches şu anda [[DMatch], [DMatch], ...]
        # şeklinde olduğu için düz listeye çeviriyoruz.
        draw_matches = [
            match[0]
            for match in good_matches
        ]

        visualization = cv2.drawMatches(
            image1,
            cv_keypoints1,
            image2,
            cv_keypoints2,
            draw_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        return visualization

    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        try:

            # ------------------------------------------------
            # SIFT OUTPUTLARINI AL
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

            # ------------------------------------------------
            # ORIGINAL IMAGES
            # ------------------------------------------------

            img1 = Image.get_frame(
                img=self.image_input_1,
                redis_db=self.redis_db
            )

            img2 = Image.get_frame(
                img=self.image_input_2,
                redis_db=self.redis_db
            )

            image1 = img1.value
            image2 = img2.value

            # ------------------------------------------------
            # YETERSİZ DESCRIPTOR
            # ------------------------------------------------

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

                # Yine de görüntüleri yan yana göster.
                visualization = np.hstack(
                    (
                        image1,
                        image2
                    )
                )

                visualization_image = Image.set_frame(
                    img=img1,
                    package_uID=self.uID,
                    redis_db=self.redis_db
                )

                visualization_image.value = visualization

                self.output_visualization = Image.set_frame(
                    img=visualization_image,
                    package_uID=self.uID,
                    redis_db=self.redis_db
                )

                return build_response_sift_comparison_test(
                    context=self
                )

            # ------------------------------------------------
            # MATCHER
            # ------------------------------------------------

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

            # ------------------------------------------------
            # KNN MATCHING
            # ------------------------------------------------

            matches = matcher.knnMatch(
                descriptors1,
                descriptors2,
                k=2
            )

            # ------------------------------------------------
            # LOWE RATIO TEST
            # ------------------------------------------------

            good_matches = []

            for pair in matches:

                if len(pair) < 2:
                    continue

                m, n = pair

                if m.distance < (
                    self.ratio_threshold * n.distance
                ):
                    good_matches.append([m])

            # ------------------------------------------------
            # MATCH COUNT
            # ------------------------------------------------

            good_matches_count = len(
                good_matches
            )

            images_match = (
                good_matches_count
                >= self.good_matches_threshold
            )

            # ------------------------------------------------
            # KEYPOINT OUTPUT
            # ------------------------------------------------

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
                    cx=float(kp["pt"][0]),
                    cy=float(kp["pt"][1]),
                    confidence=1.0
                )

                for kp in all_keypoints_dicts
            ]

            # ------------------------------------------------
            # CONNECTION OUTPUT
            # ------------------------------------------------

            connections = [

                Connection(
                    p1=match[0].queryIdx,
                    p2=match[0].trainIdx + offset
                )

                for match in good_matches
            ]

            # ------------------------------------------------
            # DETECTION OUTPUT
            # ------------------------------------------------

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

            # ------------------------------------------------
            # VISUALIZATION
            # ------------------------------------------------

            visualization = self._create_visualization(
                image1=image1,
                image2=image2,
                keypoints1_dicts=keypoints1_dicts,
                keypoints2_dicts=keypoints2_dicts,
                good_matches=good_matches
            )

            # ------------------------------------------------
            # VISUALIZATION IMAGE
            # ------------------------------------------------

            visualization_image = Image.set_frame(
                img=img1,
                package_uID=self.uID,
                redis_db=self.redis_db
            )

            visualization_image.value = visualization

            self.output_visualization = Image.set_frame(
                img=visualization_image,
                package_uID=self.uID,
                redis_db=self.redis_db
            )

        except Exception as e:

            print(
                "SIFT Comparison Error:",
                str(e)
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


if __name__ == "__main__":
    Executor(sys.argv[1]).run()