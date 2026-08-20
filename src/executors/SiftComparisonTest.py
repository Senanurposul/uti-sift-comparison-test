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

        # ----------------------------------------------------
        # CONFIG
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

        self.visualization_input_1 = self.request.get_param(
            "InputVisualization1"
        )

        self.visualization_input_2 = self.request.get_param(
            "InputVisualization2"
        )

        # ----------------------------------------------------
        # OUTPUTS
        # ----------------------------------------------------

        self.output_detections = []

        self.output_visualization = None

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    # ========================================================
    # SIFT OUTPUT PARSE
    # ========================================================

    def _extract_keypoints_and_descriptors(self, sift_output):

        keypoints_dicts = []
        descriptors = []

        if isinstance(sift_output, str):
            sift_output = json.loads(sift_output)

        # Bazı durumlarda output doğrudan liste,
        # bazı durumlarda value altında gelebilir.
        if isinstance(sift_output, dict):
            sift_output = sift_output.get("value", [])

        if sift_output is None:
            return [], np.empty((0, 128), dtype=np.float32)

        for detection in sift_output:

            if not isinstance(detection, dict):
                continue

            for kp in detection.get("keyPoints", []):

                descriptor = kp.get("descriptor")

                if descriptor is None:
                    continue

                keypoints_dicts.append({
                    "pt": (
                        float(kp["cx"]),
                        float(kp["cy"])
                    ),
                    "size": float(kp.get("size", 1.0)),
                    "angle": float(kp.get("angle", -1.0)),
                    "response": float(kp.get("response", 0.0)),
                    "octave": int(kp.get("octave", 0)),
                })

                descriptors.append(descriptor)

        if len(descriptors) == 0:
            return (
                keypoints_dicts,
                np.empty((0, 128), dtype=np.float32)
            )

        return (
            keypoints_dicts,
            np.asarray(descriptors, dtype=np.float32)
        )

    # ========================================================
    # OPENCV KEYPOINT OLUŞTUR
    # ========================================================

    def _create_cv_keypoints(self, keypoints_dicts):

        cv_keypoints = []

        for kp in keypoints_dicts:

            cv_kp = cv2.KeyPoint(
                x=float(kp["pt"][0]),
                y=float(kp["pt"][1]),
                size=max(float(kp["size"]), 1.0),
                angle=float(kp["angle"]),
                response=float(kp["response"]),
                octave=int(kp["octave"]),
                class_id=-1
            )

            cv_keypoints.append(cv_kp)

        return cv_keypoints

    # ========================================================
    # VISUALIZATION
    # ========================================================

    def _create_visualization(
        self,
        keypoints1_dicts,
        keypoints2_dicts,
        good_matches
    ):

        # ----------------------------------------------------
        # Image 1
        # ----------------------------------------------------

        image1 = Image.get_frame(
            img=self.visualization_input_1,
            redis_db=self.redis_db
        )

        # ----------------------------------------------------
        # Image 2
        # ----------------------------------------------------

        image2 = Image.get_frame(
            img=self.visualization_input_2,
            redis_db=self.redis_db
        )

        if image1 is None:
            raise ValueError("Visualization Image 1 alınamadı.")

        if image2 is None:
            raise ValueError("Visualization Image 2 alınamadı.")

        frame1 = image1.value
        frame2 = image2.value

        if frame1 is None:
            raise ValueError("Visualization Image 1 frame değeri boş.")

        if frame2 is None:
            raise ValueError("Visualization Image 2 frame değeri boş.")

        frame1 = np.asarray(frame1)
        frame2 = np.asarray(frame2)

        # ----------------------------------------------------
        # OpenCV KeyPoints
        # ----------------------------------------------------

        cv_keypoints1 = self._create_cv_keypoints(
            keypoints1_dicts
        )

        cv_keypoints2 = self._create_cv_keypoints(
            keypoints2_dicts
        )

        # ----------------------------------------------------
        # DMatch listesini hazırla
        # ----------------------------------------------------

        draw_matches = []

        for match in good_matches:

            if isinstance(match, list):
                if len(match) == 0:
                    continue

                draw_matches.append(match[0])

            else:
                draw_matches.append(match)

        # ----------------------------------------------------
        # MATCH VISUALIZATION
        # ----------------------------------------------------

        visualization = cv2.drawMatches(
            frame1,
            cv_keypoints1,
            frame2,
            cv_keypoints2,
            draw_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # ----------------------------------------------------
        # Visualization görüntüsünü NovaVision Image'a yaz
        # ----------------------------------------------------

        image1.value = visualization

        output_image = Image.set_frame(
            img=image1,
            package_uID=self.uID,
            redis_db=self.redis_db
        )

        return output_image

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
            # MATCHER
            # ------------------------------------------------

            good_matches = []

            if len(descriptors1) >= 2 and len(descriptors2) >= 2:

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

                matches = matcher.knnMatch(
                    descriptors1,
                    descriptors2,
                    k=2
                )

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
                        good_matches.append([m])

            # ------------------------------------------------
            # GOOD MATCH COUNT
            # ------------------------------------------------

            good_matches_count = len(good_matches)

            images_match = (
                good_matches_count
                >= int(self.good_matches_threshold)
            )

            # ------------------------------------------------
            # DETECTION KEYPOINTS
            # ------------------------------------------------

            all_keypoints_dicts = (
                keypoints1_dicts
                +
                keypoints2_dicts
            )

            offset = len(keypoints1_dicts)

            keypoints = [

                KeyPoints(
                    cx=float(kp["pt"][0]),
                    cy=float(kp["pt"][1]),
                    confidence=1.0
                )

                for kp in all_keypoints_dicts
            ]

            # ------------------------------------------------
            # CONNECTIONS
            # ------------------------------------------------

            connections = [

                Connection(
                    p1=match[0].queryIdx,
                    p2=match[0].trainIdx + offset
                )

                for match in good_matches
            ]

            # ------------------------------------------------
            # OUTPUT DETECTIONS
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
                        1 if images_match else 0
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

            self.output_visualization = (
                self._create_visualization(
                    keypoints1_dicts,
                    keypoints2_dicts,
                    good_matches
                )
            )

        except Exception as e:

            # ------------------------------------------------
            # HATA DURUMU
            # ------------------------------------------------

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

            print(
                "SIFT Comparison Test Error:",
                repr(e)
            )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return build_response_sift_comparison_test(
            context=self
        )


if "__main__" == __name__:
    Executor(sys.argv[1]).run()