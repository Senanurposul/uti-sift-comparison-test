import os
import sys
import json
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.base.component import Component
from sdks.novavision.src.base.model import KeyPoints, Detection, Connection
from sdks.novavision.src.helper.executor import Executor
from components.SiftComparisonTest.src.utils.response import build_response_sift_comparison_test
from components.SiftComparisonTest.src.models.PackageModel import PackageModel


class SiftComparisonTest(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        self.good_matches_threshold = self.request.get_param("GoodMatchesThreshold")
        self.ratio_threshold = self.request.get_param("RatioThreshold")
        self.matcher = self.request.get_param("Matcher")
        self.sift_output_1 = self.request.get_param("InputSIFTOutput1")
        self.sift_output_2 = self.request.get_param("InputSIFTOutput2")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _extract_keypoints_and_descriptors(self, sift_output):
        keypoints_dicts = []
        descriptors = []

        if isinstance(sift_output, str):
            sift_output = json.loads(sift_output)

        for detection in sift_output:
            for kp in detection.get("keyPoints", []):
                keypoints_dicts.append({
                    "pt": (kp["cx"], kp["cy"]),
                    "size": kp["size"],
                    "angle": kp["angle"],
                    "response": kp["response"],
                    "octave": kp["octave"],
                })
                descriptors.append(kp["descriptor"])
        return keypoints_dicts, np.array(descriptors, dtype=np.float32)

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
            keypoints1_dicts, descriptors1 = self._extract_keypoints_and_descriptors(self.sift_output_1)
            keypoints2_dicts, descriptors2 = self._extract_keypoints_and_descriptors(self.sift_output_2)

            if len(descriptors1) < 2 or len(descriptors2) < 2:
                self.output_detections = self._no_match_result()
                return build_response_sift_comparison_test(context=self)

            if self.matcher == "BFMatcher":
                matcher = cv2.BFMatcher(cv2.NORM_L2)
            else:
                matcher = cv2.FlannBasedMatcher(
                    dict(algorithm=1, trees=5),
                    dict(checks=50)
                )

            matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

            good_matches = []
            for m, n in matches:
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)

            good_matches_count = len(good_matches)
            images_match = good_matches_count >= self.good_matches_threshold

            all_keypoints_dicts = keypoints1_dicts + keypoints2_dicts
            offset = len(keypoints1_dicts)

            keypoints = [
                KeyPoints(cx=float(kp["pt"][0]), cy=float(kp["pt"][1]), confidence=1.0)
                for kp in all_keypoints_dicts
            ]

            connections = [
                Connection(p1=m.queryIdx, p2=m.trainIdx + offset)
                for m in good_matches
            ]

            self.output_detections = [
                Detection(
                    boundingBox=None,
                    keyPoints=keypoints,
                    connections=connections,
                    confidence=float(good_matches_count),
                    classId=1 if images_match else 0,
                    classLabel="Match" if images_match else "NoMatch",
                    imgUID=self.uID
                )
            ]

        except Exception:
            self.output_detections = self._no_match_result()

        return build_response_sift_comparison_test(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()