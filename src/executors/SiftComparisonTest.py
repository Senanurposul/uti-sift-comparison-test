import os
import sys
import json
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image as MediaImage
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.base.model import KeyPoints, Detection, Connection
from sdks.novavision.src.helper.executor import Executor
from components.SiftComparisonTest.src.models.PackageModel import PackageModel
from components.SiftComparisonTest.src.utils.response import build_response_sift_comparison


class SiftComparisonExecutor(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        self.good_matches_threshold = self.request.get_param("goodMatchesThreshold")
        self.ratio_threshold = self.request.get_param("ratioThreshold")
        self.matcher = self.request.get_param("matcher")

        self.sift_output_1 = self.request.get_param("inputSIFTOutput1")
        self.sift_output_2 = self.request.get_param("inputSIFTOutput2")

        self.image_one_raw = self.request.get_param("inputImageOne")
        self.image_two_raw = self.request.get_param("inputImageTwo")

        self.output_detections = []
        self.output_matches_image = None

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _compute_sift_fallback(self, img_np):
        if img_np is None:
            return [], np.empty((0, 128), dtype=np.float32)

        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_np

        sift = cv2.SIFT_create()
        cv2_kps, desc = sift.detectAndCompute(gray, None)

        if desc is None:
            return [], np.empty((0, 128), dtype=np.float32)

        keypoints_dicts = [
            {
                "pt": (float(kp.pt[0]), float(kp.pt[1])),
                "size": float(kp.size),
                "angle": float(kp.angle)
            }
            for kp in cv2_kps
        ]
        return keypoints_dicts, desc.astype(np.float32)

    def _extract_keypoints_and_descriptors(self, sift_output):
        keypoints_dicts = []
        descriptors = []

        if isinstance(sift_output, str):
            sift_output = json.loads(sift_output)

        if not isinstance(sift_output, list):
            return [], np.empty((0, 128), dtype=np.float32)

        for detection in sift_output:
            kps = detection.get("keyPoints", []) or detection.get("keypoints", [])
            for kp in kps:
                if "descriptor" not in kp:
                    continue

                cx = kp.get("cx", kp.get("pt", [0, 0])[0] if isinstance(kp.get("pt"), (list, tuple)) else kp.get("x", 0.0))
                cy = kp.get("cy", kp.get("pt", [0, 0])[1] if isinstance(kp.get("pt"), (list, tuple)) else kp.get("y", 0.0))

                keypoints_dicts.append({
                    "pt": (float(cx), float(cy)),
                    "size": float(kp.get("size", 1.0)),
                    "angle": float(kp.get("angle", -1.0))
                })
                descriptors.append(kp["descriptor"])

        if not descriptors:
            return keypoints_dicts, np.empty((0, 128), dtype=np.float32)

        return keypoints_dicts, np.asarray(descriptors, dtype=np.float32)

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
        # 1. Görseller varsa Redis'ten frame olarak çekilir
        frame_one = None
        frame_two = None
        if self.image_one_raw:
            frame_one = MediaImage.get_frame(img=self.image_one_raw, redis_db=self.redis_db)
        if self.image_two_raw:
            frame_two = MediaImage.get_frame(img=self.image_two_raw, redis_db=self.redis_db)

        # 2. SIFT verisi varsa oku, yoksa görselden doğrudan hesapla
        if self.sift_output_1:
            kps1, desc1 = self._extract_keypoints_and_descriptors(self.sift_output_1)
        elif frame_one is not None:
            kps1, desc1 = self._compute_sift_fallback(frame_one.value)
        else:
            kps1, desc1 = [], np.empty((0, 128), dtype=np.float32)

        if self.sift_output_2:
            kps2, desc2 = self._extract_keypoints_and_descriptors(self.sift_output_2)
        elif frame_two is not None:
            kps2, desc2 = self._compute_sift_fallback(frame_two.value)
        else:
            kps2, desc2 = [], np.empty((0, 128), dtype=np.float32)

        # 3. Yeterli descriptor kontrolü
        if len(desc1) < 2 or len(desc2) < 2:
            self.output_detections = self._no_match_result()
            return build_response_sift_comparison(context=self)

        # 4. Matcher
        if self.matcher == "FlannBasedMatcher":
            matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        matches = matcher.knnMatch(desc1, desc2, k=2)

        # 5. Lowe's Ratio Test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)

        good_matches_count = len(good_matches)
        images_match = good_matches_count >= self.good_matches_threshold

        # 6. Detection Metadata (Keypoints & Connections)
        all_kps = kps1 + kps2
        offset = len(kps1)

        keypoints = [
            KeyPoints(
                cx=float(kp["pt"][0]),
                cy=float(kp["pt"][1]),
                confidence=1.0
            )
            for kp in all_kps
        ]

        connections = [
            Connection(
                p1=m.queryIdx,
                p2=m.trainIdx + offset
            )
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

        # 7. Visualization Görseli (Eğer iki görsel de verildiyse drawMatches ile oluşturup Redis'e yazılır)
        if frame_one is not None and frame_two is not None:
            cv2_kp1 = [cv2.KeyPoint(x=kp["pt"][0], y=kp["pt"][1], size=kp.get("size", 1.0), angle=kp.get("angle", -1.0)) for kp in kps1]
            cv2_kp2 = [cv2.KeyPoint(x=kp["pt"][0], y=kp["pt"][1], size=kp.get("size", 1.0), angle=kp.get("angle", -1.0)) for kp in kps2]

            drawn = cv2.drawMatches(
                frame_one.value, cv2_kp1,
                frame_two.value, cv2_kp2,
                good_matches,
                None,
                matchColor=(0, 255, 0),
                singlePointColor=(0, 0, 255),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            frame_one.value = drawn
            self.output_matches_image = MediaImage.set_frame(img=frame_one, package_uID=self.uID, redis_db=self.redis_db)

        return build_response_sift_comparison(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()