import os
import sys
import json
import cv2
import numpy as np

# ============================================================
# ROOT PATH
# ============================================================

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        '../../../../'
    )
)

# ============================================================
# NOVAVISION IMPORTS
# ============================================================

from sdks.novavision.src.base.component import Component
from sdks.novavision.src.base.model import (
    KeyPoints,
    Detection,
    Connection,
    Image
)
from sdks.novavision.src.helper.executor import Executor

from components.SiftComparisonTest.src.utils.response import (
    build_response_sift_comparison_test
)

from components.SiftComparisonTest.src.models.PackageModel import (
    PackageModel
)


# ============================================================
# SIFT COMPARISON COMPONENT
# ============================================================

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

        self.sift_output_1 = self.request.get_param(
            "InputSIFTOutput1"
        )

        self.sift_output_2 = self.request.get_param(
            "InputSIFTOutput2"
        )

        # Görseller (Test sırasında SIFT hesaplamak ve görselleştirmek için)
        self.image_1 = self.request.get_param("InputImage1")
        self.image_2 = self.request.get_param("InputImage2")

        self.output_matches_image = None

    # ========================================================
    # BOOTSTRAP
    # ========================================================

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    # ========================================================
    # GEÇİCİ SIFT HESAPLAMA (TEST FALLBACK)
    # ========================================================

    def _compute_sift_fallback(self, image_input):
        """
        Dışarıdan SIFT verisi gelmediğinde görselden SIFT hesaplar.
        SIFT paketi hazır olunca bu fonksiyon silinecek.
        """
        if not image_input:
            return [], np.empty((0, 128), dtype=np.float32)

        # NovaVision Image nesnesinden numpy array alımı
        img_np = image_input.get_image() if hasattr(image_input, 'get_image') else image_input

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

    # ========================================================
    # SIFT OUTPUT PARSING
    # ========================================================

    def _extract_keypoints_and_descriptors(self, sift_output):
        keypoints_dicts = []
        descriptors = []

        if isinstance(sift_output, str):
            sift_output = json.loads(sift_output)

        if not isinstance(sift_output, list):
            raise ValueError("SIFT output must be a list.")

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

    # ========================================================
    # VISUALIZATION HELPER
    # ========================================================

    def _create_matches_visualization(self, kp1_dicts, kp2_dicts, good_matches):
        if not self.image_1 or not self.image_2:
            return None

        img1 = self.image_1.get_image() if hasattr(self.image_1, 'get_image') else self.image_1
        img2 = self.image_2.get_image() if hasattr(self.image_2, 'get_image') else self.image_2

        cv2_kp1 = [cv2.KeyPoint(x=kp["pt"][0], y=kp["pt"][1], size=kp.get("size", 1.0), angle=kp.get("angle", -1.0)) for kp in kp1_dicts]
        cv2_kp2 = [cv2.KeyPoint(x=kp["pt"][0], y=kp["pt"][1], size=kp.get("size", 1.0), angle=kp.get("angle", -1.0)) for kp in kp2_dicts]

        drawn_matches = cv2.drawMatches(
            img1, cv2_kp1,
            img2, cv2_kp2,
            good_matches,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(0, 0, 255),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        res_image = Image()
        res_image.set_image(drawn_matches)
        return res_image

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
                imgUID=self.uID
            )
        ]

    # ========================================================
    # RUN
    # ========================================================

    def run(self):
        try:
            # 1. SIFT verisi varsa parse et, yoksa görsellerden kendin hesapla (Test Mode)
            if self.sift_output_1:
                keypoints1_dicts, descriptors1 = self._extract_keypoints_and_descriptors(self.sift_output_1)
            else:
                keypoints1_dicts, descriptors1 = self._compute_sift_fallback(self.image_1)

            if self.sift_output_2:
                keypoints2_dicts, descriptors2 = self._extract_keypoints_and_descriptors(self.sift_output_2)
            else:
                keypoints2_dicts, descriptors2 = self._compute_sift_fallback(self.image_2)

            # 2. Yeterli descriptor yoksa direkt dön
            if len(descriptors1) < 2 or len(descriptors2) < 2:
                self.output_detections = self._no_match_result()
                return build_response_sift_comparison_test(context=self)

            # 3. Matcher seçimi
            if self.matcher == "FlannBasedMatcher":
                index_params = {"algorithm": 1, "trees": 5}
                search_params = {"checks": 50}
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
            elif self.matcher == "BFMatcher":
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            else:
                raise ValueError(f"Unsupported matcher: {self.matcher}")

            # 4. Matching & Lowe's Ratio Test
            matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

            good_matches = []
            for match_pair in matches:
                if len(match_pair) < 2:
                    continue
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)

            good_matches_count = len(good_matches)
            images_match = good_matches_count >= self.good_matches_threshold

            # 5. Metadata Çıktıları (Dashboard Çizgileri)
            all_keypoints_dicts = keypoints1_dicts + keypoints2_dicts
            offset = len(keypoints1_dicts)

            keypoints = [
                KeyPoints(
                    cx=float(kp["pt"][0]),
                    cy=float(kp["pt"][1]),
                    confidence=1.0
                )
                for kp in all_keypoints_dicts
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

            # 6. Yan yana çizilmiş sonuç görseli
            if self.image_1 and self.image_2:
                self.output_matches_image = self._create_matches_visualization(
                    keypoints1_dicts,
                    keypoints2_dicts,
                    good_matches
                )

        except Exception as e:
            print("SIFT Comparison Error:", repr(e))
            raise

        return build_response_sift_comparison_test(context=self)


# ============================================================
# EXECUTOR
# ============================================================

if __name__ == "__main__":
    Executor(
        sys.argv[1]
    ).run()