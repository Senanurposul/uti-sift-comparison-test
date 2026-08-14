import os
import sys
import json
import base64
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
    Connection
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

        # Request'i PackageModel ile dogruluyoruz.
        self.request.model = PackageModel(
            **self.request.data
        )

        # Minimum good match sayisi.
        self.good_matches_threshold = self.request.get_param(
            "GoodMatchesThreshold"
        )

        # Lowe Ratio Test threshold.
        self.ratio_threshold = self.request.get_param(
            "RatioThreshold"
        )

        # Matcher secimi.
        self.matcher = self.request.get_param(
            "Matcher"
        )

        # Visualize aktif mi.
        self.visualize = self.request.get_param(
            "Visualize"
        )

        # Birinci SIFT output'u.
        self.sift_output_1 = self.request.get_param(
            "InputSIFTOutput1"
        )

        # Ikinci SIFT output'u.
        self.sift_output_2 = self.request.get_param(
            "InputSIFTOutput2"
        )

        # Visualize icin opsiyonel goruntuler.
        self.image_1 = self.request.get_param(
            "InputImage1"
        )

        self.image_2 = self.request.get_param(
            "InputImage2"
        )


    # ========================================================
    # BOOTSTRAP
    # ========================================================

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
        """
        SIFT output'undan:

        - keypoint koordinatlarini
        - descriptor'lari

        ayirir.
        """

        keypoints_dicts = []
        descriptors = []

        # JSON string geldiyse Python objesine ceviriyoruz.
        if isinstance(sift_output, str):
            sift_output = json.loads(sift_output)

        # SIFT output'unun liste olmasi gerekiyor.
        if not isinstance(sift_output, list):
            raise ValueError(
                "SIFT output must be a list."
            )

        # Detection'larin icerisindeki keypoint'leri
        # topluyoruz.
        for detection in sift_output:

            for kp in detection.get("keyPoints", []):

                # Descriptor yoksa bu keypoint'i
                # matching islemine almiyoruz.
                if "descriptor" not in kp:
                    continue

                # Keypoint koordinatlarini saklıyoruz.
                keypoints_dicts.append(
                    {
                        "pt": (
                            float(kp["cx"]),
                            float(kp["cy"])
                        )
                    }
                )

                # Descriptor'i saklıyoruz.
                descriptors.append(
                    kp["descriptor"]
                )

        # Descriptor bulunamadiysa bos matris.
        if not descriptors:
            return (
                keypoints_dicts,
                np.empty(
                    (0, 128),
                    dtype=np.float32
                )
            )

        # OpenCV icin float32 kullaniyoruz.
        descriptors = np.asarray(
            descriptors,
            dtype=np.float32
        )

        return (
            keypoints_dicts,
            descriptors
        )


    # ========================================================
    # IMAGE DECODE (SADECE VISUALIZE ICIN)
    # ========================================================

    def _decode_image(self, raw_input):
        """
        Novavision image temsilini (base64 string ya da
        zaten numpy array) OpenCV image'ine cevirir.

        NOT: Bu decode mantigi projenizdeki "Image Load"
        component'inin kullandigi yontemle ayni olmalidir.
        Format farkliysa burayi ona gore uyarlayin.
        """

        if raw_input is None:
            return None

        if isinstance(raw_input, np.ndarray):
            return raw_input

        if isinstance(raw_input, str):
            try:
                img_bytes = base64.b64decode(raw_input)
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                return None

        return None


    # ========================================================
    # VISUALIZATION
    # ========================================================

    def _build_visualization(
        self,
        img1,
        img2,
        keypoints1_dicts,
        keypoints2_dicts,
        good_matches
    ):
        """
        Gercek goruntuler uzerinde eslesen keypoint'leri
        cv2.drawMatches ile gorsellestirir.
        """

        cv_kp1 = [
            cv2.KeyPoint(x=kp["pt"][0], y=kp["pt"][1], size=10)
            for kp in keypoints1_dicts
        ]

        cv_kp2 = [
            cv2.KeyPoint(x=kp["pt"][0], y=kp["pt"][1], size=10)
            for kp in keypoints2_dicts
        ]

        result_img = cv2.drawMatches(
            img1, cv_kp1,
            img2, cv_kp2,
            good_matches,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        _, buffer = cv2.imencode(".png", result_img)

        return base64.b64encode(buffer).decode("utf-8")


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

        # Visualize varsayilan olarak None.
        self.output_visualization = None

        try:

            # ------------------------------------------------
            # 1. SIFT output'larini aliyoruz.
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
            # 2. DEBUG
            # ------------------------------------------------

            print("")
            print("========================================")
            print("       SIFT COMPARISON DEBUG")
            print("========================================")
            print(
                "GoodMatchesThreshold:",
                self.good_matches_threshold
            )
            print(
                "RatioThreshold:",
                self.ratio_threshold
            )
            print(
                "Matcher:",
                self.matcher
            )
            print(
                "Visualize:",
                self.visualize
            )
            print(
                "Descriptor1 count:",
                len(descriptors1)
            )
            print(
                "Descriptor2 count:",
                len(descriptors2)
            )
            print("========================================")
            print("")


            # ------------------------------------------------
            # 3. En az 2 descriptor gerekiyor.
            # ------------------------------------------------

            if (
                len(descriptors1) < 2
                or len(descriptors2) < 2
            ):

                self.output_detections = (
                    self._no_match_result()
                )

                return build_response_sift_comparison_test(
                    context=self
                )


            # ------------------------------------------------
            # 4. MATCHER
            # ------------------------------------------------

            if self.matcher == "FlannBasedMatcher":

                # SIFT descriptor'lari float oldugu icin
                # FLANN + KD-Tree kullaniyoruz.
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

            elif self.matcher == "BFMatcher":

                # SIFT descriptor'lari icin
                # Euclidean distance / L2 norm.
                matcher = cv2.BFMatcher(
                    cv2.NORM_L2,
                    crossCheck=False
                )

            else:

                raise ValueError(
                    f"Unsupported matcher: {self.matcher}"
                )


            # ------------------------------------------------
            # 5. KNN MATCHING
            # ------------------------------------------------

            matches = matcher.knnMatch(
                descriptors1,
                descriptors2,
                k=2
            )


            # ------------------------------------------------
            # 6. LOWE RATIO TEST
            # ------------------------------------------------

            good_matches = []

            for match_pair in matches:

                # Iki sonuc yoksa ratio testi yapilamaz.
                if len(match_pair) < 2:
                    continue

                m, n = match_pair

                # Lowe Ratio Test.
                if (
                    m.distance
                    < self.ratio_threshold * n.distance
                ):
                    good_matches.append(m)


            # ------------------------------------------------
            # 7. GOOD MATCH COUNT
            # ------------------------------------------------

            good_matches_count = len(
                good_matches
            )

            print(
                "Good matches count:",
                good_matches_count
            )


            # ------------------------------------------------
            # 8. MATCH / NOMATCH
            # ------------------------------------------------

            images_match = (
                good_matches_count
                >= self.good_matches_threshold
            )


            # ------------------------------------------------
            # 9. KEYPOINTS
            # ------------------------------------------------

            all_keypoints_dicts = (
                keypoints1_dicts
                + keypoints2_dicts
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
            # 10. CONNECTIONS
            # ------------------------------------------------

            connections = [
                Connection(
                    p1=m.queryIdx,
                    p2=m.trainIdx + offset
                )
                for m in good_matches
            ]


            # ------------------------------------------------
            # 11. FINAL DETECTION
            # ------------------------------------------------

            self.output_detections = [
                Detection(
                    boundingBox=None,

                    keyPoints=keypoints,

                    connections=connections,

                    # Good match sayisi.
                    confidence=float(
                        good_matches_count
                    ),

                    # Match = 1
                    # NoMatch = 0
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
            # 12. VISUALIZE (opsiyonel)
            # ------------------------------------------------

            if self.visualize:

                img1 = self._decode_image(self.image_1)
                img2 = self._decode_image(self.image_2)

                if img1 is not None and img2 is not None:

                    self.output_visualization = (
                        self._build_visualization(
                            img1,
                            img2,
                            keypoints1_dicts,
                            keypoints2_dicts,
                            good_matches
                        )
                    )

                else:

                    print(
                        "Visualize istendi ama "
                        "InputImage1 / InputImage2 "
                        "saglanmadi ya da decode edilemedi."
                    )


        except Exception as e:

            # Gercek hatayi gizlemiyoruz.
            print(
                "SIFT Comparison Error:",
                repr(e)
            )

            raise


        # ----------------------------------------------------
        # 13. RESPONSE
        # ----------------------------------------------------

        return build_response_sift_comparison_test(
            context=self
        )


# ============================================================
# EXECUTOR
# ============================================================

if __name__ == "__main__":

    Executor(
        sys.argv[1]
    ).run()