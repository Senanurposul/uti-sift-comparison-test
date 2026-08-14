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

        # ----------------------------------------------------
        # Request model
        # ----------------------------------------------------

        self.request.model = PackageModel(
            **self.request.data
        )

        # ----------------------------------------------------
        # Configs
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
        # SIFT outputs
        # ----------------------------------------------------

        self.sift_output_1 = self.request.get_param(
            "InputSIFTOutput1"
        )

        self.sift_output_2 = self.request.get_param(
            "InputSIFTOutput2"
        )

        # ----------------------------------------------------
        # Original images for visualization
        # ----------------------------------------------------

        self.input_image_1 = self.request.get_param(
            "InputImage1"
        )

        self.input_image_2 = self.request.get_param(
            "InputImage2"
        )

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

        keypoints_dicts = []
        descriptors = []

        # SIFT output JSON string olarak geldiyse
        # Python objesine çeviriyoruz.
        if isinstance(sift_output, str):
            sift_output = json.loads(sift_output)

        # Bazı durumlarda output doğrudan
        # {"value": [...]} şeklinde gelebilir.
        if isinstance(sift_output, dict):
            sift_output = sift_output.get(
                "value",
                []
            )

        if not isinstance(sift_output, list):
            raise ValueError(
                "SIFT output must be a list."
            )

        # Detection'ların içindeki keypoint
        # ve descriptor bilgilerini topluyoruz.
        for detection in sift_output:

            for kp in detection.get(
                "keyPoints",
                []
            ):

                # Descriptor yoksa bu keypoint'i
                # matching işlemine almıyoruz.
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

        # Descriptor bulunamadıysa boş matris.
        if not descriptors:
            return (
                keypoints_dicts,
                np.empty(
                    (0, 128),
                    dtype=np.float32
                )
            )

        descriptors = np.asarray(
            descriptors,
            dtype=np.float32
        )

        return (
            keypoints_dicts,
            descriptors
        )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    def _create_visualization(
        self,
        image1,
        image2,
        keypoints1_dicts,
        keypoints2_dicts,
        good_matches
    ):
        """
        İki görüntüyü yan yana getirir ve
        good match olan keypoint'leri çizgilerle bağlar.
        """

        # SIFT output'undan gelen koordinatları
        # OpenCV KeyPoint nesnelerine çeviriyoruz.
        cv_keypoints1 = [
            cv2.KeyPoint(
                float(kp["pt"][0]),
                float(kp["pt"][1]),
                1
            )
            for kp in keypoints1_dicts
        ]

        cv_keypoints2 = [
            cv2.KeyPoint(
                float(kp["pt"][0]),
                float(kp["pt"][1]),
                1
            )
            for kp in keypoints2_dicts
        ]

        # Good match'leri iki görüntü üzerinde çiziyoruz.
        visualization = cv2.drawMatches(
            image1,
            cv_keypoints1,
            image2,
            cv_keypoints2,
            good_matches,
            None
        )

        return visualization

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

            # ------------------------------------------------
            # 1. Görüntüleri Novavision Image modelinden al
            # ------------------------------------------------

            image1 = Image.get_frame(
                img=self.input_image_1,
                redis_db=self.redis_db
            )

            image2 = Image.get_frame(
                img=self.input_image_2,
                redis_db=self.redis_db
            )

            image1_frame = image1.value
            image2_frame = image2.value

            # ------------------------------------------------
            # 2. SIFT output'larını ayır
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
            # 3. Descriptor sayısı kontrolü
            # ------------------------------------------------

            if (
                len(descriptors1) < 2
                or len(descriptors2) < 2
            ):

                # Eşleşme yoksa boş match listesiyle
                # visualization oluşturuyoruz.
                visualization = self._create_visualization(
                    image1_frame,
                    image2_frame,
                    keypoints1_dicts,
                    keypoints2_dicts,
                    []
                )

                image1.value = visualization

                self.output_visualization = Image.set_frame(
                    img=image1,
                    package_uID=self.uID,
                    redis_db=self.redis_db
                )

                self.output_detections = (
                    self._no_match_result()
                )

                return build_response_sift_comparison_test(
                    context=self
                )

            # ------------------------------------------------
            # 4. Matcher oluştur
            # ------------------------------------------------

            if self.matcher == "FlannBasedMatcher":

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

                matcher = cv2.BFMatcher(
                    cv2.NORM_L2,
                    crossCheck=False
                )

            else:

                raise ValueError(
                    f"Unsupported matcher: {self.matcher}"
                )

            # ------------------------------------------------
            # 5. KNN matching
            # ------------------------------------------------

            matches = matcher.knnMatch(
                descriptors1,
                descriptors2,
                k=2
            )

            # ------------------------------------------------
            # 6. Lowe Ratio Test
            # ------------------------------------------------

            good_matches = []

            for match_pair in matches:

                if len(match_pair) < 2:
                    continue

                m, n = match_pair

                if (
                    m.distance
                    < self.ratio_threshold * n.distance
                ):
                    good_matches.append(m)

            # ------------------------------------------------
            # 7. Good match sayısı
            # ------------------------------------------------

            good_matches_count = len(
                good_matches
            )

            images_match = (
                good_matches_count
                >= self.good_matches_threshold
            )

            # ------------------------------------------------
            # 8. Keypoint output
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
            # 9. Connection output
            # ------------------------------------------------

            connections = [
                Connection(
                    p1=m.queryIdx,
                    p2=m.trainIdx + offset
                )
                for m in good_matches
            ]

            # ------------------------------------------------
            # 10. Detection output
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
            # 11. Visualization oluştur
            # ------------------------------------------------

            visualization = self._create_visualization(
                image1_frame,
                image2_frame,
                keypoints1_dicts,
                keypoints2_dicts,
                good_matches
            )

            # ------------------------------------------------
            # 12. Visualization'ı Image output'a çevir
            # ------------------------------------------------

            image1.value = visualization

            self.output_visualization = Image.set_frame(
                img=image1,
                package_uID=self.uID,
                redis_db=self.redis_db
            )

        except Exception as e:

            print(
                "SIFT Comparison Error:",
                repr(e)
            )

            raise

        # ----------------------------------------------------
        # 13. Response
        # ----------------------------------------------------

        return build_response_sift_comparison_test(
            context=self
        )


# ============================================================
# EXECUTOR
# ============================================================

if "__name__" == "__main__":
    Executor(
        sys.argv[1]
    ).run()