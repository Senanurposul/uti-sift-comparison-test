import os
import sys
import json
import cv2
import numpy as np

# Projenin root dizinine ulaşabilmek için path ekleniyor.
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        '../../../../'
    )
)

from sdks.novavision.src.media.image import Image as MediaImage
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

        # Gelen request'i PackageModel ile doğruluyoruz.
        self.request.model = PackageModel(
            **self.request.data
        )

        # Match kabul edilmesi için gereken
        # minimum good match sayısı.
        self.good_matches_threshold = self.request.get_param(
            "GoodMatchesThreshold"
        )

        # Lowe Ratio Test threshold değeri.
        self.ratio_threshold = self.request.get_param(
            "RatioThreshold"
        )

        # Kullanılacak matcher.
        self.matcher = self.request.get_param(
            "Matcher"
        )

        # Birinci SIFT output'u.
        self.sift_output_1 = self.request.get_param(
            "InputSIFTOutput1"
        )

        # İkinci SIFT output'u.
        self.sift_output_2 = self.request.get_param(
            "InputSIFTOutput2"
        )

        # Görselleştirme için orijinal görüntüler (Opsiyonel)
        self.image_1_raw = self.request.get_param(
            "InputImage1"
        )
        self.image_2_raw = self.request.get_param(
            "InputImage2"
        )

        self.output_detections = []
        self.output_matches_image = None

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _extract_keypoints_and_descriptors(self, sift_output):
        """
        SIFT output'undan keypoint koordinatlarını
        ve descriptor'ları ayırır.
        """
        keypoints_dicts = []
        descriptors = []

        if isinstance(sift_output, str):
            sift_output = json.loads(sift_output)

        if not isinstance(sift_output, list):
            raise ValueError(
                "SIFT output must be a list."
            )

        for detection in sift_output:
            kps = detection.get("keyPoints", []) or detection.get("keypoints", [])
            for kp in kps:
                if "descriptor" not in kp:
                    continue

                cx = kp.get("cx", kp.get("pt", [0, 0])[0] if isinstance(kp.get("pt"), (list, tuple)) else kp.get("x", 0.0))
                cy = kp.get("cy", kp.get("pt", [0, 0])[1] if isinstance(kp.get("pt"), (list, tuple)) else kp.get("y", 0.0))

                keypoints_dicts.append(
                    {
                        "pt": (
                            float(cx),
                            float(cy)
                        ),
                        "size": float(kp.get("size", 1.0)),
                        "angle": float(kp.get("angle", -1.0))
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

        descriptors = np.asarray(
            descriptors,
            dtype=np.float32
        )

        return (
            keypoints_dicts,
            descriptors
        )

    def _no_match_result(self):
        """
        Yeterli descriptor bulunmadığında
        kullanılacak NoMatch sonucu.
        """
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
            # ------------------------------------------------
            # 1. SIFT output'larını ayırıyoruz.
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
            # 2. knnMatch(k=2) kontrolü
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
            # 3. Matcher oluşturuluyor.
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
            # 4. KNN Matching
            # ------------------------------------------------
            matches = matcher.knnMatch(
                descriptors1,
                descriptors2,
                k=2
            )

            # Lowe Ratio Test'i geçen eşleşmeler.
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
            # 5. Good match sayısını hesaplıyoruz.
            # ------------------------------------------------
            good_matches_count = len(
                good_matches
            )

            # Threshold'a ulaşıldıysa Match.
            images_match = (
                good_matches_count
                >= self.good_matches_threshold
            )

            # ------------------------------------------------
            # 6. Keypoints birleştirme
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
            # 7. Connections (Bağlantılar)
            # ------------------------------------------------
            connections = [
                Connection(
                    p1=m.queryIdx,
                    p2=m.trainIdx + offset
                )
                for m in good_matches
            ]

            # ------------------------------------------------
            # 8. Final Detection
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
            # 9. GÖRSELLEŞTİRME (ROBOFLOW MATCH VISUALIZATION)
            # ------------------------------------------------
            if self.image_1_raw and self.image_2_raw:
                frame_1 = MediaImage.get_frame(img=self.image_1_raw, redis_db=self.redis_db)
                frame_2 = MediaImage.get_frame(img=self.image_2_raw, redis_db=self.redis_db)

                if frame_1 is not None and frame_2 is not None:
                    cv2_kp1 = [
                        cv2.KeyPoint(
                            x=kp["pt"][0],
                            y=kp["pt"][1],
                            size=kp.get("size", 1.0),
                            angle=kp.get("angle", -1.0)
                        )
                        for kp in keypoints1_dicts
                    ]
                    cv2_kp2 = [
                        cv2.KeyPoint(
                            x=kp["pt"][0],
                            y=kp["pt"][1],
                            size=kp.get("size", 1.0),
                            angle=kp.get("angle", -1.0)
                        )
                        for kp in keypoints2_dicts
                    ]

                    # Roboflow mantığı: İki resmi yan yana birleştirip yeşil eşleşme çizgilerini çizer
                    drawn_matches = cv2.drawMatches(
                        frame_1.value, cv2_kp1,
                        frame_2.value, cv2_kp2,
                        good_matches,
                        None,
                        matchColor=(0, 255, 0),
                        singlePointColor=(0, 0, 255),
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )

                    frame_1.value = drawn_matches
                    self.output_matches_image = MediaImage.set_frame(
                        img=frame_1,
                        package_uID=self.uID,
                        redis_db=self.redis_db
                    )

        except Exception as e:
            print(
                "SIFT Comparison Error:",
                repr(e)
            )
            raise

        # ------------------------------------------------
        # 10. Response oluşturuluyor.
        # ------------------------------------------------
        return build_response_sift_comparison_test(
            context=self
        )


# Executor'ın başlatılması.
if __name__ == "__main__":
    Executor(
        sys.argv[1]
    ).run()