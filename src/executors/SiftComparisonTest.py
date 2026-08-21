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
        # Değer:
        # "FlannBasedMatcher"
        # veya
        # "BFMatcher"
        self.matcher = self.request.get_param(
            "Matcher"
        )

        # Visualization input images
        self.visualization_input_1 = self.request.get_param(
            "InputVisualization1"
        )

        self.visualization_input_2 = self.request.get_param(
            "InputVisualization2"
        )

        self.output_visualization = None

        # Birinci SIFT output'u.
        self.sift_output_1 = self.request.get_param(
            "InputSIFTOutput1"
        )

        # İkinci SIFT output'u.
        self.sift_output_2 = self.request.get_param(
            "InputSIFTOutput2"
        )

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

        # Input JSON string olarak geldiyse Python
        # objesine dönüştürüyoruz.
        if isinstance(sift_output, str):
            sift_output = json.loads(sift_output)

        # SIFT output'unun liste olması gerekiyor.
        if not isinstance(sift_output, list):
            raise ValueError(
                "SIFT output must be a list."
            )

        # Her detection içerisindeki keypoint'leri
        # ve descriptor'ları topluyoruz.
        for detection in sift_output:

            for kp in detection.get("keyPoints", []):

                # Descriptor yoksa bu keypoint'i
                # matching işlemine dahil etmiyoruz.
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

        # Hiç descriptor bulunmadıysa boş descriptor
        # matrisi döndürüyoruz.
        if not descriptors:
            return (
                keypoints_dicts,
                np.empty(
                    (0, 128),
                    dtype=np.float32
                )
            )

        # OpenCV matcher'ları için descriptor'ların
        # float32 olması gerekiyor.
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

    def _create_visualization(
        self,
        frame1,
        frame2,
        keypoints1_dicts,
        keypoints2_dicts,
        good_matches,
    ):
        if frame1 is None or frame2 is None:
            raise ValueError("Visualization frame is None")

        frame1 = np.asarray(frame1)
        frame2 = np.asarray(frame2)

        if frame1.size == 0 or frame2.size == 0:
            raise ValueError("Visualization frame is empty")

        if frame1.dtype != np.uint8:
            frame1 = frame1.astype(np.uint8)

        if frame2.dtype != np.uint8:
            frame2 = frame2.astype(np.uint8)

        if len(frame1.shape) == 2:
            frame1 = cv2.cvtColor(
                frame1,
                cv2.COLOR_GRAY2BGR
            )

        if len(frame2.shape) == 2:
            frame2 = cv2.cvtColor(
                frame2,
                cv2.COLOR_GRAY2BGR
            )

        cv_keypoints1 = [
            cv2.KeyPoint(
                float(kp["pt"][0]),
                float(kp["pt"][1]),
                1.0
            )
            for kp in keypoints1_dicts
        ]

        cv_keypoints2 = [
            cv2.KeyPoint(
                float(kp["pt"][0]),
                float(kp["pt"][1]),
                1.0
            )
            for kp in keypoints2_dicts
        ]

        # Roboflow-style visualization:
        # draw all good matches that passed Lowe's Ratio Test.
        matches_to_draw = list(good_matches)

        print(
            "SIFTCOMPARISONTEST - VISUALIZATION: AUTOMATIC",
            flush=True
        )

        print(
            "SIFTCOMPARISONTEST - TOTAL GOOD MATCHES:",
            len(good_matches),
            flush=True
        )

        print(
            "SIFTCOMPARISONTEST - MATCHES TO DRAW:",
            len(matches_to_draw),
            flush=True
        )

        return cv2.drawMatches(
            frame1,
            cv_keypoints1,
            frame2,
            cv_keypoints2,
            matches_to_draw,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

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
            # 2. knnMatch(k=2) kullanacağımız için
            # iki görüntüde de en az 2 descriptor
            # bulunması gerekiyor.
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
            #
            # Burada FLANN ve BFMatcher'ı açıkça
            # birbirinden ayırıyoruz.
            # ------------------------------------------------

            if self.matcher == "FlannBasedMatcher":

                # SIFT descriptor'ları float olduğu için
                # FLANN'de KD-Tree kullanıyoruz.
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

                # BFMatcher bütün descriptor'ları
                # brute-force olarak karşılaştırır.
                #
                # SIFT descriptor'ları için L2 norm kullanılır.
                matcher = cv2.BFMatcher(
                    cv2.NORM_L2,
                    crossCheck=False
                )

            else:

                # Desteklenmeyen bir matcher gelirse
                # sessizce FLANN kullanmak yerine hata veriyoruz.
                raise ValueError(
                    f"Unsupported matcher: {self.matcher}"
                )

            # ------------------------------------------------
            # 4. Her descriptor için en yakın 2 descriptor'ı
            # buluyoruz.
            #
            # k=2 -> Lowe Ratio Test için gerekli.
            # ------------------------------------------------

            matches = matcher.knnMatch(
                descriptors1,
                descriptors2,
                k=2
            )

            # Lowe Ratio Test'i geçen eşleşmeler.
            good_matches = []

            for match_pair in matches:

                # İki eşleşme yoksa ratio testi
                # yapılamaz.
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
            # 6. İki görüntünün keypoint'lerini
            # tek listede birleştiriyoruz.
            # ------------------------------------------------

            all_keypoints_dicts = (
                keypoints1_dicts
                + keypoints2_dicts
            )

            # Image 2 keypoint'lerinin başladığı index.
            offset = len(
                keypoints1_dicts
            )

            # Novavision KeyPoints modeline dönüştürüyoruz.
            keypoints = [
                KeyPoints(
                    cx=float(kp["pt"][0]),
                    cy=float(kp["pt"][1]),
                    confidence=1.0
                )
                for kp in all_keypoints_dicts
            ]

            # ------------------------------------------------
            # 7. Good match'leri Connection nesnelerine
            # dönüştürüyoruz.
            #
            # queryIdx -> image 1
            # trainIdx -> image 2
            #
            # Image 2 keypoint'leri listenin devamında
            # olduğu için offset ekliyoruz.
            # ------------------------------------------------

            connections = [
                Connection(
                    p1=m.queryIdx,
                    p2=m.trainIdx + offset
                )
                for m in good_matches
            ]

            # ------------------------------------------------
            # 8. Final Detection oluşturuluyor.
            # ------------------------------------------------

            self.output_detections = [
                Detection(
                    boundingBox=None,

                    # İki görüntünün keypoint'leri.
                    keyPoints=keypoints,

                    # Good match bağlantıları.
                    connections=connections,

                    # Good match sayısını confidence
                    # olarak kullanıyoruz.
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
            # 9. Visualization
            # ------------------------------------------------
            # Matching logic above is kept unchanged.
            # Visualization is created only from the already
            # calculated good_matches.
            # ------------------------------------------------

            if (
                self.visualization_input_1 is not None
                and self.visualization_input_2 is not None
            ):
                image1_frame = Image.get_frame(
                    img=self.visualization_input_1,
                    redis_db=self.redis_db
                )

                image2_frame = Image.get_frame(
                    img=self.visualization_input_2,
                    redis_db=self.redis_db
                )

                if image1_frame is not None and image2_frame is not None:
                    visualization = self._create_visualization(
                        image1_frame.value,
                        image2_frame.value,
                        keypoints1_dicts,
                        keypoints2_dicts,
                        good_matches
                    )

                    image1_frame.value = visualization

                    self.output_visualization = Image.set_frame(
                        img=image1_frame,
                        package_uID=self.uID,
                        redis_db=self.redis_db
                    )

        except Exception as e:

            # Gerçek bir kod hatasını NoMatch olarak
            # gizlemek yerine terminal/log'a yazıyoruz.
            print(
                "SIFT Comparison Error:",
                repr(e)
            )

            raise

        # ------------------------------------------------
        # 9. Response oluşturuluyor.
        # ------------------------------------------------

        return build_response_sift_comparison_test(
            context=self
        )


# Executor'ın başlatılması.
if __name__ == "__main__":
    Executor(
        sys.argv[1]
    ).run()