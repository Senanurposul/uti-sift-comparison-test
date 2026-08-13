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

        # Gelen request içerisindeki parametreleri
        # PackageModel'e göre doğruluyoruz.
        self.request.model = PackageModel(
            **self.request.data
        )

        # Eşleşmenin Match kabul edilmesi için
        # gereken minimum good match sayısı.
        self.good_matches_threshold = self.request.get_param(
            "GoodMatchesThreshold"
        )

        # Lowe's Ratio Test için kullanılan oran.
        # Örneğin 0.7 ise:
        # m.distance < 0.7 * n.distance
        # koşulunu sağlayan eşleşmeler good match kabul edilir.
        self.ratio_threshold = self.request.get_param(
            "RatioThreshold"
        )

        # Kullanılacak matcher:
        # BFMatcher veya FLANN.
        self.matcher = self.request.get_param(
            "Matcher"
        )

        # Birinci SIFT paketinden gelen output.
        self.sift_output_1 = self.request.get_param(
            "InputSIFTOutput1"
        )

        # İkinci SIFT paketinden gelen output.
        self.sift_output_2 = self.request.get_param(
            "InputSIFTOutput2"
        )

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _extract_keypoints_and_descriptors(self, sift_output):
        """
        SIFT paketinin output'undan:

        1. Keypoint koordinatlarını
        2. Descriptor'ları

        ayırır.

        Comparison algoritması descriptor'lar üzerinden
        çalışır.
        """

        keypoints_dicts = []
        descriptors = []

        # Bazı durumlarda input JSON string olarak gelebilir.
        # Böyle bir durumda Python listesine çeviriyoruz.
        if isinstance(sift_output, str):
            sift_output = json.loads(sift_output)

        # SIFT output'unun liste olması bekleniyor.
        if not isinstance(sift_output, list):
            raise ValueError(
                "SIFT output must be a list."
            )

        # SIFT output yapısı:
        #
        # [
        #   {
        #       "keyPoints": [
        #           {
        #               "cx": ...,
        #               "cy": ...,
        #               "descriptor": [...]
        #           }
        #       ]
        #   }
        # ]
        #
        # Burada bütün keypoint ve descriptor'ları topluyoruz.
        for detection in sift_output:

            for kp in detection.get("keyPoints", []):

                # Descriptor yoksa matching yapılamaz.
                if "descriptor" not in kp:
                    continue

                # Output'ta göstereceğimiz keypoint bilgisi.
                keypoints_dicts.append(
                    {
                        "pt": (
                            float(kp["cx"]),
                            float(kp["cy"])
                        )
                    }
                )

                # SIFT descriptor'ını listeye ekliyoruz.
                descriptors.append(
                    kp["descriptor"]
                )

        # Hiç descriptor bulunamadıysa boş bir descriptor matrisi
        # döndürüyoruz.
        if not descriptors:
            return (
                keypoints_dicts,
                np.empty(
                    (0, 128),
                    dtype=np.float32
                )
            )

        # OpenCV matcher'larının beklediği veri tipi float32.
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
        Yeterli descriptor veya match bulunmadığında
        kullanılacak standart NoMatch sonucu.
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

            # -------------------------------------------------
            # 1. İki SIFT output'unu ayırıyoruz.
            # -------------------------------------------------

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

            # -------------------------------------------------
            # 2. Descriptor sayısını kontrol ediyoruz.
            #
            # knnMatch(k=2) kullanacağımız için iki tarafta da
            # en az 2 descriptor olması gerekiyor.
            # -------------------------------------------------

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

            # -------------------------------------------------
            # 3. Kullanıcının seçtiği matcher oluşturuluyor.
            # -------------------------------------------------

            if self.matcher == "BFMatcher":

                # SIFT descriptor'ları için L2 norm kullanılır.
                matcher = cv2.BFMatcher(
                    cv2.NORM_L2,
                    crossCheck=False
                )

            else:

                # FLANN için KD-Tree kullanıyoruz.
                index_params = {
                    "algorithm": 1,
                    "trees": 5
                }

                # checks değeri aramanın ne kadar kapsamlı
                # yapılacağını belirler.
                search_params = {
                    "checks": 50
                }

                matcher = cv2.FlannBasedMatcher(
                    index_params,
                    search_params
                )

            # -------------------------------------------------
            # 4. Her descriptor için en yakın 2 descriptor'ı
            # buluyoruz.
            #
            # k=2 olmasının sebebi Lowe Ratio Test'tir.
            # -------------------------------------------------

            matches = matcher.knnMatch(
                descriptors1,
                descriptors2,
                k=2
            )

            # Lowe Ratio Test'i geçen eşleşmeler burada tutulur.
            good_matches = []

            for match_pair in matches:

                # Bazı durumlarda 2 eşleşme gelmeyebilir.
                # Böyle bir durumda bu match grubunu atlıyoruz.
                if len(match_pair) < 2:
                    continue

                m, n = match_pair

                # Lowe's Ratio Test.
                #
                # m = en iyi eşleşme
                # n = ikinci en iyi eşleşme
                #
                # Birinci eşleşme ikinciye göre yeterince iyiyse
                # good match olarak kabul ediyoruz.
                if (
                    m.distance
                    < self.ratio_threshold * n.distance
                ):
                    good_matches.append(m)

            # -------------------------------------------------
            # 5. Good match sayısını hesaplıyoruz.
            # -------------------------------------------------

            good_matches_count = len(
                good_matches
            )

            # Belirlediğimiz threshold'un üzerinde
            # yeterli eşleşme varsa görüntüler Match kabul edilir.
            images_match = (
                good_matches_count
                >= self.good_matches_threshold
            )

            # -------------------------------------------------
            # 6. İki görüntünün keypoint'lerini tek listede
            # birleştiriyoruz.
            #
            # Önce image 1'in keypoint'leri,
            # sonra image 2'nin keypoint'leri bulunuyor.
            # -------------------------------------------------

            all_keypoints_dicts = (
                keypoints1_dicts
                + keypoints2_dicts
            )

            # Image 2 keypoint indexlerinin başladığı nokta.
            offset = len(
                keypoints1_dicts
            )

            # Novavision'ın KeyPoints modeline dönüştürüyoruz.
            keypoints = [
                KeyPoints(
                    cx=float(kp["pt"][0]),
                    cy=float(kp["pt"][1]),
                    confidence=1.0
                )
                for kp in all_keypoints_dicts
            ]

            # -------------------------------------------------
            # 7. Good match'leri Connection'a dönüştürüyoruz.
            #
            # queryIdx → image 1
            # trainIdx → image 2
            #
            # Image 2'nin keypoint'leri listenin devamında
            # olduğu için offset ekliyoruz.
            # -------------------------------------------------

            connections = [
                Connection(
                    p1=m.queryIdx,
                    p2=m.trainIdx + offset
                )
                for m in good_matches
            ]

            # -------------------------------------------------
            # 8. Final Detection output'u oluşturuyoruz.
            # -------------------------------------------------

            self.output_detections = [
                Detection(
                    boundingBox=None,

                    # İki görüntünün bütün keypoint'leri.
                    keyPoints=keypoints,

                    # Lowe Ratio Test'i geçen eşleşmeler.
                    connections=connections,

                    # Confidence burada good match sayısını
                    # temsil ediyor.
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

                    # Kullanıcıya okunabilir sonuç.
                    classLabel=(
                        "Match"
                        if images_match
                        else "NoMatch"
                    ),

                    imgUID=self.uID
                )
            ]

        except Exception as e:

            # Gerçek bir kod hatasını sessizce NoMatch'e
            # çevirmek yerine hatayı terminal/log tarafında
            # görebilmek için tekrar raise ediyoruz.
            print(
                "SIFT Comparison Error:",
                repr(e)
            )

            raise

        # -------------------------------------------------
        # 9. Novavision response modelimizi oluşturuyoruz.
        # -------------------------------------------------

        return build_response_sift_comparison_test(
            context=self
        )


# Executor'ın çalıştırılması.
if __name__ == "__main__":
    Executor(
        sys.argv[1]
    ).run()