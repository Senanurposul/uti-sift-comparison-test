import os
import sys
import json
import base64
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

        # Birinci SIFT output'u.
        self.sift_output_1 = self.request.get_param(
            "InputSIFTOutput1"
        )

        # İkinci SIFT output'u.
        self.sift_output_2 = self.request.get_param(
            "InputSIFTOutput2"
        )

        # ------------------------------------------------
        # Görselleştirme (visualization) parametreleri.
        # ------------------------------------------------

        # Görselleştirmeyi açıp kapatan flag.
        self.enable_visualization = self.request.get_param(
            "EnableVisualization"
        )

        # Keypoint dairelerinin yarıçapı.
        self.point_radius = self.request.get_param(
            "PointRadius"
        )

        # Match çizgilerinin kalınlığı.
        self.line_thickness = self.request.get_param(
            "LineThickness"
        )

        # Görselleştirme için opsiyonel görüntüler.
        # Bu component SIFT hesaplamasını kendisi yapmadığı
        # için görüntüler yalnızca çizim amaçlıdır, matching
        # sonucunu etkilemez.
        self.image_1 = self.request.get_param(
            "InputImage1"
        )

        self.image_2 = self.request.get_param(
            "InputImage2"
        )

        # Görselleştirme çıktısı (base64). Üretilmezse None kalır.
        self.output_visualization = None

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

    # ==================================================
    # GÖRSELLEŞTİRME (VISUALIZATION) YARDIMCI METODLARI
    # ==================================================

    def _decode_base64_image(self, image_value):
        """
        Base64 (veya data-URI önekli base64) string'ini
        OpenCV BGR görüntüsüne çevirir.

        Görüntü çözülemezse None döner (hata fırlatmaz),
        böylece görselleştirme opsiyonel bir adım olarak
        kalır ve ana matching akışını bozmaz.
        """

        if image_value is None:
            return None

        try:

            if isinstance(image_value, str):
                encoded = image_value

                # "data:image/jpeg;base64,...." formatını
                # destekliyoruz.
                if "," in encoded and "base64" in encoded[:30]:
                    encoded = encoded.split(",", 1)[1]

                image_bytes = base64.b64decode(encoded)

            elif isinstance(image_value, (bytes, bytearray)):
                image_bytes = image_value

            else:
                # Beklenmeyen bir tip gelirse görselleştirmeyi
                # sessizce atlıyoruz.
                return None

            np_array = np.frombuffer(
                image_bytes,
                dtype=np.uint8
            )

            image = cv2.imdecode(
                np_array,
                cv2.IMREAD_COLOR
            )

            return image

        except Exception as e:
            print(
                "SIFT Visualization - image decode error:",
                repr(e)
            )
            return None

    def _encode_image_to_base64(self, image, ext=".jpg"):
        """
        OpenCV görüntüsünü base64 string'e çevirir.
        """

        success, buffer = cv2.imencode(ext, image)

        if not success:
            return None

        return base64.b64encode(buffer).decode("utf-8")

    def _stack_images_same_height(self, image_1, image_2):
        """
        İki görüntüyü aynı yüksekliğe getirip yan yana
        (Roboflow tarzı side-by-side) birleştirir.

        Görüntüler farklı yükseklikte ise, kısa olan
        siyah ile alta doldurulur (padding), böylece
        koordinatlar bozulmadan (x ofseti hariç) korunur.
        """

        h1, w1 = image_1.shape[:2]
        h2, w2 = image_2.shape[:2]

        max_height = max(h1, h2)

        # Yükseklik farkını siyah padding ile tamamlıyoruz.
        # Böylece keypoint koordinatları (cx, cy) değişmez.
        if h1 < max_height:
            image_1 = cv2.copyMakeBorder(
                image_1,
                0, max_height - h1,
                0, 0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )

        if h2 < max_height:
            image_2 = cv2.copyMakeBorder(
                image_2,
                0, max_height - h2,
                0, 0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )

        canvas = np.hstack(
            (image_1, image_2)
        )

        # image_2'nin canvas üzerindeki x ofseti.
        offset_x = w1

        return canvas, offset_x

    def _build_visualization(
        self,
        keypoints1_dicts,
        keypoints2_dicts,
        good_matches
    ):
        """
        İki görüntüyü yan yana koyup, good match'leri
        birbirine bağlayan çizgilerle görselleştirir.

        Roboflow'daki "annotated image" mantığına benzer
        şekilde: keypoint'ler daire, eşleşmeler çizgi
        olarak çizilir ve tek bir base64 görsel olarak
        döndürülür.

        Herhangi bir sebeple (görüntü yok, decode hatası
        vb.) görselleştirme üretilemezse None döner;
        bu durum SIFT karşılaştırma sonucunu etkilemez.
        """

        if not self.enable_visualization:
            return None

        if self.image_1 is None or self.image_2 is None:
            print(
                "SIFT Visualization - skipped: "
                "InputImage1/InputImage2 not provided."
            )
            return None

        try:

            image_1 = self._decode_base64_image(self.image_1)
            image_2 = self._decode_base64_image(self.image_2)

            if image_1 is None or image_2 is None:
                print(
                    "SIFT Visualization - skipped: "
                    "image decode failed."
                )
                return None

            canvas, offset_x = self._stack_images_same_height(
                image_1,
                image_2
            )

            point_color = (0, 255, 0)      # yeşil - keypoint
            match_color = (0, 0, 255)      # kırmızı - eşleşme çizgisi

            # ------------------------------------------------
            # Tüm keypoint'leri (good match olsun olmasın)
            # zayıf/ince şekilde işaretliyoruz.
            # ------------------------------------------------

            for kp in keypoints1_dicts:
                x, y = kp["pt"]
                cv2.circle(
                    canvas,
                    (int(x), int(y)),
                    self.point_radius,
                    point_color,
                    1
                )

            for kp in keypoints2_dicts:
                x, y = kp["pt"]
                cv2.circle(
                    canvas,
                    (int(x) + offset_x, int(y)),
                    self.point_radius,
                    point_color,
                    1
                )

            # ------------------------------------------------
            # Good match'leri birbirine bağlayan çizgiler.
            #
            # m.queryIdx -> keypoints1_dicts index'i
            # m.trainIdx -> keypoints2_dicts index'i
            #
            # (Bu indeksler offset eklenmeden önceki,
            # orijinal descriptor sırasına göredir.)
            # ------------------------------------------------

            for m in good_matches:

                x1, y1 = keypoints1_dicts[m.queryIdx]["pt"]
                x2, y2 = keypoints2_dicts[m.trainIdx]["pt"]

                pt1 = (int(x1), int(y1))
                pt2 = (int(x2) + offset_x, int(y2))

                cv2.line(
                    canvas,
                    pt1,
                    pt2,
                    match_color,
                    self.line_thickness
                )

                # Eşleşen noktaları biraz daha belirgin çiziyoruz.
                cv2.circle(canvas, pt1, self.point_radius, match_color, -1)
                cv2.circle(canvas, pt2, self.point_radius, match_color, -1)

            return self._encode_image_to_base64(canvas)

        except Exception as e:

            # Görselleştirme hatası SIFT karşılaştırma
            # sonucunu etkilememeli; sadece logluyoruz.
            print(
                "SIFT Visualization Error:",
                repr(e)
            )
            return None

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

                # Yeterli descriptor yoksa görselleştirilecek
                # anlamlı bir eşleşme de yok.
                self.output_visualization = None

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
            # 8.5 Görselleştirme (opsiyonel).
            #
            # EnableVisualization=True ve InputImage1/2
            # sağlandıysa, good match'leri iki görüntü
            # üzerinde çizip base64 olarak üretiyoruz.
            #
            # Görselleştirme sırasında bir hata olursa
            # ana matching sonucu etkilenmez; sadece
            # output_visualization None kalır.
            # ------------------------------------------------

            self.output_visualization = self._build_visualization(
                keypoints1_dicts,
                keypoints2_dicts,
                good_matches
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