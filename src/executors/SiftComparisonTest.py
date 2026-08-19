import os
import sys
import cv2
import numpy as np

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

        super().__init__(
            request,
            bootstrap
        )

        self.request.model = PackageModel(
            **self.request.data
        )

        # ====================================================
        # CONFIGS
        # ====================================================

        self.good_matches_threshold = (
            self.request.get_param(
                "GoodMatchesThreshold"
            )
        )

        self.ratio_threshold = (
            self.request.get_param(
                "RatioThreshold"
            )
        )

        self.matcher = (
            self.request.get_param(
                "Matcher"
            )
        )

        # ====================================================
        # INPUTS
        # ====================================================
        # İsimleri değiştirmiyoruz.
        # Ancak artık bunlar SIFT output değil,
        # doğrudan IMAGE alıyor.

        self.sift_output_1 = (
            self.request.get_param(
                "InputSIFTOutput1"
            )
        )

        self.sift_output_2 = (
            self.request.get_param(
                "InputSIFTOutput2"
            )
        )

    # ========================================================
    # BOOTSTRAP
    # ========================================================

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    # ========================================================
    # IMAGE -> NUMPY
    # ========================================================

    def _get_numpy_image(self, image):

        if image is None:
            raise ValueError(
                "Input image is None."
            )

        # ----------------------------------------------------
        # Direkt numpy array
        # ----------------------------------------------------

        if isinstance(
            image,
            np.ndarray
        ):
            return image

        # ----------------------------------------------------
        # Image objesinin value alanı
        # ----------------------------------------------------

        if hasattr(
            image,
            "value"
        ):

            value = image.value

            if isinstance(
                value,
                np.ndarray
            ):
                return value

            if isinstance(
                value,
                bytes
            ):

                encoded = np.frombuffer(
                    value,
                    dtype=np.uint8
                )

                decoded = cv2.imdecode(
                    encoded,
                    cv2.IMREAD_COLOR
                )

                if decoded is not None:
                    return decoded

        # ----------------------------------------------------
        # Bytes
        # ----------------------------------------------------

        if isinstance(
            image,
            bytes
        ):

            encoded = np.frombuffer(
                image,
                dtype=np.uint8
            )

            decoded = cv2.imdecode(
                encoded,
                cv2.IMREAD_COLOR
            )

            if decoded is not None:
                return decoded

        raise TypeError(
            f"Unsupported image type: {type(image)}"
        )

    # ========================================================
    # SIFT
    # ========================================================

    def _calculate_sift(
        self,
        image
    ):

        sift = cv2.SIFT_create()

        keypoints, descriptors = (
            sift.detectAndCompute(
                image,
                None
            )
        )

        if descriptors is None:

            descriptors = np.empty(
                (0, 128),
                dtype=np.float32
            )

        else:

            descriptors = np.asarray(
                descriptors,
                dtype=np.float32
            )

        return (
            keypoints,
            descriptors
        )

    # ========================================================
    # NO MATCH
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

            print("")
            print("========================================")
            print(" SIFT COMPARISON EXECUTOR STARTED")
            print("========================================")

            # =================================================
            # 1. INPUT IMAGE'LARI AL
            # =================================================

            image1 = self._get_numpy_image(
                self.sift_output_1
            )

            image2 = self._get_numpy_image(
                self.sift_output_2
            )

            print(
                "Image 1 received:",
                image1.shape
            )

            print(
                "Image 2 received:",
                image2.shape
            )

            # =================================================
            # 2. GRAYSCALE
            # =================================================

            if len(image1.shape) == 3:

                gray1 = cv2.cvtColor(
                    image1,
                    cv2.COLOR_BGR2GRAY
                )

            else:

                gray1 = image1

            if len(image2.shape) == 3:

                gray2 = cv2.cvtColor(
                    image2,
                    cv2.COLOR_BGR2GRAY
                )

            else:

                gray2 = image2

            # =================================================
            # 3. SIFT HESAPLA
            # =================================================

            keypoints1, descriptors1 = (
                self._calculate_sift(
                    gray1
                )
            )

            keypoints2, descriptors2 = (
                self._calculate_sift(
                    gray2
                )
            )

            print(
                "Keypoints 1:",
                len(keypoints1)
            )

            print(
                "Keypoints 2:",
                len(keypoints2)
            )

            print(
                "Descriptors 1:",
                len(descriptors1)
            )

            print(
                "Descriptors 2:",
                len(descriptors2)
            )

            # =================================================
            # 4. DESCRIPTOR KONTROLÜ
            # =================================================

            if (
                len(descriptors1) < 2
                or
                len(descriptors2) < 2
            ):

                print(
                    "Not enough descriptors for matching."
                )

                self.output_detections = (
                    self._no_match_result()
                )

                return build_response_sift_comparison_test(
                    context=self
                )

            # =================================================
            # 5. MATCHER
            # =================================================

            if (
                self.matcher
                == "FlannBasedMatcher"
            ):

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

            elif (
                self.matcher
                == "BFMatcher"
            ):

                matcher = cv2.BFMatcher(
                    cv2.NORM_L2,
                    crossCheck=False
                )

            else:

                raise ValueError(
                    f"Unsupported matcher: {self.matcher}"
                )

            # =================================================
            # 6. KNN MATCHING
            # =================================================

            matches = matcher.knnMatch(
                descriptors1,
                descriptors2,
                k=2
            )

            # =================================================
            # 7. LOWE RATIO TEST
            # =================================================

            good_matches = []

            for match_pair in matches:

                if len(match_pair) < 2:
                    continue

                m, n = match_pair

                if (
                    m.distance
                    <
                    self.ratio_threshold * n.distance
                ):

                    good_matches.append(m)

            # =================================================
            # 8. GOOD MATCH COUNT
            # =================================================

            good_matches_count = len(
                good_matches
            )

            print(
                "Good matches count:",
                good_matches_count
            )

            # =================================================
            # 9. MATCH / NO MATCH
            # =================================================

            images_match = (
                good_matches_count
                >= self.good_matches_threshold
            )

            print(
                "Images match:",
                images_match
            )

            # =================================================
            # 10. KEYPOINTS
            # =================================================

            keypoints = []

            for kp in keypoints1:

                keypoints.append(
                    KeyPoints(
                        cx=float(kp.pt[0]),
                        cy=float(kp.pt[1]),
                        confidence=1.0
                    )
                )

            offset = len(keypoints)

            for kp in keypoints2:

                keypoints.append(
                    KeyPoints(
                        cx=float(kp.pt[0]),
                        cy=float(kp.pt[1]),
                        confidence=1.0
                    )
                )

            # =================================================
            # 11. CONNECTIONS
            # =================================================

            connections = []

            for match in good_matches:

                connections.append(
                    Connection(
                        p1=match.queryIdx,
                        p2=match.trainIdx + offset
                    )
                )

            # =================================================
            # 12. OUTPUT
            # =================================================

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

            print("")
            print("========================================")
            print(" SIFT COMPARISON OUTPUT CREATED")
            print("========================================")

            # =================================================
            # 13. RESPONSE
            # =================================================

            return build_response_sift_comparison_test(
                context=self
            )

        except Exception as e:

            print("")
            print("========================================")
            print(" SIFT COMPARISON ERROR")
            print("========================================")

            print(
                repr(e)
            )

            print("========================================")
            print("")

            raise


# ============================================================
# EXECUTOR
# ============================================================

if __name__ == "__main__":

    Executor(
        sys.argv[1]
    ).run()