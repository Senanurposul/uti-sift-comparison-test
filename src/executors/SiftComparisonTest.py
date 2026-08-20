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

from sdks.novavision.src.base.component import Component
from sdks.novavision.src.base.model import (
    KeyPoints,
    Detection,
    Connection
)
from sdks.novavision.src.helper.executor import Executor
from sdks.novavision.src.media.image import Image

from components.SiftComparisonTest.src.utils.response import (
    build_response_sift_comparison_test
)

from components.SiftComparisonTest.src.models.PackageModel import (
    PackageModel
)


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
    # SIFT OUTPUT PARSE
    # ========================================================

    def _extract_keypoints_and_descriptors(
        self,
        sift_output
    ):

        keypoints_dicts = []
        descriptors = []

        if isinstance(sift_output, str):
            sift_output = json.loads(
                sift_output
            )

        if not isinstance(
            sift_output,
            list
        ):
            raise ValueError(
                "SIFT output must be a list."
            )

        for detection in sift_output:

            for kp in detection.get(
                "keyPoints",
                []
            ):

                if "descriptor" not in kp:
                    continue

                keypoints_dicts.append(
                    {
                        "pt": (
                            float(kp["cx"]),
                            float(kp["cy"])
                        ),
                        "size": float(
                            kp.get(
                                "size",
                                1.0
                            )
                        ),
                        "angle": float(
                            kp.get(
                                "angle",
                                -1
                            )
                        )
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


    # ========================================================
    # IMAGE VISUALIZATION
    # ========================================================

    def _create_keypoints(
        self,
        keypoints_dicts
    ):

        keypoints = []

        for kp in keypoints_dicts:

            keypoints.append(
                cv2.KeyPoint(
                    float(kp["pt"][0]),
                    float(kp["pt"][1]),
                    float(kp["size"]),
                    float(kp["angle"])
                )
            )

        return keypoints


    def _create_visualization(
        self,
        image,
        keypoints
    ):

        if image is None:
            raise ValueError(
                "Image is None."
            )

        return cv2.drawKeypoints(
            image,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
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

            # ------------------------------------------------
            # 1. SIFT OUTPUT
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
            # 2. ORIGINAL IMAGES
            # ------------------------------------------------

            img1 = Image.get_frame(
                img=self.input_image_1,
                redis_db=self.redis_db
            )

            img2 = Image.get_frame(
                img=self.input_image_2,
                redis_db=self.redis_db
            )

            image1 = img1.value.copy()
            image2 = img2.value.copy()


            # ------------------------------------------------
            # 3. KEYPOINT VISUALIZATION
            # ------------------------------------------------

            cv_keypoints_1 = self._create_keypoints(
                keypoints1_dicts
            )

            cv_keypoints_2 = self._create_keypoints(
                keypoints2_dicts
            )

            visualization_1 = self._create_visualization(
                image1,
                cv_keypoints_1
            )

            visualization_2 = self._create_visualization(
                image2,
                cv_keypoints_2
            )


            # ------------------------------------------------
            # 4. MATCHER
            # ------------------------------------------------

            if (
                len(descriptors1) < 2
                or len(descriptors2) < 2
            ):

                self.output_detections = (
                    self._no_match_result()
                )

                # Empty match visualization
                visualization_matches = (
                    np.hstack(
                        (
                            visualization_1,
                            visualization_2
                        )
                    )
                )

            else:

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
                # 5. MATCH
                # ------------------------------------------------

                matches = matcher.knnMatch(
                    descriptors1,
                    descriptors2,
                    k=2
                )

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


                # ------------------------------------------------
                # 6. RESULT
                # ------------------------------------------------

                good_matches_count = len(
                    good_matches
                )

                images_match = (
                    good_matches_count
                    >= self.good_matches_threshold
                )


                # ------------------------------------------------
                # 7. NOVAVISION KEYPOINTS
                # ------------------------------------------------

                all_keypoints_dicts = (
                    keypoints1_dicts
                    +
                    keypoints2_dicts
                )

                offset = len(
                    keypoints1_dicts
                )

                keypoints = [

                    KeyPoints(
                        cx=float(
                            kp["pt"][0]
                        ),
                        cy=float(
                            kp["pt"][1]
                        ),
                        confidence=1.0
                    )

                    for kp in all_keypoints_dicts
                ]


                # ------------------------------------------------
                # 8. CONNECTIONS
                # ------------------------------------------------

                connections = [

                    Connection(
                        p1=m.queryIdx,
                        p2=m.trainIdx + offset
                    )

                    for m in good_matches
                ]


                # ------------------------------------------------
                # 9. DETECTION
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
                # 10. MATCH VISUALIZATION
                # ------------------------------------------------

                cv_matches = cv2.drawMatches(

                    image1,

                    cv_keypoints_1,

                    image2,

                    cv_keypoints_2,

                    good_matches,

                    None,

                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )

                visualization_matches = cv_matches


            # ------------------------------------------------
            # 11. SAVE VISUALIZATION IMAGES
            # ------------------------------------------------

            img1.value = visualization_1

            img2.value = visualization_2


            self.visualization_1 = Image.set_frame(
                img=img1,
                package_uID=self.uID,
                redis_db=self.redis_db
            )


            self.visualization_2 = Image.set_frame(
                img=img2,
                package_uID=self.uID,
                redis_db=self.redis_db
            )


            # ------------------------------------------------
            # MATCH VISUALIZATION IMAGE
            # ------------------------------------------------

            match_img = Image(
                value=visualization_matches
            )

            self.visualization_matches = Image.set_frame(
                img=match_img,
                package_uID=self.uID,
                redis_db=self.redis_db
            )


            # ------------------------------------------------
            # 12. RESPONSE
            # ------------------------------------------------

            return build_response_sift_comparison_test(
                context=self
            )


        except Exception as e:

            print(
                "SIFT Comparison Error:",
                repr(e)
            )

            raise


# ============================================================
# EXECUTOR
# ============================================================

if __name__ == "__main__":

    Executor(
        sys.argv[1]
    ).run()