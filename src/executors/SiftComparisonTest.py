import cv2
import numpy as np

from sdks.novavision.src.base.component import Component


class SiftComparisonTestExecutor(Component):

    def __init__(self, context):
        super().__init__(context)

    def _image_to_numpy(self, image):
        """
        NovaVision Image objesini numpy array'e çevirir.
        """

        if image is None:
            raise ValueError("Image input is None.")

        # numpy array olarak geldiyse
        if isinstance(image, np.ndarray):
            return image

        # bytes olarak geldiyse
        if isinstance(image, bytes):
            array = np.frombuffer(image, dtype=np.uint8)
            decoded = cv2.imdecode(array, cv2.IMREAD_COLOR)

            if decoded is None:
                raise ValueError("Image bytes could not be decoded.")

            return decoded

        # Image objesinin value/data gibi alanlarını kontrol et
        for attribute in ["value", "data", "image", "array"]:

            if hasattr(image, attribute):

                value = getattr(image, attribute)

                if isinstance(value, np.ndarray):
                    return value

                if isinstance(value, bytes):
                    array = np.frombuffer(
                        value,
                        dtype=np.uint8
                    )

                    decoded = cv2.imdecode(
                        array,
                        cv2.IMREAD_COLOR
                    )

                    if decoded is not None:
                        return decoded

        raise TypeError(
            f"Unsupported image type: {type(image)}"
        )

    def _keypoints_to_list(self, keypoints):

        result = []

        for kp in keypoints:

            result.append({
                "cx": float(kp.pt[0]),
                "cy": float(kp.pt[1]),
                "size": float(kp.size),
                "angle": float(kp.angle),
                "response": float(kp.response),
                "octave": int(kp.octave)
            })

        return result

    def _descriptors_to_list(self, descriptors):

        if descriptors is None:
            return []

        return descriptors.astype(
            np.float32
        ).tolist()

    def _create_matcher(self, matcher_name):

        if matcher_name == "BFMatcher":

            return cv2.BFMatcher(
                cv2.NORM_L2,
                crossCheck=False
            )

        # Default = FLANN
        index_params = {
            "algorithm": 1,
            "trees": 5
        }

        search_params = {
            "checks": 50
        }

        return cv2.FlannBasedMatcher(
            index_params,
            search_params
        )

    def run(self):

        try:

            # ------------------------------------------------
            # INPUTS
            # ------------------------------------------------

            input_1 = self.context.inputs.InputSIFTOutput1.value
            input_2 = self.context.inputs.InputSIFTOutput2.value

            if input_1 is None:
                raise ValueError(
                    "InputSIFTOutput1 is empty."
                )

            if input_2 is None:
                raise ValueError(
                    "InputSIFTOutput2 is empty."
                )

            print(
                "SIFT Comparison - received two images."
            )

            # ------------------------------------------------
            # IMAGE -> NUMPY
            # ------------------------------------------------

            image_1 = self._image_to_numpy(
                input_1
            )

            image_2 = self._image_to_numpy(
                input_2
            )

            print(
                "Image 1 shape:",
                image_1.shape
            )

            print(
                "Image 2 shape:",
                image_2.shape
            )

            # ------------------------------------------------
            # GRAYSCALE
            # ------------------------------------------------

            if len(image_1.shape) == 3:
                gray_1 = cv2.cvtColor(
                    image_1,
                    cv2.COLOR_BGR2GRAY
                )
            else:
                gray_1 = image_1

            if len(image_2.shape) == 3:
                gray_2 = cv2.cvtColor(
                    image_2,
                    cv2.COLOR_BGR2GRAY
                )
            else:
                gray_2 = image_2

            # ------------------------------------------------
            # SIFT
            # ------------------------------------------------

            sift = cv2.SIFT_create()

            keypoints_1, descriptors_1 = (
                sift.detectAndCompute(
                    gray_1,
                    None
                )
            )

            keypoints_2, descriptors_2 = (
                sift.detectAndCompute(
                    gray_2,
                    None
                )
            )

            print(
                "Keypoints 1:",
                len(keypoints_1)
            )

            print(
                "Keypoints 2:",
                len(keypoints_2)
            )

            # ------------------------------------------------
            # DESCRIPTOR CHECK
            # ------------------------------------------------

            if descriptors_1 is None:
                descriptors_1 = np.empty(
                    (0, 128),
                    dtype=np.float32
                )

            if descriptors_2 is None:
                descriptors_2 = np.empty(
                    (0, 128),
                    dtype=np.float32
                )

            # ------------------------------------------------
            # CONFIG
            # ------------------------------------------------

            configs = self.context.configs

            good_matches_threshold = (
                configs.GoodMatchesThreshold.value
            )

            ratio_threshold = (
                configs.RatioThreshold.value
            )

            matcher_config = configs.Matcher.value

            matcher_name = matcher_config.name

            print(
                "Matcher:",
                matcher_name
            )

            print(
                "Ratio threshold:",
                ratio_threshold
            )

            print(
                "Good matches threshold:",
                good_matches_threshold
            )

            # ------------------------------------------------
            # MATCHING
            # ------------------------------------------------

            good_matches = []

            if (
                len(descriptors_1) > 0
                and len(descriptors_2) > 0
            ):

                matcher = self._create_matcher(
                    matcher_name
                )

                matches = matcher.knnMatch(
                    descriptors_1,
                    descriptors_2,
                    k=2
                )

                for pair in matches:

                    if len(pair) < 2:
                        continue

                    m, n = pair

                    if (
                        m.distance
                        <
                        ratio_threshold * n.distance
                    ):
                        good_matches.append(m)

            # ------------------------------------------------
            # COMPARISON RESULT
            # ------------------------------------------------

            good_matches_count = len(
                good_matches
            )

            images_match = (
                good_matches_count
                >= good_matches_threshold
            )

            print(
                "Good matches:",
                good_matches_count
            )

            print(
                "Images match:",
                images_match
            )

            # ------------------------------------------------
            # OUTPUT
            # ------------------------------------------------

            output = {

                "keypoints_1":
                    self._keypoints_to_list(
                        keypoints_1
                    ),

                "descriptors_1":
                    self._descriptors_to_list(
                        descriptors_1
                    ),

                "keypoints_2":
                    self._keypoints_to_list(
                        keypoints_2
                    ),

                "descriptors_2":
                    self._descriptors_to_list(
                        descriptors_2
                    ),

                "good_matches_count":
                    good_matches_count,

                "images_match":
                    images_match
            }

            self.context.output_detections = [
                output
            ]

            print(
                "SIFT Comparison output created successfully."
            )

            return self.context

        except Exception as e:

            print(
                "SIFT Comparison Error:",
                repr(e)
            )

            raise