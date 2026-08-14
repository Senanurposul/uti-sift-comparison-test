from typing import Optional, Union, Literal, Any

from sdks.novavision.src.base.model import (
    Package,
    Inputs,
    Outputs,
    Configs,
    Response,
    Request,
    Output,
    Input,
    Config
)


# ============================================================
# INPUTS
# ============================================================

class InputSIFTOutput1(Input):
    name: Literal["InputSIFTOutput1"] = "InputSIFTOutput1"
    value: Optional[Any]
    type: Literal["object"] = "object"

    class Config:
        title = "SIFT Output 1"


class InputSIFTOutput2(Input):
    name: Literal["InputSIFTOutput2"] = "InputSIFTOutput2"
    value: Optional[Any]
    type: Literal["object"] = "object"

    class Config:
        title = "SIFT Output 2"


class InputImage1(Input):
    name: Literal["InputImage1"] = "InputImage1"
    value: Optional[Any]
    type: Literal["object"] = "object"

    class Config:
        title = "Image 1 (Visualize icin opsiyonel)"


class InputImage2(Input):
    name: Literal["InputImage2"] = "InputImage2"
    value: Optional[Any]
    type: Literal["object"] = "object"

    class Config:
        title = "Image 2 (Visualize icin opsiyonel)"


# ============================================================
# OUTPUTS
# ============================================================

class OutputDetections(Output):
    name: Literal["OutputDetections"] = "OutputDetections"
    value: Optional[Any]
    type: Literal["list"] = "list"

    class Config:
        title = "Output Detections"


class OutputVisualization(Output):
    name: Literal["OutputVisualization"] = "OutputVisualization"
    value: Optional[Any]
    type: Literal["string"] = "string"

    class Config:
        title = "Visualization Image"


# ============================================================
# CONFIGS
# ============================================================

class GoodMatchesThreshold(Config):
    """
    Minimum number of good matches required
    to classify two images as Match.
    """

    name: Literal["GoodMatchesThreshold"] = "GoodMatchesThreshold"
    value: int = 50
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Good Matches Threshold"
        json_schema_extra = {
            "shortDescription": "Minimum good matches required"
        }


class RatioThreshold(Config):
    """
    Lowe's Ratio Test threshold.

    Lower value:
        More strict matching.

    Higher value:
        More tolerant matching.
    """

    name: Literal["RatioThreshold"] = "RatioThreshold"
    value: float = 0.7
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Ratio Threshold"
        json_schema_extra = {
            "shortDescription": "Lowe's ratio test threshold"
        }


class Visualize(Config):
    """
    Eslesen keypoint'leri orijinal goruntuler
    uzerinde gorsellestiren bir cikti uretir.

    True ise InputImage1 / InputImage2
    saglanmis olmalidir.
    """

    name: Literal["Visualize"] = "Visualize"
    value: bool = False
    type: Literal["boolean"] = "boolean"
    field: Literal["checkbox"] = "checkbox"

    class Config:
        title = "Visualize"
        json_schema_extra = {
            "shortDescription": "Eslesmeleri gorsellestir"
        }


# ============================================================
# MATCHER OPTIONS
# ============================================================

class MatcherFlann(Config):
    """
    FLANN matcher for SIFT descriptors.
    """

    name: Literal["FlannBasedMatcher"] = "FlannBasedMatcher"
    value: Literal["FlannBasedMatcher"] = "FlannBasedMatcher"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "FLANN Based Matcher"
        json_schema_extra = {
            "shortDescription": "FLANN nearest-neighbor matching"
        }


class MatcherBF(Config):
    """
    Brute Force matcher for SIFT descriptors.
    """

    name: Literal["BFMatcher"] = "BFMatcher"
    value: Literal["BFMatcher"] = "BFMatcher"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Brute Force Matcher"
        json_schema_extra = {
            "shortDescription": "Brute-force descriptor matching"
        }


class Matcher(Config):
    """
    Select the descriptor matching algorithm.

    Options:
        - FlannBasedMatcher
        - BFMatcher
    """

    name: Literal["Matcher"] = "Matcher"

    value: Union[
        MatcherFlann,
        MatcherBF
    ]

    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Matcher Algorithm"
        json_schema_extra = {
            "shortDescription": "Select FLANN or Brute Force"
        }


# ============================================================
# CONFIG MODEL
# ============================================================

class SiftComparisonTestConfigs(Configs):
    GoodMatchesThreshold: GoodMatchesThreshold
    RatioThreshold: RatioThreshold
    Matcher: Matcher
    Visualize: Visualize


# ============================================================
# INPUT / OUTPUT MODELS
# ============================================================

class SiftComparisonTestInputs(Inputs):
    InputSIFTOutput1: InputSIFTOutput1
    InputSIFTOutput2: InputSIFTOutput2
    InputImage1: Optional[InputImage1]
    InputImage2: Optional[InputImage2]


class SiftComparisonTestOutputs(Outputs):
    OutputDetections: OutputDetections
    OutputVisualization: OutputVisualization


# ============================================================
# REQUEST
# ============================================================

class SiftComparisonTestRequest(Request):
    inputs: Optional[SiftComparisonTestInputs]
    configs: SiftComparisonTestConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


# ============================================================
# RESPONSE
# ============================================================

class SiftComparisonTestResponse(Response):
    outputs: SiftComparisonTestOutputs


# ============================================================
# EXECUTOR CONFIGURATION
# ============================================================

class SiftComparisonTest(Config):
    """
    Compares two images using SIFT descriptors
    received from external SIFT blocks.

    The package itself does not calculate SIFT features
    unless an image is additionally provided for
    visualization purposes.
    """

    name: Literal["SiftComparisonTest"] = "SiftComparisonTest"

    value: Union[
        SiftComparisonTestRequest,
        SiftComparisonTestResponse
    ]

    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "SIFT Comparison"
        json_schema_extra = {
            "target": {
                "value": 0
            },
            "shortDescription": "Feature-based image matching"
        }


class ConfigExecutor(Config):
    """
    SIFT Comparison executor.

    Compares SIFT descriptors using either
    FLANN or Brute Force matching.
    """

    name: Literal["ConfigExecutor"] = "ConfigExecutor"

    value: Union[SiftComparisonTest]

    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {
            "target": "value"
        }


# ============================================================
# PACKAGE CONFIG
# ============================================================

class PackageConfigs(Configs):
    executor: ConfigExecutor


# ============================================================
# PACKAGE MODEL
# ============================================================

class PackageModel(Package):
    name: Literal["SiftComparisonTest"] = "SiftComparisonTest"
    configs: PackageConfigs
    type: Literal["component"] = "component"