from pydantic import validator
from typing import List, Optional, Union, Literal, Any

from sdks.novavision.src.base.model import (
    Package,
    Image,
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


# ============================================================
# OUTPUTS
# ============================================================

class OutputDetections(Output):
    name: Literal["OutputDetections"] = "OutputDetections"
    value: Optional[Any]
    type: Literal["list"] = "list"

    class Config:
        title = "Output Detections"


# ============================================================
# CONFIGS
# ============================================================

class GoodMatchesThreshold(Config):
    """
    Minimum number of good feature matches required to
    consider the two images as matching.

    Lower values are more lenient.
    Higher values are stricter.

    Default: 50
    """

    name: Literal["GoodMatchesThreshold"] = "GoodMatchesThreshold"
    value: int = 50
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Good Matches Threshold"
        json_schema_extra = {
            "shortDescription": "Min matches to consider a match"
        }


class RatioThreshold(Config):
    """
    Threshold used by Lowe's ratio test.

    Lower values are stricter.
    Higher values are more lenient.

    Default: 0.7
    """

    name: Literal["RatioThreshold"] = "RatioThreshold"
    value: float = 0.7
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Ratio Threshold"
        json_schema_extra = {
            "shortDescription": "Lowe's ratio test (0.0-1.0)"
        }


class MatcherFlann(Config):
    """
    FLANN performs efficient approximate nearest-neighbor
    search for SIFT descriptors.
    """

    name: Literal["FlannBasedMatcher"] = "FlannBasedMatcher"
    value: Literal["FlannBasedMatcher"] = "FlannBasedMatcher"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "FLANN Based Matcher"
        json_schema_extra = {
            "shortDescription": "Approximate nearest-neighbor matching"
        }


class MatcherBF(Config):
    """
    BFMatcher performs brute-force nearest-neighbor matching
    using the L2 distance for SIFT descriptors.
    """

    name: Literal["BFMatcher"] = "BFMatcher"
    value: Literal["BFMatcher"] = "BFMatcher"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Brute Force Matcher"
        json_schema_extra = {
            "shortDescription": "Brute-force exact matching"
        }


class Matcher(Config):
    """
    Selects the matcher algorithm used to compare
    SIFT descriptors.

    FlannBasedMatcher:
        Approximate nearest-neighbor search.

    BFMatcher:
        Brute-force nearest-neighbor search.
    """

    name: Literal["Matcher"] = "Matcher"
    value: Union[MatcherFlann, MatcherBF]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Matcher Algorithm"
        json_schema_extra = {
            "shortDescription": "FLANN or Brute Force"
        }


class SiftComparisonTestConfigs(Configs):
    GoodMatchesThreshold: GoodMatchesThreshold
    RatioThreshold: RatioThreshold
    Matcher: Matcher


# ============================================================
# INPUT / OUTPUT MODELS
# ============================================================

class SiftComparisonTestInputs(Inputs):
    InputSIFTOutput1: InputSIFTOutput1
    InputSIFTOutput2: InputSIFTOutput2


class SiftComparisonTestOutputs(Outputs):
    OutputDetections: OutputDetections


class SiftComparisonTestRequest(Request):
    inputs: Optional[SiftComparisonTestInputs]
    configs: SiftComparisonTestConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class SiftComparisonTestResponse(Response):
    outputs: SiftComparisonTestOutputs


# ============================================================
# EXECUTOR CONFIGURATION
# ============================================================

class SiftComparisonTest(Config):
    """
    Compares two images using SIFT descriptors received
    from external SIFT blocks.

    The package does not calculate SIFT features itself.
    It only performs descriptor matching.

    Output:
        - Keypoints
        - Connections
        - Good match count as confidence
        - Match / NoMatch classification
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
            "target": {"value": 0},
            "shortDescription": "Feature-based image matching"
        }


class ConfigExecutor(Config):
    """
    SIFT Comparison executor.

    Compares SIFT descriptors using either FLANN
    or Brute Force matching.
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


class PackageConfigs(Configs):
    executor: ConfigExecutor


# ============================================================
# PACKAGE MODEL
# ============================================================

class PackageModel(Package):
    name: Literal["SiftComparisonTest"] = "SiftComparisonTest"
    configs: PackageConfigs
    type: Literal["component"] = "component"