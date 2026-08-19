from typing import Optional, Union, Literal, Any

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
    value: Optional[Any] = None
    type: Literal["object"] = "object"

    class Config:
        title = "SIFT Output 1"


class InputSIFTOutput2(Input):
    name: Literal["InputSIFTOutput2"] = "InputSIFTOutput2"
    value: Optional[Any] = None
    type: Literal["object"] = "object"

    class Config:
        title = "SIFT Output 2"


class InputImage1(Input):
    name: Literal["InputImage1"] = "InputImage1"
    value: Optional[Image] = None
    type: Literal["image"] = "image"

    class Config:
        title = "Image 1"


class InputImage2(Input):
    name: Literal["InputImage2"] = "InputImage2"
    value: Optional[Image] = None
    type: Literal["image"] = "image"

    class Config:
        title = "Image 2"


# ============================================================
# OUTPUTS
# ============================================================

class OutputDetections(Output):
    name: Literal["OutputDetections"] = "OutputDetections"
    value: Optional[Any] = None
    type: Literal["list"] = "list"

    class Config:
        title = "Output Detections"


class OutputMatchesImage(Output):
    name: Literal["OutputMatchesImage"] = "OutputMatchesImage"
    value: Optional[Image] = None
    type: Literal["image"] = "image"

    class Config:
        title = "Matches Visualization Image"


# ============================================================
# CONFIGS
# ============================================================

class GoodMatchesThreshold(Config):
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
    name: Literal["RatioThreshold"] = "RatioThreshold"
    value: float = 0.7
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Ratio Threshold"
        json_schema_extra = {
            "shortDescription": "Lowe's ratio test threshold"
        }


# ============================================================
# MATCHER OPTIONS
# ============================================================

class MatcherFlann(Config):
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
    name: Literal["Matcher"] = "Matcher"
    value: Union[MatcherFlann, MatcherBF]
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


# ============================================================
# INPUT / OUTPUT CONTAINER MODELS
# ============================================================

class SiftComparisonTestInputs(Inputs):
    InputImage1: Optional[InputImage1] = None
    InputImage2: Optional[InputImage2] = None
    InputSIFTOutput1: Optional[InputSIFTOutput1] = None
    InputSIFTOutput2: Optional[InputSIFTOutput2] = None


class SiftComparisonTestOutputs(Outputs):
    OutputDetections: OutputDetections
    OutputMatchesImage: Optional[OutputMatchesImage] = None


# ============================================================
# REQUEST & RESPONSE
# ============================================================

class SiftComparisonTestRequest(Request):
    inputs: Optional[SiftComparisonTestInputs] = None
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
# PACKAGE ROOT MODEL
# ============================================================

class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    name: Literal["SiftComparisonTest"] = "SiftComparisonTest"
    inputs: Optional[SiftComparisonTestInputs] = None
    configs: PackageConfigs
    type: Literal["component"] = "component"