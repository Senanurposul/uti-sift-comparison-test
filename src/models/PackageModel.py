from pydantic import validator
from typing import List, Union, Literal, Optional, Any

from sdks.novavision.src.base.model import (
    Package,
    Image,
    Inputs,
    Configs,
    Outputs,
    Response,
    Request,
    Output,
    Input,
    Config
)


# ============================================================
# VISUALIZATION IMAGE INPUT 1
# ============================================================

class InputVisualization1(Input):
    name: Literal["InputVisualization1"] = "InputVisualization1"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get("value")

        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

        return "object"

    class Config:
        title = "Visualization Image 1"


# ============================================================
# VISUALIZATION IMAGE INPUT 2
# ============================================================

class InputVisualization2(Input):
    name: Literal["InputVisualization2"] = "InputVisualization2"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get("value")

        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

        return "object"

    class Config:
        title = "Visualization Image 2"


# ============================================================
# SIFT OUTPUT INPUT 1
# ============================================================

class InputSIFTOutput1(Input):
    name: Literal["InputSIFTOutput1"] = "InputSIFTOutput1"
    value: Optional[Any]
    type: Literal["object"] = "object"

    class Config:
        title = "SIFT Output 1"


# ============================================================
# SIFT OUTPUT INPUT 2
# ============================================================

class InputSIFTOutput2(Input):
    name: Literal["InputSIFTOutput2"] = "InputSIFTOutput2"
    value: Optional[Any]
    type: Literal["object"] = "object"

    class Config:
        title = "SIFT Output 2"


# ============================================================
# DETECTION OUTPUT
# ============================================================

class OutputDetections(Output):
    name: Literal["OutputDetections"] = "OutputDetections"
    value: Optional[Any]
    type: Literal["list"] = "list"

    class Config:
        title = "Output Detections"


# ============================================================
# VISUALIZATION OUTPUT
# ============================================================

class OutputVisualization(Output):
    name: Literal["OutputVisualization"] = "OutputVisualization"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get("value")

        if isinstance(value, Image):
            return "object"
        elif isinstance(value, list):
            return "list"

        return "object"

    class Config:
        title = "Visualization"


# ============================================================
# GOOD MATCHES THRESHOLD
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


# ============================================================
# RATIO THRESHOLD
# ============================================================

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
# FLANN MATCHER
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


# ============================================================
# BF MATCHER
# ============================================================

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


# ============================================================
# MATCHER
# ============================================================

class Matcher(Config):
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


# ============================================================
# INPUT MODEL
# ============================================================

class SiftComparisonTestInputs(Inputs):
    InputVisualization1: InputVisualization1
    InputVisualization2: InputVisualization2
    InputSIFTOutput1: InputSIFTOutput1
    InputSIFTOutput2: InputSIFTOutput2


# ============================================================
# OUTPUT MODEL
# ============================================================

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
# EXECUTOR CONFIG
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


# ============================================================
# CONFIG EXECUTOR
# ============================================================

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