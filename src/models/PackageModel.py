from pydantic import validator
from typing import List, Union, Literal, Optional, Any

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


class InputImage1(Input):
    name: Literal["InputImage1"] = "InputImage1"
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
        title = "Image 1"


class InputImage2(Input):
    name: Literal["InputImage2"] = "InputImage2"
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
        title = "Image 2"


# ============================================================
# OUTPUTS
# ============================================================

class OutputDetections(Output):
    name: Literal["OutputDetections"] = "OutputDetections"
    value: Optional[Any]
    type: Literal["list"] = "list"

    class Config:
        title = "Output Detections"


class Visualization1(Output):
    name: Literal["Visualization1"] = "Visualization1"
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
        title = "Visualization 1"


class Visualization2(Output):
    name: Literal["Visualization2"] = "Visualization2"
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
        title = "Visualization 2"


class VisualizationMatches(Output):
    name: Literal["VisualizationMatches"] = "VisualizationMatches"
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
        title = "Visualization Matches"


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


class MatcherFlann(Config):
    name: Literal["FlannBasedMatcher"] = "FlannBasedMatcher"
    value: Literal["FlannBasedMatcher"] = "FlannBasedMatcher"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "FLANN Based Matcher"


class MatcherBF(Config):
    name: Literal["BFMatcher"] = "BFMatcher"
    value: Literal["BFMatcher"] = "BFMatcher"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Brute Force Matcher"


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
    InputImage1: InputImage1
    InputImage2: InputImage2


class SiftComparisonTestOutputs(Outputs):
    OutputDetections: OutputDetections
    Visualization1: Visualization1
    Visualization2: Visualization2
    VisualizationMatches: VisualizationMatches


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