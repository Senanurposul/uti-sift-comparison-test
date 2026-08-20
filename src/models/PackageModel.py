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
    Config,
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


class InputVisualization1(Input):
    name: Literal["InputVisualization1"] = "InputVisualization1"
    value: Optional[Image]
    type: Literal["object"] = "object"

    class Config:
        title = "Visualization 1"


class InputVisualization2(Input):
    name: Literal["InputVisualization2"] = "InputVisualization2"
    value: Optional[Image]
    type: Literal["object"] = "object"

    class Config:
        title = "Visualization 2"


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
    value: Optional[Image]
    type: Literal["object"] = "object"

    class Config:
        title = "Output Visualization"


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
            "shortDescription": "Min matches to consider a match"
        }


class RatioThreshold(Config):
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


class SIFTComparisonConfigs(Configs):
    GoodMatchesThreshold: GoodMatchesThreshold
    RatioThreshold: RatioThreshold
    Matcher: Matcher


# ============================================================
# INPUT / OUTPUT MODELS
# ============================================================

class SIFTComparisonInputs(Inputs):
    InputVisualization1: InputVisualization1
    InputSIFTOutput1: InputSIFTOutput1
    InputSIFTOutput2: InputSIFTOutput2
    InputVisualization2: InputVisualization2


class SIFTComparisonOutputs(Outputs):
    OutputDetections: OutputDetections
    OutputVisualization: OutputVisualization


class SIFTComparisonRequest(Request):
    inputs: Optional[SIFTComparisonInputs]
    configs: SIFTComparisonConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class SIFTComparisonResponse(Response):
    outputs: SIFTComparisonOutputs


# ============================================================
# EXECUTOR
# ============================================================

class SIFTComparison(Config):

    name: Literal["SIFTComparison"] = "SIFTComparison"

    value: Union[
        SIFTComparisonRequest,
        SIFTComparisonResponse
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

    value: Union[SIFTComparison]

    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"

        json_schema_extra = {
            "target": "value"
        }


# ============================================================
# PACKAGE
# ============================================================

class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):

    name: Literal["SIFTComparison"] = "SIFTComparison"

    configs: PackageConfigs

    type: Literal["component"] = "component"