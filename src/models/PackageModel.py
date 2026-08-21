from typing import Optional, Union, Literal, Any
from pydantic import Field
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
# Roboflow v2: input_1 / input_2 can be an image or
# pre-computed SIFT descriptors.
# ============================================================

class Input1(Input):
    name: Literal["Input1"] = "Input1"
    value: Optional[Union[Image, Any]] = None
    type: Literal["object"] = "object"

    class Config:
        title = "Input 1"


class Input2(Input):
    name: Literal["Input2"] = "Input2"
    value: Optional[Union[Image, Any]] = None
    type: Literal["object"] = "object"

    class Config:
        title = "Input 2"


# ============================================================
# OUTPUTS
# ============================================================

class ImagesMatch(Output):
    name: Literal["ImagesMatch"] = "ImagesMatch"
    value: Optional[bool] = None
    type: Literal["boolean"] = "boolean"

    class Config:
        title = "Images Match"


class GoodMatchesCount(Output):
    name: Literal["GoodMatchesCount"] = "GoodMatchesCount"
    value: Optional[int] = None
    type: Literal["number"] = "number"

    class Config:
        title = "Good Matches Count"


class KeyPoints1(Output):
    name: Literal["KeyPoints1"] = "KeyPoints1"
    value: Optional[Any] = None
    type: Literal["list"] = "list"

    class Config:
        title = "Keypoints 1"


class Descriptors1(Output):
    name: Literal["Descriptors1"] = "Descriptors1"
    value: Optional[Any] = None
    type: Literal["object"] = "object"

    class Config:
        title = "Descriptors 1"


class KeyPoints2(Output):
    name: Literal["KeyPoints2"] = "KeyPoints2"
    value: Optional[Any] = None
    type: Literal["list"] = "list"

    class Config:
        title = "Keypoints 2"


class Descriptors2(Output):
    name: Literal["Descriptors2"] = "Descriptors2"
    value: Optional[Any] = None
    type: Literal["object"] = "object"

    class Config:
        title = "Descriptors 2"


class Visualization1(Output):
    name: Literal["Visualization1"] = "Visualization1"
    value: Optional[Image] = None
    type: Literal["object"] = "object"

    class Config:
        title = "Visualization 1"


class Visualization2(Output):
    name: Literal["Visualization2"] = "Visualization2"
    value: Optional[Image] = None
    type: Literal["object"] = "object"

    class Config:
        title = "Visualization 2"


class VisualizationMatches(Output):
    name: Literal["VisualizationMatches"] = "VisualizationMatches"
    value: Optional[Image] = None
    type: Literal["object"] = "object"

    class Config:
        title = "Visualization Matches"


# ============================================================
# CONFIGS
# ============================================================

class GoodMatchesThreshold(Config):
    name: Literal["GoodMatchesThreshold"] = "GoodMatchesThreshold"
    value: int = Field(default=50, ge=1, le=100000)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Good Matches Threshold"


class RatioThreshold(Config):
    name: Literal["RatioThreshold"] = "RatioThreshold"
    value: float = Field(default=0.7, ge=0.0, le=1.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Ratio Threshold"


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
    value: Union[MatcherFlann, MatcherBF]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Matcher Algorithm"


class Visualize(Config):
    name: Literal["Visualize"] = "Visualize"
    value: bool = False
    type: Literal["boolean"] = "boolean"
    field: Literal["checkbox"] = "checkbox"

    class Config:
        title = "Visualize"


class SiftComparisonTestConfigs(Configs):
    GoodMatchesThreshold: GoodMatchesThreshold
    RatioThreshold: RatioThreshold
    Matcher: Matcher
    Visualize: Visualize


class SiftComparisonTestInputs(Inputs):
    Input1: Input1
    Input2: Input2


class SiftComparisonTestOutputs(Outputs):
    ImagesMatch: ImagesMatch
    GoodMatchesCount: GoodMatchesCount
    KeyPoints1: KeyPoints1
    Descriptors1: Descriptors1
    KeyPoints2: KeyPoints2
    Descriptors2: Descriptors2
    Visualization1: Visualization1
    Visualization2: Visualization2
    VisualizationMatches: VisualizationMatches


class SiftComparisonTestRequest(Request):
    inputs: Optional[SiftComparisonTestInputs] = None
    configs: SiftComparisonTestConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class SiftComparisonTestResponse(Response):
    outputs: SiftComparisonTestOutputs


class SiftComparisonTest(Config):
    name: Literal["SiftComparisonTest"] = "SiftComparisonTest"
    value: Union[SiftComparisonTestRequest, SiftComparisonTestResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "SIFT Comparison"


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: SiftComparisonTest
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {"target": "value"}


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    name: Literal["SiftComparisonTest"] = "SiftComparisonTest"
    configs: PackageConfigs
    type: Literal["component"] = "component"