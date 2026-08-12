from pydantic import validator
from typing import List, Optional, Union, Literal, Any
from sdks.novavision.src.base.model import (
    Package, Image, Inputs, Outputs, Configs, Response, Request, Output, Input, Config
)


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


class OutputDetections(Output):
    name: Literal["OutputDetections"] = "OutputDetections"
    value: Optional[Any]
    type: Literal["list"] = "list"

    class Config:
        title = "Output Detections"


class GoodMatchesThreshold(Config):
    """
    Minimum number of good feature matches required to consider the two images as matching.
    Lower values (e.g. 20-30) are more lenient. Higher values (e.g. 80-100) are stricter.
    Default is 50.
    """
    name: Literal["GoodMatchesThreshold"] = "GoodMatchesThreshold"
    value: int = 50
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Good Matches Threshold"
        json_schema_extra = {"shortDescription": "Min matches to consider a match"}


class RatioThreshold(Config):
    """
    Ratio threshold for Lowe's ratio test (0.0-1.0).
    Lower values (e.g. 0.6) are stricter. Higher values (e.g. 0.8) are more lenient.
    Default is 0.7.
    """
    name: Literal["RatioThreshold"] = "RatioThreshold"
    value: float = 0.7
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Ratio Threshold"
        json_schema_extra = {"shortDescription": "Lowe's ratio test (0.0-1.0)"}


class MatcherFlann(Config):
    """
    FlannBasedMatcher uses FLANN for efficient approximate nearest neighbor search.
    Faster for large descriptor sets. Recommended for most use cases.
    """
    name: Literal["FlannBasedMatcher"] = "FlannBasedMatcher"
    value: Literal["FlannBasedMatcher"] = "FlannBasedMatcher"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "FLANN Based Matcher"
        json_schema_extra = {"shortDescription": "Fast, approximate matching"}


class MatcherBF(Config):
    """
    BFMatcher uses brute force matching with L2 norm.
    Exact matching but slower for large descriptor sets.
    """
    name: Literal["BFMatcher"] = "BFMatcher"
    value: Literal["BFMatcher"] = "BFMatcher"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Brute Force Matcher"
        json_schema_extra = {"shortDescription": "Exact, slower matching"}


class Matcher(Config):
    """
    Select the matcher algorithm to use for comparing SIFT descriptors.
    FlannBasedMatcher is faster. BFMatcher is more exact but slower.
    """
    name: Literal["Matcher"] = "Matcher"
    value: Union[MatcherFlann, MatcherBF]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Matcher Algorithm"
        json_schema_extra = {"shortDescription": "FLANN or Brute Force"}


class SiftComparisonTestConfigs(Configs):
    GoodMatchesThreshold: GoodMatchesThreshold
    RatioThreshold: RatioThreshold
    Matcher: Matcher


class SiftComparisonTestInputs(Inputs):
    InputSIFTOutput1: InputSIFTOutput1
    InputSIFTOutput2: InputSIFTOutput2


class SiftComparisonTestOutputs(Outputs):
    OutputDetections: OutputDetections


class SiftComparisonTestRequest(Request):
    inputs: Optional[SiftComparisonTestInputs]
    configs: SiftComparisonTestConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class SiftComparisonTestResponse(Response):
    outputs: SiftComparisonTestOutputs


class SiftComparisonTest(Config):
    """
    Compares two images using SIFT descriptors from an external SIFT block.
    Takes SIFT outputs as inputs.
    Outputs a single OutputDetections containing keypoints, connections,
    goodMatchesCount as confidence, and classLabel Match or NoMatch.
    No repetition of tasks already done by the SIFT block.
    """
    name: Literal["SiftComparisonTest"] = "SiftComparisonTest"
    value: Union[SiftComparisonTestRequest, SiftComparisonTestResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "SIFT Comparison"
        json_schema_extra = {"target": {"value": 0}, "shortDescription": "Feature-based image matching"}


class ConfigExecutor(Config):
    """
    SIFT Comparison — compares two images using SIFT descriptors from an external SIFT block.
    No API key required. Runs entirely on the local machine using OpenCV.
    """
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[SiftComparisonTest]
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