from typing import Any, Literal, Optional, Union
from pydantic import Field
from sdks.novavision.src.base.model import (Package,Image,Inputs,Outputs,Configs,Response,Request,Output,Input,Config,)

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


class InputVisualization1(Input):
    name: Literal["InputVisualization1"] = "InputVisualization1"
    value: Optional[Image] = None
    type: Literal["object"] = "object"

    class Config:
        title = "Visualization 1"


class InputVisualization2(Input):
    name: Literal["InputVisualization2"] = "InputVisualization2"
    value: Optional[Image] = None
    type: Literal["object"] = "object"

    class Config:
        title = "Visualization 2"


class OutputDetections(Output):
    name: Literal["OutputDetections"] = "OutputDetections"
    value: Optional[Any] = None
    type: Literal["list"] = "list"

    class Config:
        title = "Output Detections"


class OutputVisualization1(Output):
    name: Literal["OutputVisualization1"] = "OutputVisualization1"
    value: Optional[Image] = None
    type: Literal["object"] = "object"

    class Config:
        title = "Visualization 1"


class OutputVisualization2(Output):
    name: Literal["OutputVisualization2"] = "OutputVisualization2"
    value: Optional[Image] = None
    type: Literal["object"] = "object"

    class Config:
        title = "Visualization 2"


class OutputVisualizationMatches(Output):
    name: Literal["OutputVisualizationMatches"] = "OutputVisualizationMatches"
    value: Optional[Image] = None
    type: Literal["object"] = "object"

    class Config:
        title = "Visualization Matches"


class GoodMatchesThreshold(Config):
    name: Literal["GoodMatchesThreshold"] = "GoodMatchesThreshold"
    value: int = Field(default=50, ge=1, le=100000)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Good Matches Threshold"
        json_schema_extra = {
            "shortDescription": "Minimum matches to consider a match."
        }


class RatioThreshold(Config):
    name: Literal["RatioThreshold"] = "RatioThreshold"
    value: float = Field(default=0.7, ge=0.0, le=1.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Ratio Threshold"
        json_schema_extra = {
            "shortDescription": "Lowe's ratio test (0.0-1.0)."
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
    value: Union[MatcherFlann, MatcherBF]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Matcher Algorithm"


class VisualizeTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "True"


class VisualizeFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "False"


class Visualize(Config):
    name: Literal["Visualize"] = "Visualize"
    value: Union[VisualizeTrue, VisualizeFalse] = Field(
        default_factory=VisualizeFalse
    )
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Visualize"
        json_schema_extra = {
            "shortDescription": (
                "Whether to generate visualization_1, "
                "visualization_2, and visualization_matches."
            )
        }


class SiftComparisonTestConfigs(Configs):
    GoodMatchesThreshold: GoodMatchesThreshold
    RatioThreshold: RatioThreshold
    Matcher: Matcher
    Visualize: Visualize


class SiftComparisonTestInputs(Inputs):
    InputSIFTOutput1: InputSIFTOutput1
    InputSIFTOutput2: InputSIFTOutput2
    InputVisualization1: InputVisualization1
    InputVisualization2: InputVisualization2


class SiftComparisonTestOutputs(Outputs):
    OutputDetections: OutputDetections
    OutputVisualization1: OutputVisualization1
    OutputVisualization2: OutputVisualization2
    OutputVisualizationMatches: OutputVisualizationMatches


class SiftComparisonTestRequest(Request):
    inputs: Optional[SiftComparisonTestInputs] = None
    configs: SiftComparisonTestConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class SiftComparisonTestResponse(Response):
    outputs: SiftComparisonTestOutputs


class SiftComparisonTest(Config):
    name: Literal["SiftComparisonTest"] = "SiftComparisonTest"
    value: Union[
        SiftComparisonTestRequest,
        SiftComparisonTestResponse,
    ]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "SIFT Comparison Test"
        json_schema_extra = {
            "target": {
                "value": 0
            },
            "shortDescription": "Feature-based image matching.",
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: SiftComparisonTest
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {
            "target": "value"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    name: Literal["SiftComparisonTest"] = "SiftComparisonTest"
    configs: PackageConfigs
    type: Literal["component"] = "component"