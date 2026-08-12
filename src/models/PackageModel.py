from pydantic import Field, validator
from typing import List, Union, Literal, Optional, Any
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Outputs, Response, Request, Output, Input, Config


# ============ GİRDİLER (INPUTS) — Her iki executor için ortak ============

class InputSIFTOutputOne(Input):
    name: Literal["inputSIFTOutputOne"] = "inputSIFTOutputOne"
    value: Optional[Any]
    type: Literal["object"] = "object"

    class Config:
        title = "SIFT Output 1"


class InputSIFTOutputTwo(Input):
    name: Literal["inputSIFTOutputTwo"] = "inputSIFTOutputTwo"
    value: Optional[Any]
    type: Literal["object"] = "object"

    class Config:
        title = "SIFT Output 2"


# ============ ÇIKTI (OUTPUT) — Her iki executor için ortak ============

class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: Optional[Any]
    type: Literal["list"] = "list"

    class Config:
        title = "Output Detections"


# ============ ORTAK CONFIG ALANLARI ============

class GoodMatchesThresholdField(Config):
    name: Literal["goodMatchesThresholdField"] = "goodMatchesThresholdField"
    value: int = Field(default=50)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Good Matches Threshold"


class RatioThresholdField(Config):
    name: Literal["ratioThresholdField"] = "ratioThresholdField"
    value: float = Field(default=0.7)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Ratio Threshold"


# ============ ADVANCED'A ÖZEL CONFIG ALANLARI ============

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


class MatcherField(Config):
    name: Literal["matcherField"] = "matcherField"
    value: Union[MatcherFlann, MatcherBF]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Matcher Algorithm"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class VisualizeOptionTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class VisualizeOptionFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class VisualizeField(Config):
    name: Literal["visualizeField"] = "visualizeField"
    value: Union[VisualizeOptionTrue, VisualizeOptionFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Visualize"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


# ============ INPUTS (EXECUTOR BAZINDA) ============

class BasicExecutorInputs(Inputs):
    inputSIFTOutputOne: InputSIFTOutputOne
    inputSIFTOutputTwo: InputSIFTOutputTwo


class AdvancedExecutorInputs(Inputs):
    inputSIFTOutputOne: InputSIFTOutputOne
    inputSIFTOutputTwo: InputSIFTOutputTwo


# ============ CONFIGS (EXECUTOR BAZINDA) ============

class BasicExecutorConfigs(Configs):
    goodMatchesThresholdField: GoodMatchesThresholdField
    ratioThresholdField: RatioThresholdField


class AdvancedExecutorConfigs(Configs):
    goodMatchesThresholdField: GoodMatchesThresholdField
    ratioThresholdField: RatioThresholdField
    matcherField: MatcherField
    visualizeField: VisualizeField


# ============ REQUEST (EXECUTOR BAZINDA) ============

class BasicExecutorRequest(Request):
    inputs: Optional[BasicExecutorInputs]
    configs: BasicExecutorConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class AdvancedExecutorRequest(Request):
    inputs: Optional[AdvancedExecutorInputs]
    configs: AdvancedExecutorConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


# ============ OUTPUTS (EXECUTOR BAZINDA) ============

class BasicExecutorOutputs(Outputs):
    outputDetections: OutputDetections


class AdvancedExecutorOutputs(Outputs):
    outputDetections: OutputDetections


# ============ RESPONSE (EXECUTOR BAZINDA) ============

class BasicExecutorResponse(Response):
    outputs: BasicExecutorOutputs


class AdvancedExecutorResponse(Response):
    outputs: AdvancedExecutorOutputs


# ============ EXECUTOR TANIMLARI ============

class SIFTComparisonBasic(Config):
    name: Literal["SIFTComparisonBasic"] = "SIFTComparisonBasic"
    value: Union[BasicExecutorRequest, BasicExecutorResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "SIFT Comparison (Basic)"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class SIFTComparisonAdvanced(Config):
    name: Literal["SIFTComparisonAdvanced"] = "SIFTComparisonAdvanced"
    value: Union[AdvancedExecutorRequest, AdvancedExecutorResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "SIFT Comparison (Advanced)"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[SIFTComparisonBasic, SIFTComparisonAdvanced]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["SiftComparisonTest"] = "SiftComparisonTest"