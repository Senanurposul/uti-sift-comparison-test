from pydantic import Field, validator
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
# INPUTS
# ============================================================

class InputImageOne(Input):
    name: Literal["inputImageOne"] = "inputImageOne"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        val = values.get("value")
        if isinstance(val, Image):
            return "object"
        elif isinstance(val, list):
            return "list"
        return "object"

    class Config:
        title = "Image Input 1"


class InputImageTwo(Input):
    name: Literal["inputImageTwo"] = "inputImageTwo"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        val = values.get("value")
        if isinstance(val, Image):
            return "object"
        elif isinstance(val, list):
            return "list"
        return "object"

    class Config:
        title = "Image Input 2"


class InputSIFTOutput1(Input):
    name: Literal["inputSIFTOutput1"] = "inputSIFTOutput1"
    value: Optional[Any] = None
    type: Literal["object"] = "object"

    class Config:
        title = "SIFT Output 1"


class InputSIFTOutput2(Input):
    name: Literal["inputSIFTOutput2"] = "inputSIFTOutput2"
    value: Optional[Any] = None
    type: Literal["object"] = "object"

    class Config:
        title = "SIFT Output 2"


# ============================================================
# OUTPUTS
# ============================================================

class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: Optional[Any] = None
    type: Literal["list"] = "list"

    class Config:
        title = "Output Detections"


class OutputMatchesImage(Output):
    name: Literal["outputMatchesImage"] = "outputMatchesImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        val = values.get("value")
        if isinstance(val, Image):
            return "object"
        elif isinstance(val, list):
            return "list"
        return "object"

    class Config:
        title = "Matches Visualization Image"


# ============================================================
# CONFIGS
# ============================================================

class GoodMatchesThreshold(Config):
    name: Literal["goodMatchesThreshold"] = "goodMatchesThreshold"
    value: int = Field(default=50)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Good Matches Threshold"


class RatioThreshold(Config):
    name: Literal["ratioThreshold"] = "ratioThreshold"
    value: float = Field(default=0.7)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Ratio Threshold"


class FlannBasedMatcher(Config):
    name: Literal["FlannBasedMatcher"] = "FlannBasedMatcher"
    value: Literal["FlannBasedMatcher"] = "FlannBasedMatcher"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "FLANN Based Matcher"


class BFMatcher(Config):
    name: Literal["BFMatcher"] = "BFMatcher"
    value: Literal["BFMatcher"] = "BFMatcher"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Brute Force Matcher"


class Matcher(Config):
    name: Literal["matcher"] = "matcher"
    value: Union[FlannBasedMatcher, BFMatcher]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Matcher Algorithm"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


# ============================================================
# EXECUTOR INPUTS / CONFIGS / OUTPUTS
# ============================================================

class SiftComparisonExecutorInputs(Inputs):
    inputImageOne: Optional[InputImageOne] = None
    inputImageTwo: Optional[InputImageTwo] = None
    inputSIFTOutput1: Optional[InputSIFTOutput1] = None
    inputSIFTOutput2: Optional[InputSIFTOutput2] = None


class SiftComparisonExecutorConfigs(Configs):
    goodMatchesThreshold: GoodMatchesThreshold
    ratioThreshold: RatioThreshold
    matcher: Matcher


class SiftComparisonExecutorOutputs(Outputs):
    outputDetections: OutputDetections
    outputMatchesImage: Optional[OutputMatchesImage] = None


class SiftComparisonExecutorRequest(Request):
    inputs: Optional[SiftComparisonExecutorInputs]
    configs: SiftComparisonExecutorConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class SiftComparisonExecutorResponse(Response):
    outputs: SiftComparisonExecutorOutputs


# ============================================================
# EXECUTOR & ROOT PACKAGE
# ============================================================

class SiftComparisonExecutor(Config):
    name: Literal["SiftComparisonExecutor"] = "SiftComparisonExecutor"
    value: Union[SiftComparisonExecutorRequest, SiftComparisonExecutorResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "SIFT Comparison Executor"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[SiftComparisonExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    name: Literal["SiftComparisonTest"] = "SiftComparisonTest"
    configs: PackageConfigs
    type: Literal["component"] = "component"