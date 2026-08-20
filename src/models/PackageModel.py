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


class InputImage1(Input):
    """
    Görselleştirme için opsiyonel birinci görüntü.

    Value: base64 encoded image string.

    EnableVisualization=True olduğunda kullanılır.
    Verilmezse görselleştirme adımı atlanır,
    SIFT karşılaştırma sonucu etkilenmez.
    """

    name: Literal["InputImage1"] = "InputImage1"
    value: Optional[Any] = None
    type: Literal["object"] = "object"

    class Config:
        title = "Image 1 (Visualization)"
        json_schema_extra = {
            "shortDescription": "Optional image for visualization"
        }


class InputImage2(Input):
    """
    Görselleştirme için opsiyonel ikinci görüntü.

    Value: base64 encoded image string.
    """

    name: Literal["InputImage2"] = "InputImage2"
    value: Optional[Any] = None
    type: Literal["object"] = "object"

    class Config:
        title = "Image 2 (Visualization)"
        json_schema_extra = {
            "shortDescription": "Optional image for visualization"
        }


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
    """
    Eşleşen keypoint'lerin iki görüntü üzerinde
    çizilmiş halini içeren base64 görsel.

    EnableVisualization=False ise veya görüntüler
    sağlanmadıysa value None döner.
    """

    name: Literal["OutputVisualization"] = "OutputVisualization"
    value: Optional[Any] = None
    type: Literal["image"] = "image"

    class Config:
        title = "Output Visualization"
        json_schema_extra = {
            "shortDescription": "Annotated match visualization (base64)"
        }


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


class EnableVisualization(Config):
    """
    Görselleştirmeyi açıp kapatır.

    True olduğunda ve InputImage1 / InputImage2
    sağlandığında, good match'ler iki görüntü
    yan yana konularak çizilir (Roboflow tarzı
    annotated output).

    Default: False
    """

    name: Literal["EnableVisualization"] = "EnableVisualization"
    value: bool = False
    type: Literal["boolean"] = "boolean"
    field: Literal["checkbox"] = "checkbox"

    class Config:
        title = "Enable Visualization"
        json_schema_extra = {
            "shortDescription": "Draw matches on the two images"
        }


class PointRadius(Config):
    """
    Görselleştirmede keypoint'lerin çizileceği
    dairelerin piksel yarıçapı.

    Default: 4
    """

    name: Literal["PointRadius"] = "PointRadius"
    value: int = 4
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Point Radius"
        json_schema_extra = {
            "shortDescription": "Keypoint circle radius (px)"
        }


class LineThickness(Config):
    """
    Görselleştirmede good match bağlantı
    çizgilerinin kalınlığı.

    Default: 2
    """

    name: Literal["LineThickness"] = "LineThickness"
    value: int = 2
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Line Thickness"
        json_schema_extra = {
            "shortDescription": "Match line thickness (px)"
        }


class SiftComparisonTestConfigs(Configs):
    GoodMatchesThreshold: GoodMatchesThreshold
    RatioThreshold: RatioThreshold
    Matcher: Matcher
    EnableVisualization: EnableVisualization
    PointRadius: PointRadius
    LineThickness: LineThickness


# ============================================================
# INPUT / OUTPUT MODELS
# ============================================================

class SiftComparisonTestInputs(Inputs):
    InputSIFTOutput1: InputSIFTOutput1
    InputSIFTOutput2: InputSIFTOutput2
    InputImage1: Optional[InputImage1] = None
    InputImage2: Optional[InputImage2] = None


class SiftComparisonTestOutputs(Outputs):
    OutputDetections: OutputDetections
    OutputVisualization: Optional[OutputVisualization] = None


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
        - (Optional) Annotated visualization image
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