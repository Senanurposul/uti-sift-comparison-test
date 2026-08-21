from sdks.novavision.src.helper.package import PackageHelper

from components.SiftComparisonTest.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    SiftComparisonTest,
    SiftComparisonTestResponse,
    SiftComparisonTestOutputs,
    ImagesMatch,
    GoodMatchesCount,
    KeyPoints1,
    Descriptors1,
    KeyPoints2,
    Descriptors2,
    Visualization1,
    Visualization2,
    VisualizationMatches,
)


def build_response_sift_comparison_test(context):

    outputs = SiftComparisonTestOutputs(
        ImagesMatch=ImagesMatch(
            value=getattr(context, "images_match", False)
        ),
        GoodMatchesCount=GoodMatchesCount(
            value=getattr(context, "good_matches_count", 0)
        ),
        KeyPoints1=KeyPoints1(
            value=getattr(context, "keypoints_1", None)
        ),
        Descriptors1=Descriptors1(
            value=getattr(context, "descriptors_1", None)
        ),
        KeyPoints2=KeyPoints2(
            value=getattr(context, "keypoints_2", None)
        ),
        Descriptors2=Descriptors2(
            value=getattr(context, "descriptors_2", None)
        ),
        Visualization1=Visualization1(
            value=getattr(context, "visualization_1", None)
        ),
        Visualization2=Visualization2(
            value=getattr(context, "visualization_2", None)
        ),
        VisualizationMatches=VisualizationMatches(
            value=getattr(context, "visualization_matches", None)
        ),
    )

    response = SiftComparisonTestResponse(outputs=outputs)
    executor = SiftComparisonTest(value=response)
    config_executor = ConfigExecutor(value=executor)
    package_configs = PackageConfigs(executor=config_executor)

    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=package_configs
    )

    return package.build_model(context)