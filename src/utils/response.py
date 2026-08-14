from sdks.novavision.src.helper.package import PackageHelper
from components.SiftComparisonTest.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    SiftComparisonTest,
    SiftComparisonTestResponse,
    SiftComparisonTestOutputs,
    OutputDetections,
    OutputVisualization,
)


def build_response_sift_comparison_test(context):

    # Executor tarafindan olusturulan
    # Detection listesini OutputDetections
    # modeline koyuyoruz.
    output_detections = OutputDetections(
        value=context.output_detections
    )

    # Visualize aktifse olusturulan base64
    # goruntuyu OutputVisualization modeline koyuyoruz.
    output_visualization = OutputVisualization(
        value=getattr(context, "output_visualization", None)
    )

    # Outputs modelini olusturuyoruz.
    outputs = SiftComparisonTestOutputs(
        OutputDetections=output_detections,
        OutputVisualization=output_visualization
    )

    # Response modelini olusturuyoruz.
    response = SiftComparisonTestResponse(
        outputs=outputs
    )

    # Executor response'unu olusturuyoruz.
    executor = SiftComparisonTest(
        value=response
    )

    # ConfigExecutor olusturuluyor.
    configExecutor = ConfigExecutor(
        value=executor
    )

    # Package config olusturuluyor.
    packageConfigs = PackageConfigs(
        executor=configExecutor
    )

    # PackageHelper ile Novavision package response'u
    # olusturuluyor.
    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=packageConfigs
    )

    return package.build_model(context)