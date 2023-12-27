from azure.ai.ml import command, Input, MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment, BuildContext, AmlCompute, Model
from azure.ai.ml.constants import AssetTypes
import configparser
# read azure credentials from local config file
config = configparser.ConfigParser()
config.read("config.ini")

# Login to configure your workspace and resource group.
credential = DefaultAzureCredential()

# Get a handle to the workspace. You can find the info on the workspace tab on ml.azure.com
ml_client = MLClient(
    credential=credential,
    subscription_id= config["AZURE"]["SubscriptionId"],
    resource_group_name=config["AZURE"]["ResourceGroupName"],
    workspace_name=config["AZURE"]["WorkspaceName"],
)

env_docker_context = Environment(
    build=BuildContext(path="azure/environment"),
    name="yolo_v8_env",
    description="environment to use ultralytics yolo v8 models",
)

ml_client.environments.create_or_update(env_docker_context)

cluster = AmlCompute(
    name="ped-detection-compute",
    type="amlcompute",
    size="Standard_DS11_v2",
    location="westeurope",
    min_instances=0,
    max_instances=2,
    idle_time_before_scale_down=600,
)

ml_client.begin_create_or_update(cluster).result()

# Register model
file_model = Model(
    path="yolov8n.pt",
    type=AssetTypes.CUSTOM_MODEL,
    name="yolov8n",
    description="yolov8n model.",
)
ml_client.models.create_or_update(file_model)

command_job = command(
    inputs=dict(
        data=Input(
            type="uri_folder",
            path="azureml:Pet_detection_sampleset:1",
        ),
        model_to_train=Input(
            type="custom_model",
            path="azureml:yolov8n:1"
        )
    ),
    code=".",
    command="""
    echo "The data asset path is ${{ inputs.data }}" &&
    # Update config.yaml to contain the correct path
    sed -i "s|path:.*$|path: ${{ inputs.data }}|" config.yaml &&
    # Now config.yaml contains the correct path so we can run the training
    yolo detect train data=config.yaml model=yolov8n.pt epochs=1 imgsz=640 seed=42 project=your-experiment name=experiment
    """,
    environment="azureml:yolo_v8_env:3",
    compute="ped-detection-compute",
    experiment_name="ped-detection-sample-experiment",
    display_name="ped-detection-sample-experiment",
)

# Submit the command
ml_client.jobs.create_or_update(command_job)