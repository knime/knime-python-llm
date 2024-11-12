import knime.extension as knext

from ..base import ChatModelPortObject, ChatModelPortObjectSpec
from ._utils import get_aws_connection_port_type

aws_connection_port_type = get_aws_connection_port_type()


class AwsBedrockChatModelPortObjectSpec(ChatModelPortObjectSpec):
    def __init__(self, aws_connection_spec):
        self._aws_connection_spec = aws_connection_spec

    @property
    def aws_connection_spec(self):
        return self._aws_connection_spec

    def serialize(self):
        return {
            "aws_connection_spec": self.aws_connection_spec.serialize(),
        }

    @classmethod
    def deserialize(cls, data):
        return cls(
            aws_connection_port_type.spec_class.deserialize(data["aws_connection_spec"])
        )


class AwsBedrockChatModelPortObject(ChatModelPortObject):
    @property
    def spec(self) -> AwsBedrockChatModelPortObjectSpec:
        return super().spec

    def create_model(self, ctx):
        pass


aws_chat_model_port_type = knext.port_type(
    "AWS Bedrock Chat Model",
    AwsBedrockChatModelPortObject,
    AwsBedrockChatModelPortObjectSpec,
)


@knext.node("AWS Bedrock Chat Model Connector", node_type=knext.NodeType.SOURCE)
class AwsBedrockChatModelConnector:
    """Connects to an AWS Bedrock chat model."""

    def configure(self, ctx, aws_connection_spec) -> AwsBedrockChatModelPortObjectSpec:
        return self._create_spec(aws_connection_spec)

    def _create_spec(self, aws_connection_spec) -> AwsBedrockChatModelPortObjectSpec:
        return AwsBedrockChatModelPortObjectSpec(aws_connection_spec)

    def execute(self, ctx, aws_connection) -> AwsBedrockChatModelPortObject:
        return AwsBedrockChatModelPortObject(self._create_spec(aws_connection.spec))
