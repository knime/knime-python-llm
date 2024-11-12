import knime.extension.nodes as kn

aws_port_type_id = "TODO"  # TODO

aws_icon = ""  # TODO


def aws_connection_available() -> bool:
    return kn.has_port_type_for_id(aws_port_type_id)


def get_aws_connection_port_type() -> kn.PortType:
    return kn.get_port_type_for_id(aws_port_type_id)
