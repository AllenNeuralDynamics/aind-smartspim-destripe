"""
Defines the parameters used in the destriping script
"""
from argschema import ArgSchema
from argschema.fields import Dict, InputDir, Int, Str


class DestripingParams(ArgSchema):
    """
    Destriping parameters
    """

    input_path = InputDir(
        required=True,
        metadata={"description": "Path where the data is located"},
    )

    output_path = Str(
        required=True,
        metadata={"description": "Path where the data will be saved"},
    )

    workers = Int(
        required=False,
        metadata={"description": "Number of workers to do batch processing"},
        dump_default=16,
    )

    chunks = Int(
        required=False,
        metadata={"description": "Number of images each worker will process at a time"},
        dump_default=1,
    )

    output_format = Str(
        required=False,
        metadata={"description": "Output format of filtered images"},
        dump_default=None,
    )
