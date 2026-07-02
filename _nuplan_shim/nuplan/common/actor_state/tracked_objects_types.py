from enum import IntEnum
class TrackedObjectType(IntEnum):
    VEHICLE=0; PEDESTRIAN=1; BICYCLE=2; TRAFFIC_CONE=3; BARRIER=4
    CZONE_SIGN=5; GENERIC_OBJECT=6; EGO=7

AGENT_TYPES = {TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE}
