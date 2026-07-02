from enum import IntEnum
class SemanticMapLayer(IntEnum):
    LANE=0; INTERSECTION=1; WALKWAYS=2; LANE_CONNECTOR=3; ROADBLOCK=4
    STOP_LINE=5; CROSSWALK=6; DRIVABLE_AREA=7; CARPARK_AREA=8
class AbstractMap: pass
class MapObject: pass
