"""把预计算位姿(precompute_feeder_poses.py)+ ZED 米制深度 逐帧喂给 octomap_server。
只依赖 numpy + rclpy(不需 pycolmap/open3d,因位姿已预算)。
每帧发布 tf map->camera + /cloud_in PointCloud2(相机系)。octomap 用 tf 原点做射线投射。

用法(ros2 env 内): python scripts/ros2_cloud_feeder.py [session]
"""
import os, sys, numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

SESS=sys.argv[1] if len(sys.argv)>1 else "111450"
DATA=f"data/{SESS}"; PPF=9000; ZMIN=0.3; ZMAX=12.0


def mat2quat(R):
    """旋转矩阵→四元数 (x,y,z,w)。"""
    t=np.trace(R)
    if t>0:
        s=np.sqrt(t+1.0)*2; w=0.25*s; x=(R[2,1]-R[1,2])/s; y=(R[0,2]-R[2,0])/s; z=(R[1,0]-R[0,1])/s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        s=np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2; w=(R[2,1]-R[1,2])/s; x=0.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]:
        s=np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2; w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
    else:
        s=np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2; w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
    return float(x),float(y),float(z),float(w)


class Feeder(Node):
    def __init__(self):
        super().__init__("cloud_feeder")
        d=np.load(f"outputs/feeder_poses_{SESS}.npz")
        self.names=d["names"]; self.Rot=d["Rot"]; self.Trans=d["Trans"]
        print(f"[feeder] {SESS} 帧{len(self.names)} scale={float(d['scale']):.2f}",flush=True)
        self.i=0
        self.pub=self.create_publisher(point_cloud2.PointCloud2,"/cloud_in",10)
        self.tfb=TransformBroadcaster(self)
        self.timer=self.create_timer(0.15,self.tick)
    def tick(self):
        if self.i>=len(self.names):
            if self.i==len(self.names): print("[feeder] 发完,idle",flush=True); self.i+=1
            return
        name=str(self.names[self.i]); Rot=self.Rot[self.i]; Tr=self.Trans[self.i]; self.i+=1
        npz=f"{DATA}/pointcloud/zed/{os.path.splitext(name)[0]}.npz"
        if not os.path.exists(npz): return
        xyz=np.load(npz)["xyzrgba"].astype(np.float32)[...,:3].reshape(-1,3)/1000.0
        m=np.isfinite(xyz).all(1)&(xyz[:,2]>ZMIN)&(xyz[:,2]<ZMAX); xyz=xyz[m]
        if len(xyz)>PPF: xyz=xyz[np.random.choice(len(xyz),PPF,replace=False)]
        now=self.get_clock().now().to_msg()
        tf=TransformStamped(); tf.header.stamp=now; tf.header.frame_id="map"; tf.child_frame_id="camera"
        tf.transform.translation.x,tf.transform.translation.y,tf.transform.translation.z=float(Tr[0]),float(Tr[1]),float(Tr[2])
        qx,qy,qz,qw=mat2quat(Rot)
        tf.transform.rotation.x,tf.transform.rotation.y,tf.transform.rotation.z,tf.transform.rotation.w=qx,qy,qz,qw
        self.tfb.sendTransform(tf)
        hdr=Header(stamp=now,frame_id="camera")
        self.pub.publish(point_cloud2.create_cloud_xyz32(hdr,xyz.tolist()))
        if self.i%20==0: print(f"[feeder] {self.i}/{len(self.names)}",flush=True)


def main():
    rclpy.init(); n=Feeder()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    n.destroy_node(); rclpy.shutdown()


if __name__=="__main__": main()
