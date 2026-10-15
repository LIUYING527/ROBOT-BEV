import sys
import numpy as np
sys.path.insert(0, '/data/DongBaorong/BEVLOOK/benchmarks/InternNav/r2r_eval')
sys.path.insert(0, '/data/DongBaorong/BEVLOOK/benchmarks/LH-VLN/our_eval')
from run_r2r import make_env
from our_agent import OurAgent

env = make_env('val_unseen', 640, 480, 79)
eps = list(env.episodes)
agent = OurAgent(ckpt='goalnav_r2r_dense_goalmlp_ep5.pt', hfov=79, escape=3)
for ep in eps[:4]:
    env.episodes = [ep]
    obs = env.reset()
    rgb = np.ascontiguousarray(obs['rgb'][:, :, :3])
    ins = ep.instruction.instruction_text
    agent.reset_episode()
    chain = agent.vlm_r2r_parse_chain(rgb, ins)
    print('INSTRUCTION:', ins[:150])
    print('  -> CHAIN:', chain)
