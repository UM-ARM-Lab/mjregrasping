#!/usr/bin/env python3
import cv2

import mujoco
import numpy as np
from PIL import Image

import rospy
from mjregrasping.homotopy_checker import get_full_h_signature_from_phy
from mjregrasping.scenarios import val_untangle
from mjregrasping.trials import load_trial
from mjregrasping.viz import make_viz


def main():
    rospy.init_node("untangle_example")

    np.set_printoptions(precision=3, suppress=True, linewidth=220)

    gl_ctx = mujoco.GLContext(1280, 720)
    gl_ctx.make_current()

    results_dir = Path("results/Untangle/g")

    scenario = val_untangle
    viz = make_viz(scenario)
    for trial_idx in range(0, 25):
        phy, _, skeletons, mov = load_trial(trial_idx, gl_ctx, scenario, viz)
        viz.viz(phy)

        h, _ = get_full_h_signature_from_phy(skeletons, phy)
        print(h)

        img = mov.r.render(phy.d)

        # draw the H signature onto the image
        img = cv2.putText(img.copy(), str(h), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        img = Image.fromarray(img)
        img.save(f"untangle_example_{trial_idx}.png")


if __name__ == "__main__":
    main()
