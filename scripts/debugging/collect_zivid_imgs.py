import zivid
from time import time, sleep
from PIL import Image
import numpy as np


def main():
    np.set_printoptions(precision=5, suppress=True, linewidth=200)

    with zivid.Application() as app:
        with app.connect_camera() as camera:
            settings = zivid.Settings.load('camera_settings.yml')
            for i in range(10):
                with camera.capture(settings) as frame:
                    point_cloud = frame.point_cloud()
                    rgba = point_cloud.copy_data("rgba")
                    rgb = rgba[:, :, :3]
                    img = Image.fromarray(rgb)
                    now = int(time())
                    filename = f"imgs/rgb_{now}.png"
                    img.save(filename)
                    print(f"Saved {filename}")
                    sleep(5)


if __name__ == '__main__':
    main()
